"""Qwen-Image-2.1 DMD with optional REFLOW initialization and FSDP1."""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.utils.checkpoint import checkpoint

from verl_distill.algorithms.dmd.qwen_image21 import (
    UpdateState,
    dmd_surrogate,
    fake_score_loss,
    reflow_loss,
    renoise,
    x0_from_velocity,
)
from verl_distill.algorithms.dmd.qwen_rollout import detached_prefix_rollout
from verl_distill.models.qwen_image21.features import install_feature_forward, predict_features, feature_cosine_loss
from verl_distill.data.qwen_image21 import (
    QwenPairDataset,
    RankCursor,
    atomic_json,
    canonical_hash,
    estimated_training_tokens,
    read_manifest,
    sha256,
)
from verl_distill.engine.distributed import cleanup_distributed, initialize_distributed
from verl_distill.engine.qwen_checkpoint import (
    capture_rng_state,
    collective_call,
    initialize_reflow_weights,
    inspect_checkpoint,
    inspect_reflow_initialization,
    restore_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from verl_distill.engine.qwen_score_offload import PhaseShardOffload, offload_score_shards
from verl_distill.models.qwen_image21.configuration import validate_qwen_config
from verl_distill.models.qwen_image21.guidance import combine_cfg, predict_fake_cfg
from verl_distill.models.qwen_image21.modeling import (
    ConditionStore,
    QwenDecoder,
    QwenSchedule,
    load_transformer,
    model_identity,
    predict_velocity,
    require_qwen_runtime,
)
from verl_distill.models.qwen_image21.negative_cache import NegativeConditionStore
from verl_distill.trainers.qwen_image21_debug import debug_event, tensor_stats

logger = logging.getLogger(__name__)


def wrap_model(model, local_rank, trainable, blocks_only=False):
    # FP32 master shards, BF16 forward weights, FP32 reductions and stored gradients.
    model.float().requires_grad_(trainable)
    if trainable:
        # Frozen conditioning encoders; train the DiT blocks and image projections.
        model.time_text_embed.requires_grad_(False)
        model.txt_in.requires_grad_(False)
        if blocks_only:
            model.requires_grad_(False)
            model.transformer_blocks.requires_grad_(True)
    audit = {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_names": [n for n, p in model.named_parameters() if p.requires_grad],
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "frozen_names": [n for n, p in model.named_parameters() if not p.requires_grad],
    }
    expected_blocks = len(model.transformer_blocks)
    if trainable:
        model.enable_gradient_checkpointing(
            gradient_checkpointing_func=partial(checkpoint, use_reentrant=False)
        )
    wrapped = FSDP(
        model,
        device_id=local_rank,
        use_orig_params=True,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: type(m).__name__ == "QwenImage21TransformerBlock",
        ),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            keep_low_precision_grads=False,
        ),
        forward_prefetch=False,
        limit_all_gathers=True,
    )

    actual_blocks = sum(
        isinstance(m, FSDP) and type(m.module).__name__ == "QwenImage21TransformerBlock"
        for m in wrapped.modules()
    )
    if actual_blocks != expected_blocks:
        raise RuntimeError(f"FSDP block coverage: {actual_blocks}/{expected_blocks}")
    audit.update(
        fsdp_blocks=actual_blocks,
        gradient_checkpointing=model.gradient_checkpointing,
        use_orig_params=True,
        trainable=trainable,
        attention_backend=getattr(model, "_qwen_attention_backend", "sdpa"),
    )
    wrapped._infra_audit = audit
    logger.info("Qwen infrastructure: %s", audit)
    return wrapped


def optimizer_for(model, spec):
    return torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(spec["lr"]),
        betas=tuple(float(x) for x in spec["betas"]),
        weight_decay=float(spec["weight_decay"]),
    )


def copy_shards(source, target):
    # Identical Qwen configs and auto-wrap produce identically ordered flat shards.
    source_parameters, target_parameters = (
        list(source.named_parameters()),
        list(target.named_parameters()),
    )
    if [(n, tuple(p.shape)) for n, p in source_parameters] != [
        (n, tuple(p.shape)) for n, p in target_parameters
    ]:
        raise ValueError("Generator/Fake FSDP layouts differ; cannot initialize Fake by shard copy")
    with torch.no_grad():
        for (_, a), (_, b) in zip(source_parameters, target_parameters, strict=True):
            if a.numel() and a.data_ptr() == b.data_ptr():
                raise ValueError("Generator/Fake unexpectedly share storage")
            b.copy_(a)


def require_finite(value, label, device):
    bad = torch.tensor(
        int(not torch.isfinite(value.detach()).all()), device=device, dtype=torch.int32
    )
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.item():
        raise FloatingPointError(f"Nonfinite {label}; stopping all ranks before optimizer update")


def scalar(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(torch.float64).mean().item()
    return value


def snapshot_tensors(tensors):
    return {k: v.detach().cpu().clone() for k, v in tensors.items() if isinstance(v, torch.Tensor)}


def post_update_actions(state, phase, reflow_total, fake_total, runtime, save, debug):
    """Commit due training state before any optional image generation or VAE load."""
    if phase == "reflow":
        checkpoints = runtime.get("reflow_checkpoint_steps", [reflow_total])
        if state.reflow_updates in checkpoints or state.reflow_updates == reflow_total:
            save(f"reflow_step_{state.reflow_updates:06d}")
        interval = int(runtime.get("debug_every_reflow_updates", 0))
        periodic = interval > 0 and state.reflow_updates % interval == 0
        final = state.reflow_updates == reflow_total and bool(
            runtime.get("debug_after_reflow", True)
        )
        if periodic or final or state.reflow_updates in runtime.get("debug_reflow_steps", []):
            debug()
    if phase == "generator":
        if (
            state.fake_updates % int(runtime["save_every_fake_updates"]) == 0
            or state.fake_updates == fake_total
            or state.fake_updates in runtime.get("early_dmd_checkpoint_steps", [])
        ):
            save(f"fake_step_{state.fake_updates:06d}")
        if state.fake_updates % int(runtime["debug_every_fake_updates"]) == 0:
            debug()


def train(config):
    validate_qwen_config(config)
    require_qwen_runtime()
    context = initialize_distributed(
        timeout_seconds=int(config["distributed"].get("timeout_seconds", 1800))
    )
    if context.device.type != "cuda":
        cleanup_distributed()
        raise RuntimeError("Qwen FSDP1 training requires CUDA; use CPU unit tests for mathematics")
    # FSDP1 requires a process group even for a one-GPU torchrun process.
    if not dist.is_initialized():
        if "MASTER_ADDR" not in os.environ:
            raise RuntimeError("Launch this trainer with torchrun (also for one GPU)")
        dist.init_process_group("nccl", rank=context.rank, world_size=context.world_size)
    try:
        _train(config, context)
    except Exception as exc:
        # Best effort local failure record: never enter a new collective from an
        # arbitrary exception path, because another worker may be in FSDP.
        try:
            atomic_json(
                Path(config["runtime"]["output_dir"]) / f"failure-rank-{context.rank:05d}.json",
                {
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "time": time.time(),
                    "rank": context.rank,
                },
            )
        except OSError:
            pass
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        # An asymmetric failure must not enter NCCL teardown or Python atexit:
        # torchrun observes this exit and terminates the remaining workers.
        os._exit(1)
    else:
        cleanup_distributed()


def _train(config, context):
    from verl_distill.engine.async_jsonl import AsyncJSONL
    writer = AsyncJSONL()
    try:
        return _train_with_writer(config, context, writer)
    finally:
        # Local only: asymmetric I/O failure must never enter a new collective.
        writer.close()


def _train_with_writer(config, context, writer):
    rank, device, world = context.rank, context.device, context.world_size
    runtime, params, data = config["runtime"], config["method"]["params"], config["data"]
    from verl_distill.engine.scaling_bench import settings, update_seed, completion_name
    bench = settings(config)
    seed = int(runtime.get("seed", 42))
    random.seed(seed + rank)
    np.random.seed((seed + rank) % 2**32)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    output = Path(runtime["output_dir"])
    collective_call("create run output", lambda: output.mkdir(parents=True, exist_ok=True))

    # A fresh run never silently adopts another run's output/checkpoints/debug files.
    def reserve_run():
        if rank == 0:
            marker = output / "run.json"
            if marker.exists():
                raise FileExistsError(
                    "Choose a fresh OUTPUT_DIR, including when resuming a checkpoint"
                )
            atomic_json(marker, {"started_at": time.time(), "config": config, "world_size": world})

    collective_call("reserve run", reserve_run)
    dataset = collective_call(
        "load training manifest",
        lambda: QwenPairDataset(data["manifest"], data.get("output_root"), verify_payloads=True),
    )
    evaluation = collective_call("load eval manifest", lambda: read_manifest(data["eval_manifest"]))
    if evaluation["purpose"] != "eval" or len(evaluation["records"]) != 64:
        raise ValueError("Evaluation manifest must contain all 64 CSV prompts")
    hashes = collective_call(
        "hash manifests",
        lambda: {"train": sha256(data["manifest"]), "eval": sha256(data["eval_manifest"])},
    )
    csv_hash = collective_call("hash debug CSV", lambda: sha256(runtime["debug_prompts_csv"]))
    if csv_hash != evaluation["prompts_csv_sha256"]:
        raise ValueError("CSV changed since evaluation manifest was created")
    identity = collective_call(
        "hash model assets on rank 0",
        lambda: model_identity(config["model"]["pretrained_model"]) if rank == 0 else None,
    )
    objects = [identity]
    dist.broadcast_object_list(objects, src=0)
    identity = objects[0]

    # Rank-local model directories must match actual weight/config identities too.
    # Operators may stage one identical path per node; hashing each rank is intentionally
    # conservative rather than trusting directory names as model identity.
    def verify_local_identity():
        local_identity = model_identity(config["model"]["pretrained_model"])
        if local_identity != identity:
            raise ValueError("Rank-local Qwen assets differ")

    collective_call("verify rank-local model assets", verify_local_identity)
    store = collective_call(
        "open condition cache", lambda: ConditionStore(data["condition_cache"], identity, hashes)
    )
    collective_call(
        "check complete condition coverage",
        lambda: [store.validate_record(r) for r in dataset.records + evaluation["records"]],
    )
    negative_store = None
    teacher_cfg = float(params.get("teacher_cfg_scale", 1.0))
    fake_cfg = float(params.get("fake_cfg_scale", 1.0))
    if teacher_cfg > 1:
        negative_store = collective_call(
            "open negative condition cache",
            lambda: NegativeConditionStore(data["negative_condition_cache"], identity, hashes),
        )
        collective_call(
            "check negative coverage",
            lambda: [negative_store.validate_record(r) for r in dataset.records],
        )
    schedule = QwenSchedule(
        config["model"]["pretrained_model"], float(params["generator_terminal"]),
        score_flow_shift=params.get("score_flow_shift"),
    )
    # Paths may move on resume, but numerical/configuration/data identities may not.
    numerical_runtime = {
        k: v
        for k, v in runtime.items()
        if k
        not in (
            "output_dir",
            "resume_from",
            "init_reflow_from",
            "debug_prompts_csv",
            "debug_prompt_count",
            "debug_comparison_steps",
            "allow_infra_resume_change",
            "resume_refinement",
            "resume_dmd_fork",
            "resume_recipe_migration",
            "resume_initial_noise_migration",
            "allow_condition_cache_rebuild",
        )
    }
    contract = {
        "manifest_hashes": hashes,
        "condition_index": sha256(Path(data["condition_cache"]) / "index.json"),
        "csv_sha256": csv_hash,
        "model_identity": canonical_hash(identity),
        "scheduler": canonical_hash(schedule.config),
        "params": params,
        "optimizer": config["optimizer"],
        "runtime": numerical_runtime,
        "distributed": config["distributed"],
        "trainable_policy": (
            "dit_transformer_blocks_only_v1"
            if runtime.get("train_transformer_blocks_only", False)
            else "dit_freeze_time_text_embed_and_txt_in_v1"
        ),
        "fsdp_use_orig_params": True,
    }
    contract["fake_guidance"] = "conditional_only_single_forward_v1"
    contract["dmd_surrogate"] = "official_dmd2_fp64_reconstruction_nan_to_num_fp32_mse_v1"
    if teacher_cfg > 1:
        contract["teacher_guidance"] = {
            "scale": teacher_cfg,
            "negative_prompt": "",
            "reference_images_preserved": True,
            "negative_index": sha256(Path(data["negative_condition_cache"]) / "index.json"),
        }
    all_contracts = [None] * world
    dist.all_gather_object(all_contracts, contract)
    if any(c != contract for c in all_contracts):
        raise ValueError("Ranks resolved different training contracts")
    state = UpdateState()
    resume = runtime.get("resume_from")
    if resume:
        saved = collective_call(
            "inspect checkpoint",
            lambda: inspect_checkpoint(
                resume,
                contract,
                migration=runtime.get("resume_recipe_migration"),
                initial_noise_migration=runtime.get("resume_initial_noise_migration"),
                allow_infra_change=bool(runtime.get("allow_infra_resume_change", False)),
                refinement=runtime.get("resume_refinement"),
                dmd_fork=bool(runtime.get("resume_dmd_fork", False)),
                allow_condition_cache_rebuild=bool(
                    runtime.get("allow_condition_cache_rebuild", False)
                ),
            ),
        )
        state = UpdateState(**saved["updates"])
    initialization = runtime.get("init_reflow_from")
    if initialization:
        initial_saved = collective_call(
            "inspect REFLOW weight initialization",
            lambda: inspect_reflow_initialization(
                initialization,
                contract,
                allow_data_extension=runtime.get("init_allow_data_extension", False),
                current_records=dataset.records,
            ),
        )
        state = UpdateState(**initial_saved["updates"])
    state.validate(int(params["reflow_updates"]), int(params["fake_updates"]))
    costs = (
        [estimated_training_tokens(r) for r in dataset.records]
        if runtime.get("data_ordering", "random") == "bucketed_v1"
        else None
    )
    homogeneous = runtime.get("data_ordering") == "homogeneous_v2"
    if homogeneous:
        from verl_distill.data.qwen_image21 import HomogeneousRankCursor
        from verl_distill.engine.homogeneous_metadata import load_metadata, validate_resume
        metadata, metadata_audit = collective_call("validate homogeneous token metadata",
            lambda: load_metadata(runtime["token_metadata_path"], dataset.records, data["condition_cache"]))
        metadata_identities = [None] * world
        dist.all_gather_object(metadata_identities, metadata_audit)
        if any(identity != metadata_audit for identity in metadata_identities):
            raise ValueError("Ranks resolved different homogeneous token metadata")
        cursor = HomogeneousRankCursor(metadata, rank, world, seed,
            max_token_ratio=float(runtime.get("max_token_ratio", 1.25)),
            max_resolution_ratio=float(runtime.get("max_resolution_ratio", 1.5)))
        if resume:
            if runtime.get("resume_recipe_migration"):
                from verl_distill.engine.homogeneous_metadata import migrate_legacy_checkpoint
                migration_audit = collective_call("audit and migrate 32 saved cursors", lambda:
                    migrate_legacy_checkpoint(resume, cursor,
                        [estimated_training_tokens(r) for r in dataset.records]))
                audits = [None] * world
                dist.all_gather_object(audits, migration_audit["all_rank_cursors_sha256"])
                if len(set(audits)) != 1:
                    raise ValueError("Ranks read different migration cursor states")
                (output / f"resume-migration-rank{rank:05d}.json").write_text(
                    json.dumps(migration_audit, indent=2))
            else:
                validate_resume(saved)
    else:
        cursor = RankCursor(
            len(dataset),
            rank,
            world,
            seed,
            costs=costs,
            bucket_batches=int(runtime.get("bucket_batches", 64)),
            allow_ordering_migration=bool(runtime.get("allow_infra_resume_change", False)),
        )
    reset_data_cursor = bool(
        resume
        and saved["contract"]["manifest_hashes"]["train"] != contract["manifest_hashes"]["train"]
    )
    if homogeneous and reset_data_cursor:
        validate_resume(saved, reset_data_cursor=True)
    if reset_data_cursor:

        def verify_dataset_extension():
            previous = read_manifest(saved["config"]["data"]["manifest"])
            current_rows = {(r["kind"], r["id"]): r for r in dataset.records}
            if any(current_rows.get((r["kind"], r["id"])) != r for r in previous["records"]):
                raise ValueError("Refinement data must include every old record unchanged")

        collective_call("verify explicitly authorized dataset extension", verify_dataset_extension)
    if resume and saved["contract"] != contract:
        collective_call(
            "record explicit infrastructure migration",
            lambda: atomic_json(
                output / "infra_migration.json",
                {
                    "checkpoint": str(resume),
                    "old_contract": saved["contract"],
                    "new_contract": contract,
                    "data_policy": (
                        "restore G/Fake models and optimizers, cursor and RNG; fresh rollout noise and checkpoint interval only"
                        if runtime.get("resume_initial_noise_migration")
                        else "restore model, optimizer, cursor and RNG; regenerated conditions for identical manifests"
                        if runtime.get("allow_condition_cache_rebuild", False)
                        else "restore model, cursor and RNG; fresh DMD optimizers"
                        if runtime.get("resume_dmd_fork", False)
                        else "new dataset epoch zero; restore optimizer and RNG"
                        if reset_data_cursor
                        else "only rebucket unconsumed epoch suffix; restore optimizer and RNG"
                    ),
                },
            )
            if rank == 0
            else None,
        )
    generator = wrap_model(
        load_transformer(
            config["model"]["pretrained_model"], runtime.get("attention_backend", "sdpa")
        ),
        context.local_rank,
        True,
        blocks_only=runtime.get("train_transformer_blocks_only", False),
    )
    if initialization:
        initialize_reflow_weights(initialization, generator)
        collective_call(
            "record weights-only initialization",
            lambda: atomic_json(
                output / f"initialization-rank-{rank:05d}.json",
                {
                    "source": initialization,
                    "source_world_size": initial_saved["world_size"],
                    "target_world_size": world,
                    "source_state_sha256": sha256(Path(initialization) / "state.json"),
                    "optimizer": "fresh",
                    "cursor": "new epoch zero",
                    "rng": "new run seed plus rank",
                    "updates": state.state_dict(),
                },
            ),
        )
    generator.train()
    collective_call(
        "write infrastructure audit",
        lambda: atomic_json(output / f"infra-rank-{rank:05d}.json", generator._infra_audit),
    )
    feature_reflow = params.get("reflow_loss", "velocity_mse") == "tdm_feature_cosine"
    feature_teacher = None
    if feature_reflow and state.reflow_updates < int(params["reflow_updates"]):
        teacher_model = load_transformer(config["model"]["pretrained_model"], runtime.get("attention_backend", "sdpa"))
        install_feature_forward(teacher_model)
        teacher_model.enable_gradient_checkpointing(gradient_checkpointing_func=partial(checkpoint, use_reentrant=False))
        feature_teacher = wrap_model(teacher_model, context.local_rank, False)
        feature_teacher.eval()
        del teacher_model
    fake = real = fake_optimizer = None
    gen_spec = config["optimizer"]["generator" if state.dmd_initialized else "reflow"]
    gen_optimizer = optimizer_for(generator, gen_spec)

    def load_scores():
        fake_model = wrap_model(
            load_transformer(
                config["model"]["pretrained_model"], runtime.get("attention_backend", "sdpa")
            ),
            context.local_rank,
            True,
            blocks_only=runtime.get("train_transformer_blocks_only", False),
        )
        real_model = wrap_model(
            load_transformer(
                config["model"]["pretrained_model"], runtime.get("attention_backend", "sdpa")
            ),
            context.local_rank,
            False,
        )
        fake_model.train()
        real_model.eval()
        collective_call(
            "write score infrastructure",
            lambda: atomic_json(
                output / f"score-infra-rank-{rank:05d}.json",
                {
                    "fake": fake_model._infra_audit,
                    "real": real_model._infra_audit,
                    "teacher_cfg_scale": teacher_cfg,
                },
            ),
        )
        return fake_model, real_model

    if state.dmd_initialized:
        fake, real = load_scores()
        fake_optimizer = optimizer_for(fake, config["optimizer"]["fake_score"])
    models = {"generator": generator}
    optimizers = {"generator": gen_optimizer}
    if fake is not None:
        models["fake"] = fake
        optimizers["fake"] = fake_optimizer
    if resume:
        if saved["models"] != list(models):
            raise ValueError("Checkpoint phase/model inventory mismatch")
        restore_checkpoint(
            resume,
            models,
            optimizers,
            cursor,
            rank=rank,
            reset_data_cursor=reset_data_cursor,
            migrated_cursor=bool(runtime.get("resume_recipe_migration")),
            model_only=bool(runtime.get("resume_dmd_fork", False)) and not state.dmd_initialized,
        )
        # DCP restores optimizer param_groups, including the OLD LR. Apply the
        # validated new recipe only after loading; retain Adam moments and step.
        loaded_lrs = [float(group["lr"]) for group in gen_optimizer.param_groups]
        for group in gen_optimizer.param_groups:
            group["lr"] = float(gen_spec["lr"])
        collective_call(
            "record restored optimizer learning rate",
            lambda: atomic_json(
                output / f"resume-optimizer-rank-{rank:05d}.json",
                {
                    "loaded_lrs": loaded_lrs,
                    "effective_lrs": [g["lr"] for g in gen_optimizer.param_groups],
                    "optimizer_state_entries": len(gen_optimizer.state),
                    "fake_optimizer_state_entries": len(fake_optimizer.state) if fake_optimizer is not None else 0,
                    "generator_adam_steps": sorted({float(v["step"].item()) if torch.is_tensor(v["step"]) else float(v["step"]) for v in gen_optimizer.state.values() if "step" in v}),
                    "fake_adam_steps": sorted({float(v["step"].item()) if torch.is_tensor(v["step"]) else float(v["step"]) for v in fake_optimizer.state.values() if "step" in v}) if fake_optimizer is not None else [],
                    "reset_data_cursor": reset_data_cursor,
                    "updates": state.state_dict(),
                },
            ),
        )
    del models, optimizers  # Do not retain the REFLOW AdamW state after phase transition.
    decoder = None  # VAE initialized only for the first scheduled debug event.
    reflow_total, fake_total = int(params["reflow_updates"]), int(params["fake_updates"])
    debug_snapshots = {}
    if homogeneous:
        if cursor._indices is None:
            cursor._indices = cursor._permutation()
        collective_call("record homogeneous sampler audit", lambda: atomic_json(
            output / f"sampler-rank-{rank:05d}.json",
            {"ordering": "homogeneous_v2", **metadata_audit,
             "epoch": cursor.epoch, "position": cursor.position,
             "epoch_stats": cursor.epoch_stats,
             "max_token_ratio": cursor.max_token_ratio,
             "max_resolution_ratio": cursor.max_resolution_ratio}))
    last_checkpoint = None
    log_path = output / "logs" / f"rank-{rank:05d}.jsonl"
    collective_call("create logs", lambda: log_path.parent.mkdir(parents=True, exist_ok=True))

    def save(name):
        nonlocal last_checkpoint
        collective_call("flush scalar logs before checkpoint", writer.flush)
        if rank == 0:
            logger.info("Saving checkpoint %s before any scheduled debug", name)
        cp_models, cp_optimizers = {"generator": generator}, {"generator": gen_optimizer}
        if state.dmd_initialized:
            cp_models["fake"], cp_optimizers["fake"] = fake, fake_optimizer
        last_checkpoint = save_checkpoint(
            output / "checkpoints" / name,
            cp_models,
            cp_optimizers,
            state,
            cursor,
            contract,
            config,
            rank=rank,
        )
        if rank == 0:
            logger.info("Checkpoint committed: %s", last_checkpoint)

    def get_decoder():
        nonlocal decoder
        if decoder is None:
            rng = capture_rng_state()
            try:
                decoder = collective_call(
                    "load debug VAE",
                    lambda: QwenDecoder(config["model"]["pretrained_model"], device),
                )
            finally:
                restore_rng_state(rng)
        return decoder

    generator_residency = real_residency = None
    while True:
        writer.check()
        phase = state.next_phase(reflow_total, fake_total)
        if phase == "complete":
            break
        if phase != "reflow" and not state.dmd_initialized:
            # Only the fresh phase transition copies weights. Resume never repeats it.
            # Release REFLOW Adam moments before loading the two score models;
            # the previous loop's `optimizer` may still reference the old optimizer.
            gen_optimizer.state.clear()
            gen_optimizer = optimizer_for(generator, config["optimizer"]["generator"])
            if feature_teacher is not None:
                del feature_teacher
                feature_teacher = None
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            fake, real = load_scores()
            fake_initialization = params.get("fake_initialization", "reflow_generator")
            if fake_initialization == "reflow_generator":
                collective_call(
                    "initialize independent Fake shards", lambda: copy_shards(generator, fake)
                )
            elif fake_initialization != "hf":
                raise ValueError(f"Unsupported fake_initialization: {fake_initialization}")
            collective_call(
                "record DMD initialization",
                lambda: atomic_json(
                    output / f"dmd-initialization-rank-{rank:05d}.json",
                    {
                        "generator_checkpoint": str(
                            resume
                            or initialization
                            or ("hf" if reflow_total == 0 else "in-process REFLOW")
                        ),
                        "fake_initialization": fake_initialization,
                        "fake_cfg_scale": fake_cfg,
                        "real_cfg_scale": teacher_cfg,
                        "real_initialization": "hf",
                        "hf_model": config["model"]["pretrained_model"],
                        "generator_lr": config["optimizer"]["generator"]["lr"],
                        "fake_lr": config["optimizer"]["fake_score"]["lr"],
                        "fresh_dmd_optimizers": True,
                    },
                ),
            )
            fake_optimizer = optimizer_for(fake, config["optimizer"]["fake_score"])
            state.dmd_initialized = True
        if phase != "reflow" and runtime.get("offload_inactive_for_fake", False):
            if generator_residency is None:
                generator_residency = PhaseShardOffload(generator)
                real_residency = PhaseShardOffload(real)
            if phase == "fake_score":
                real_residency.offload()  # Remains on CPU for all five Fake updates.
            else:
                real_residency.load()
                generator_residency.load()
        before = state.state_dict()
        ga = int(
            runtime["reflow_gradient_accumulation_steps"]
            if phase == "reflow"
            else runtime["gradient_accumulation_steps"]
        )
        all_exits = phase != "reflow" and params.get("dmd_rollout_loss_mode", "random_exit") == "all_exits"
        exits_per_sample = 6 if all_exits else 1
        is_fake = phase == "fake_score"
        active = fake if is_fake else generator
        optimizer = fake_optimizer if is_fake else gen_optimizer
        spec = config["optimizer"][phase]
        optimizer.zero_grad(set_to_none=True)
        from verl_distill.engine.debug_exit import capture_update, force_debug_exit
        capture = capture_update(phase, state.fake_updates,
            int(runtime["debug_every_fake_updates"]), benchmark=bench is not None)
        captures, micro_logs = [], []
        hit_counts = [0] * 6
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        from verl_distill.engine.bounded_profile import BoundedProfile
        step_profiler = BoundedProfile(output, rank, phase, before, {"generator": generator, "fake": fake, "real": real})
        if bench is not None:
            bench_seed = update_seed(bench, phase, before)
            random.seed(bench_seed)
            np.random.seed(bench_seed)
            torch.manual_seed(bench_seed)
            torch.cuda.manual_seed_all(bench_seed)
        # A bounded first-shard probe exposes accidental no-op/phase leakage in logs.
        probe_param = next(p for p in active.parameters() if p.requires_grad and p.numel())
        probe_before = probe_param.detach().flatten()[:1024].clone()
        for ga_index in range(ga):
            fetched = time.monotonic()
            item = collective_call("load next data pair", lambda: dataset[bench["sample_index"] if bench is not None else cursor.next_index()])
            row = item["record"]
            condition = collective_call("load sample condition", lambda: store.get(row, device))
            target_tokens = item["clean"].shape[1]
            reference_tokens = (
                condition["reference_latents"].shape[1]
                if condition["reference_latents"] is not None
                else 0
            )
            vlm_length = condition["encoder_hidden_states"].shape[1]
            text_tokens = int((~condition["img_mask"][0, :vlm_length]).sum().item())
            atomic_json(
                output / f"active-microbatch-rank-{rank:05d}.json",
                {
                    "phase": phase,
                    "updates_before": before,
                    "ga_index": ga_index,
                    "sample_id": row["id"],
                    "reference_count": len(row["reference_images"]),
                    "target_tokens": target_tokens,
                    "reference_tokens": reference_tokens,
                    "text_tokens": text_tokens,
                    "total_tokens": target_tokens + reference_tokens + text_tokens,
                    "time": time.time(),
                },
            )
            clean = item["clean"].to(device)
            levels = schedule.levels(row["height"], row["width"], device=device)
            rollout_input = (phase == "reflow" and feature_reflow) or (phase != "reflow" and params["generator_input"] == "rollout_dataset_noise")
            exit_draw = torch.randint(6, (1,), device=device)
            if bench is not None:
                exit_draw.fill_(bench["exit_index"])
            forced_debug_exit = force_debug_exit(capture,
                runtime.get("debug_force_last_exit", False), rollout_input)
            if all_exits:
                forced_debug_exit = False
            if forced_debug_exit:
                exit_draw.fill_(len(levels) - 2)
            if rollout_input:
                # FSDP all-gathers must have the same forward-call count on all ranks.
                dist.broadcast(exit_draw, src=0)
            step_index = int(exit_draw.item())
            sigma = levels[step_index : step_index + 1]
            from verl_distill.algorithms.dmd.qwen_rollout import initial_rollout_noise
            noise = initial_rollout_noise(clean, item["noise"], phase=phase,
                rollout_input=rollout_input, source=params.get("rollout_initial_noise", "dataset"))
            gen_input = None if rollout_input else renoise(clean, noise, sigma)
            data_seconds = time.monotonic() - fetched
            rollout_state = noise.detach().float() if all_exits else None
            exit_indices = range(len(levels) - 1) if all_exits else (step_index,)
            if all_exits and len(levels) - 1 != exits_per_sample:
                raise ValueError("all_exits currently requires the six-step schedule")
            for step_index in exit_indices:
                hit_counts[step_index] += 1
                sigma = levels[step_index:step_index + 1]
                forward_start = time.monotonic()
                tensors = {
                    "generator_input": gen_input,
                    "generator_input_noise": noise,
                    "generator_sigma": sigma,
                }
                fake_offload_stats = {}
                if phase == "reflow" and feature_reflow:
                    generated, gen_input = detached_prefix_rollout(
                        noise, levels, step_index,
                        lambda x, t: predict_velocity(generator, x, t, condition), train_exit=True,
                    )
                    score_sigma = schedule.score_sigma(row["height"], row["width"],
                        float(params["score_sigma_min"]), float(params["score_sigma_max"]), device)
                    with torch.no_grad():
                        target_features = predict_features(feature_teacher, renoise(clean, noise, score_sigma), score_sigma, condition)
                    predicted_features = predict_features(feature_teacher, renoise(generated, noise, score_sigma), score_sigma, condition)
                    loss = feature_cosine_loss(predicted_features, target_features)
                    tensors.update(generator_input=gen_input, generator_x0=generated, score_sigma=score_sigma)
                    aux = {}
                    del predicted_features, target_features
                elif phase == "reflow":
                    velocity = predict_velocity(generator, gen_input, sigma, condition)
                    loss = reflow_loss(velocity, clean, noise)
                    tensors.update(
                        generator_x0=x0_from_velocity(gen_input, velocity, sigma),
                        reflow_target_flow=noise - clean,
                        predicted_velocity=velocity,
                    )
                    aux = {}
                else:
                    if is_fake and generator_residency is not None and (not all_exits or step_index == 0):
                        fake_offload_stats["generator_reload_seconds"] = generator_residency.load()
                    if all_exits:
                        from verl_distill.algorithms.dmd.qwen_rollout import detached_rollout_step
                        generated, gen_input, rollout_state = detached_rollout_step(
                            rollout_state, levels, step_index,
                            lambda x, t: predict_velocity(generator, x, t, condition),
                            train_exit=not is_fake,
                        )
                        tensors["generator_input"] = gen_input
                    elif rollout_input:
                        generated, gen_input = detached_prefix_rollout(
                            noise, levels, step_index,
                            lambda x, t: predict_velocity(generator, x, t, condition),
                            train_exit=not is_fake,
                        )
                        tensors["generator_input"] = gen_input
                    else:
                        with torch.set_grad_enabled(not is_fake):
                            gen_velocity = predict_velocity(generator, gen_input, sigma, condition)
                            generated = x0_from_velocity(gen_input, gen_velocity, sigma)
                    if is_fake and generator_residency is not None and (not all_exits or step_index == len(levels) - 2):
                        fake_offload_stats.update(generator_residency.offload())
                        fake_offload_stats["real_on_cpu"] = real_residency.on_cpu
                    require_finite(generated, "Generator output", device)
                    score_sigma = schedule.score_sigma(
                        row["height"],
                        row["width"],
                        float(params["score_sigma_min"]),
                        float(params["score_sigma_max"]),
                        device,
                    )
                    score_noise = torch.randn_like(generated)
                    noisy = renoise(generated.detach(), score_noise, score_sigma)
                    if is_fake:
                        fake_velocity = predict_fake_cfg(fake, noisy, score_sigma, condition, scale=fake_cfg)
                        loss, aux = fake_score_loss(
                            noisy,
                            fake_velocity,
                            generated,
                            score_noise,
                            score_sigma,
                            max_weight=float(params["fake_max_weight"]),
                            min_alpha=float(params["fake_min_alpha"]),
                            loss_weight=float(params["fake_loss_weight"]),
                            loss_mode=params["fake_loss"],
                        )
                        tensors["fake_velocity"] = fake_velocity
                    else:
                        with torch.no_grad():
                            fake_velocity = predict_fake_cfg(fake, noisy, score_sigma, condition, scale=fake_cfg)
                            conditional_real = predict_velocity(real, noisy, score_sigma, condition)
                            if teacher_cfg > 1:
                                negative_condition = collective_call(
                                    "load Teacher negative condition",
                                    lambda: negative_store.get(row, condition, device),
                                )
                                negative_real = predict_velocity(
                                    real, noisy, score_sigma, negative_condition
                                )
                                real_velocity = combine_cfg(
                                    conditional_real, negative_real, teacher_cfg
                                )
                                tensors.update(
                                    real_conditional_velocity=conditional_real,
                                    real_negative_velocity=negative_real,
                                )
                                del negative_condition
                            else:
                                real_velocity = conditional_real
                        loss, aux = dmd_surrogate(
                            generated,
                            noisy,
                            fake_velocity,
                            real_velocity,
                            score_sigma,
                            loss_weight=float(params["generator_loss_weight"]),
                            normalization_eps=float(params["normalization_eps"]),
                            dtype=params.get("dmd_surrogate_dtype", "float64"),
                        )
                        tensors.update(fake_velocity=fake_velocity, real_velocity=real_velocity)
                    tensors.update(
                        generator_x0=generated,
                        score_noise=score_noise,
                        score_noisy_latent=noisy,
                        score_sigma=score_sigma,
                        **aux,
                    )
                require_finite(loss, f"{phase} loss", device)
                forward_seconds = time.monotonic() - forward_start
                # Synchronize/reduce each microbatch: avoid no_sync's unsharded-gradient peak.
                backward_start = time.monotonic()
                with offload_score_shards(
                    (fake, real) if phase == "generator" else (),
                    enabled=phase == "generator"
                    and runtime.get("offload_scores_for_generator_backward", False),
                ) as offload_stats:
                    (loss / (ga * exits_per_sample)).backward()
                    torch.cuda.synchronize(device)
                backward_seconds = time.monotonic() - backward_start
                if phase == "reflow" and feature_reflow:
                    if any(p.requires_grad or p.grad is not None for p in feature_teacher.parameters()):
                        raise RuntimeError("Frozen feature teacher unexpectedly acquired parameter gradients")
                meta = {
                    "phase": phase,
                    "updates_before": before,
                    "ga_index": ga_index,
                    "rank": rank,
                    "ga": ga,
                    "generator_step_index": step_index,
                    "forced_debug_exit": forced_debug_exit,
                    "dmd_rollout_loss_mode": "all_exits" if all_exits else "random_exit",
                    "loss_accumulation_divisor": ga * exits_per_sample,
                    "generator_sigma": sigma.item(),
                    "generator_sigma_grid": levels.tolist(),
                    "sample_id": row["id"],
                    "kind": row["kind"],
                    "width": row["width"],
                    "height": row["height"],
                    "reference_count": len(row["reference_images"]),
                    "prompt_characters": len(row["prompt"]),
                    "target_tokens": clean.shape[1],
                    "estimated_sample_tokens": estimated_training_tokens(row),
                    "img_shapes": condition["img_shapes"],
                    "vlm_sequence_length": condition["encoder_hidden_states"].shape[1],
                    "lr": float(spec["lr"]),
                    "loss": loss.detach().item(),
                    "data_seconds": data_seconds if not all_exits or step_index == 0 else 0.0,
                    "forward_seconds": forward_seconds,
                    "backward_seconds": backward_seconds,
                }
                if phase in ("fake_score", "generator"):
                    meta["fake_score_forward_branches"] = 1
                    meta["real_score_forward_branches"] = (2 if teacher_cfg > 1 else 1) if phase == "generator" else 0
                    meta["fake_guidance"] = "conditional_only"
                if phase == "generator":
                    meta["fake_cfg_scale"] = fake_cfg
                    meta["teacher_cfg_scale"] = teacher_cfg
                    meta["teacher_negative_prompt"] = "" if teacher_cfg > 1 else None
                    meta["teacher_reference_images_preserved"] = teacher_cfg > 1
                meta.update(
                    actual_total_tokens=target_tokens + reference_tokens + text_tokens,
                    reference_tokens=reference_tokens,
                    text_tokens=text_tokens,
                )
                if phase == "reflow" and feature_reflow:
                    meta.update(reflow_loss="tdm_feature_cosine", initial_noise_source="dataset",
                        generator_nfe=step_index + 1, prefix_detached=True,
                        rollout_path=levels[:step_index + 1].tolist() + [0.0],
                        score_sigma=score_sigma.item(), score_flow_shift=params["score_flow_shift"],
                        feature_noise_source="dataset", feature_layer="last_transformer_block_pre_norm_proj_unpatch",
                        feature_reduction="channel_cosine_mean_target_tokens", real_frozen=True)
                if fake_offload_stats:
                    meta["fake_phase_offload"] = fake_offload_stats
                if offload_stats:
                    meta["score_offload"] = offload_stats
                if phase != "reflow":
                    meta["generator_input_mode"] = params["generator_input"]
                    meta["initial_noise_source"] = "dataset" if rollout_input and params.get("rollout_initial_noise", "dataset") == "dataset" else "fresh_gaussian"
                    meta["generator_nfe"] = step_index + 1 if rollout_input else 1
                    meta["prefix_detached"] = rollout_input
                    meta["rollout_path"] = levels[:step_index + 1].tolist() + [0.0] if rollout_input else None
                    meta["score_flow_shift"] = params.get("score_flow_shift", "dynamic")
                    meta["score_sigma"] = score_sigma.item()
                for key in (
                    "epsilon_mse",
                    "velocity_mse",
                    "x0_mse",
                    "velocity_weight",
                    "weight_raw",
                    "weight",
                    "cap_fraction",
                    "loss_unweighted",
                    "loss_weighted",
                    "denominator",
                ):
                    if key in aux:
                        meta[key] = scalar(aux[key])
                meta["tensor_stats"] = {
                    k: tensor_stats(tensors[k])
                    for k in ("generator_x0", "fake_x0", "real_x0", "diff_x0", "normalized_direction")
                    if k in tensors
                }
                if "normalized_direction" in tensors:
                    meta["direction_rms"] = (
                        tensors["normalized_direction"].detach().square().mean().sqrt().item()
                    )
                micro_logs.append(meta)
                if rank == 0 and phase == "reflow" and ga > 4 and (ga_index + 1) % 4 == 0:
                    logger.info(
                        "REFLOW accumulation update=%d microbatch=%d/%d local_loss=%.6g",
                        before["reflow_updates"] + 1,
                        ga_index + 1,
                        ga,
                        meta["loss"],
                    )
                if capture and (not all_exits or step_index == len(levels) - 2):
                    captures.append(
                        {
                            "record": copy.deepcopy(row),
                            "metadata": meta,
                            "tensors": snapshot_tensors(tensors),
                        }
                    )
                del tensors, loss, aux
            del condition
        # All FSDP gradient shards are FP32; detect nonfinite globally before AdamW.
        local_bad = any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in active.parameters()
        )
        require_finite(
            torch.tensor(float("nan") if local_bad else 0.0, device=device), "gradients", device
        )
        norm = active.clip_grad_norm_(float(spec["max_grad_norm"]))
        require_finite(norm, "global gradient norm", device)
        optimizer.step()
        state.advance(phase)
        probe_delta = (probe_param.detach().flatten()[:1024] - probe_before).float()
        global_loss = torch.tensor(sum(m["loss"] for m in micro_logs) / (ga * exits_per_sample), device=device)
        dist.all_reduce(global_loss)
        global_loss /= world
        hits = torch.tensor(hit_counts, device=device)
        dist.all_reduce(hits)
        losses_by_sigma = torch.tensor(
            [
                sum(m["loss"] for m in micro_logs if m["generator_step_index"] == index)
                for index in range(6)
            ],
            device=device,
        )
        dist.all_reduce(losses_by_sigma)
        # Separate score-sigma bins from the six Generator input timesteps.
        score_bins = None
        if phase != "reflow":
            from verl_distill.trainers.qwen_image21_metrics import score_bin_report, score_bin_sums

            score_bins = score_bin_sums(micro_logs, device=device)
            dist.all_reduce(score_bins)
            score_bins = score_bin_report(score_bins)
        log = {
            "time": time.time(),
            "phase": phase,
            **state.state_dict(),
            "outer": state.outer,
            "microbatches": micro_logs,
            "ga": ga,
            "effective_samples": ga * world,
            "global_loss": global_loss.item(),
            "sigma_hits_global": hits.tolist(),
            "score_sigma_bins_global": score_bins,
            "sigma_loss_mean_global": [
                losses_by_sigma[i].item() / hits[i].item() if hits[i].item() else None
                for i in range(6)
            ],
            "grad_norm_before_clip": norm.item(),
            "grad_norm_after_clip_bound": min(norm.item(), float(spec["max_grad_norm"])),
            "probe_delta_abs_mean": probe_delta.abs().mean().item() if probe_delta.numel() else 0.0,
            "lr": float(spec["lr"]),
            "seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "nonfinite_count": 0,
        }

        if bench is not None:
            log["scaling_benchmark"] = {**bench, "update_seed": bench_seed,
                "world_size": world, "comparison": "fixed per-rank work; weak scaling"}
        diagnostics_started = time.monotonic()
        writer.enqueue(log_path, log)
        if bench is not None:
            # Separate record: measures bounded CPU enqueue, not background disk completion.
            diagnostics_finished = time.monotonic()
            timing = {"phase": phase, **state.state_dict(),
                "step_seconds_before_log": log["seconds"],
                "scalar_log_enqueue_seconds": diagnostics_finished - diagnostics_started,
                "step_through_log_seconds": diagnostics_finished - started}
            writer.enqueue(log_path.parent / f"timing-rank-{rank:05d}.jsonl", timing)
        if step_profiler.selected:
            collective_call("export bounded CUDA profile", lambda: step_profiler.finish(log))
        if rank == 0:
            logger.info(
                "Qwen %s reflow=%d fake=%d gen=%d loss=%.6g grad_norm=%.6g",
                phase,
                state.reflow_updates,
                state.fake_updates,
                state.generator_updates,
                log["global_loss"],
                log["grad_norm_before_clip"],
            )
        # Do not keep full gradients resident during validation or next phase loading.
        optimizer.zero_grad(set_to_none=True)
        if capture:
            debug_snapshots[phase] = captures

        def run_debug():
            if phase == "generator" and set(debug_snapshots) != {"fake_score", "generator"}:
                raise RuntimeError("Missing actual GA snapshots for scheduled debug event")
            debug_event(
                output,
                state,
                generator,
                {} if phase == "reflow" else debug_snapshots,
                evaluation,
                store,
                schedule,
                get_decoder(),
                device,
                rank,
                world,
                data.get("reference_root") or dataset.document["reference_root"],
                runtime["debug_prompts_csv"],
                contract,
                prompt_count=int(runtime.get("debug_prompt_count", 8)),
                comparison_steps=int(runtime.get("debug_comparison_steps", 25)),
            )
            debug_snapshots.clear()

        if bench is None:
            post_update_actions(state, phase, reflow_total, fake_total, runtime, save, run_debug)
    state.validate(reflow_total, fake_total)
    if (state.reflow_updates, state.fake_updates, state.generator_updates) != (
        reflow_total,
        fake_total,
        fake_total // 5,
    ):
        raise RuntimeError("Training ended before all required optimizer updates")
    collective_call("flush scalar logs before completion", writer.flush)
    collective_call(
        "record training completion",
        lambda: atomic_json(
            output / completion_name(bench),
            {
                "updates": state.state_dict(),
                "checkpoint": None if bench is not None else str(last_checkpoint or resume),
                **({"scaling_benchmark": bench} if bench is not None else {}),
                "contract": contract,
            },
        )
        if rank == 0
        else None,
    )
