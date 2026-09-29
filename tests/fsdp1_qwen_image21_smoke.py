"""Manual tiny-model FSDP1 smoke. Run only when requested, with torchrun on >=2 GPUs.

Checks different rank layouts, GA4, separate score gradients, shard copying,
and exact checkpoint/RNG/cursor replay at Fake=5 before the Generator update.
This file was authored but NOT executed as part of the implementation.
"""
import argparse
from pathlib import Path

import torch
import torch.distributed as dist

from verl_distill.algorithms.dmd.qwen_image21 import (
    UpdateState, dmd_surrogate, fake_score_loss, renoise, x0_from_velocity,
)
from verl_distill.data.qwen_image21 import RankCursor
from verl_distill.engine.distributed import cleanup_distributed, initialize_distributed
from verl_distill.engine.qwen_checkpoint import inspect_checkpoint, restore_checkpoint, save_checkpoint
from verl_distill.models.qwen_image21.modeling import predict_velocity, require_qwen_runtime
from verl_distill.trainers.qwen_image21 import copy_shards, optimizer_for, wrap_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Fresh shared test directory")
    parser.add_argument("--fresh-process", choices=("save", "restore"))
    parser.add_argument("--attention-backend", choices=("sdpa", "flash2_segmented"), default="sdpa")
    parser.add_argument("--score-offload", action="store_true")
    parser.add_argument("--fake-phase-offload", action="store_true")
    parser.add_argument("--tdm-rollout", action="store_true")
    args = parser.parse_args()
    require_qwen_runtime()
    context = initialize_distributed()
    if context.world_size < 2 or context.device.type != "cuda":
        raise RuntimeError("Use torchrun on at least two CUDA GPUs")
    try:
        from diffusers import QwenImage21Transformer2DModel

        def build(trainable):
            torch.manual_seed(7)
            model = QwenImage21Transformer2DModel(
                num_layers=2, attention_head_dim=16, num_attention_heads=2,
                context_in_dim=16, axes_dims_rope=(4, 6, 6))
            from verl_distill.models.qwen_image21.attention import configure_attention
            configure_attention(model, args.attention_backend)
            return wrap_model(model, context.local_rank, trainable)

        generator, fake, real = build(True), build(True), build(False)
        copy_shards(generator, fake)
        frozen_before = {n: p.detach().clone() for n, p in generator.named_parameters()
                         if not p.requires_grad}
        assert frozen_before
        assert generator._infra_audit["fsdp_blocks"] == 2
        assert generator._infra_audit["gradient_checkpointing"]
        teacher_before = [p.detach().clone() for p in real.parameters()]
        opt_spec = {"lr": 1e-4, "betas": [.9, .95], "weight_decay": .1}
        optimizers = {"generator": optimizer_for(generator, opt_spec),
                      "fake": optimizer_for(fake, opt_spec)}
        models = {"generator": generator, "fake": fake}
        cursor = RankCursor(32, context.rank, context.world_size, 42)
        torch.manual_seed(40 + context.rank)
        references = 2 if context.rank % 2 else 0
        condition = {
            "encoder_hidden_states": torch.randn(1, 2 + references, 16, device=context.device,
                                                 dtype=torch.bfloat16),
            "encoder_hidden_states_mask": (None if args.attention_backend == "flash2_segmented" else
                torch.ones(1, 2 + references, device=context.device, dtype=torch.bool)),
            "img_mask": torch.tensor([[False, False] + [True] * (references + 1)], device=context.device),
            "img_shapes": [[(1, 2, 2)] * (references + 1)],
            "reference_latents": torch.randn(1, 4 * references, 64, device=context.device,
                                             dtype=torch.bfloat16) if references else None}

        from verl_distill.engine.qwen_score_offload import PhaseShardOffload
        gen_residency = PhaseShardOffload(generator) if args.fake_phase_offload else None
        real_residency = PhaseShardOffload(real) if args.fake_phase_offload else None

        def update(fake_phase):
            if gen_residency is not None:
                if fake_phase:
                    real_residency.offload()
                else:
                    gen_residency.load()
                    real_residency.load()
            active = fake if fake_phase else generator
            optimizer = optimizers["fake" if fake_phase else "generator"]
            optimizer.zero_grad(set_to_none=True)
            inactive = generator if fake_phase else fake
            inactive_before = [p.detach().clone() for p in inactive.parameters()]
            for micro in range(4):
                cursor.next_index()
                clean = torch.randn(1, 4, 64, device=context.device)
                s = torch.tensor([.4], device=context.device)
                xt = renoise(clean, torch.randn_like(clean), s)
                if fake_phase and gen_residency is not None:
                    gen_residency.load()
                if args.tdm_rollout:
                    from verl_distill.algorithms.dmd.qwen_rollout import detached_prefix_rollout
                    levels = torch.tensor([1., .94, .87, .77, .63, .4, 0.], device=context.device)
                    y, _ = detached_prefix_rollout(
                        clean, levels, [0, 1, 3, 5][micro],
                        lambda x, t: predict_velocity(generator, x, t, condition),
                        train_exit=not fake_phase)
                else:
                    with torch.set_grad_enabled(not fake_phase):
                        y = x0_from_velocity(xt, predict_velocity(generator, xt, s, condition), s)
                if fake_phase and gen_residency is not None:
                    gen_residency.offload()
                    assert real_residency.on_cpu
                    assert all(p.device.type == "cpu" for p in generator.parameters())
                noise = torch.randn_like(y)
                noisy = renoise(y.detach(), noise, s)
                if fake_phase:
                    loss, _ = fake_score_loss(noisy, predict_velocity(fake, noisy, s, condition), y, noise, s)
                else:
                    with torch.no_grad():
                        vf = predict_velocity(fake, noisy, s, condition)
                        vr = predict_velocity(real, noisy, s, condition)
                        if args.tdm_rollout:
                            from verl_distill.models.qwen_image21.guidance import combine_cfg
                            negative = {**condition, "encoder_hidden_states": torch.zeros_like(condition["encoder_hidden_states"])}
                            vr = combine_cfg(vr, predict_velocity(real, noisy, s, negative), 2.)
                    loss, _ = dmd_surrogate(y, noisy, vf, vr, s)
                from verl_distill.engine.qwen_score_offload import offload_score_shards
                with offload_score_shards((fake, real), enabled=args.score_offload and not fake_phase):
                    (loss / 4).backward()
            assert torch.isfinite(active.clip_grad_norm_(1.))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for p, before in zip(inactive.parameters(), inactive_before, strict=True):
                torch.testing.assert_close(p.cpu(), before.cpu(), rtol=0, atol=0)
                assert p.grad is None

        path = Path(args.output) / "fake5_before_generator"
        expected_path = Path(args.output) / f"expected-rank-{context.rank:05d}.pt"
        if args.fresh_process != "restore":
            # Initialize both optimizer states, then checkpoint a mid-outer boundary.
            update(False)
            for _ in range(5):
                update(True)
            state = UpdateState(reflow_updates=1, fake_updates=5, dmd_initialized=True)
            if gen_residency is not None:
                gen_residency.load()
                real_residency.load()
            save_checkpoint(path, models, optimizers, state, cursor, {"test": True}, {}, rank=context.rank)
            update(False)
            expected = [p.detach().cpu().clone() for p in generator.parameters()]
            expected_cursor = cursor.state_dict()
            torch.save({"parameters": expected, "cursor": expected_cursor,
                        "optimizer": optimizers["generator"].state_dict()}, expected_path)
            if args.fresh_process == "save":
                dist.barrier()
                if context.rank == 0:
                    print("Saved checkpoint and uninterrupted next-update reference", flush=True)
                return
        else:
            reference = torch.load(expected_path, map_location="cpu", weights_only=False)
            expected, expected_cursor = reference["parameters"], reference["cursor"]
            assert not optimizers["generator"].state and not optimizers["fake"].state
        inspect_checkpoint(path, {"test": True})
        restore_checkpoint(path, models, optimizers, cursor, rank=context.rank)
        update(False)
        for p, value in zip(generator.parameters(), expected, strict=True):
            torch.testing.assert_close(p.cpu(), value, rtol=1e-5, atol=1e-6)
        assert cursor.state_dict() == expected_cursor
        reference = torch.load(expected_path, map_location="cpu", weights_only=False)
        actual_opt = optimizers["generator"].state_dict()
        assert actual_opt["param_groups"] == reference["optimizer"]["param_groups"]
        for key, values in actual_opt["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(value.cpu(), reference["optimizer"]["state"][key][name],
                                           rtol=1e-5, atol=1e-6)
        for p, before in zip(real.parameters(), teacher_before, strict=True):
            torch.testing.assert_close(p.cpu(), before.cpu(), rtol=0, atol=0)
            assert not p.requires_grad and p.grad is None
        for n, p in generator.named_parameters():
            if n in frozen_before:
                torch.testing.assert_close(p, frozen_before[n], rtol=0, atol=0)
                assert p.grad is None
        dist.barrier()
        if context.rank == 0:
            print("Tiny Qwen FSDP1 checkpoint replay and GA/phase isolation passed")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
