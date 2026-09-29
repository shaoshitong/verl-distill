"""The explicit Qwen training contract; no model or accelerator imports."""

import math

DIFFUSERS_REVISION = "80c7ed262aeffbeb43ef13ae04baeb9b84515a69"
MODEL_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
CACHE_SCHEMA = 1


def validate_qwen_config(config):
    model, data, runtime = (config[k] for k in ("model", "data", "runtime"))
    params = config["method"]["params"]
    if type(runtime.get("debug_force_last_exit", False)) is not bool:
        raise ValueError("debug_force_last_exit must be boolean")
    if runtime.get("debug_force_last_exit", False) and params.get("generator_input") != "rollout_dataset_noise":
        raise ValueError("Forced debug exit requires dataset-noise rollout")
    if params.get("dmd_surrogate_dtype", "float64") not in ("float32", "float64"):
        raise ValueError("dmd_surrogate_dtype must be float32 or float64")
    if params.get("reflow_loss", "velocity_mse") not in ("velocity_mse", "tdm_feature_cosine"):
        raise ValueError("Unknown reflow loss")
    if params.get("reflow_loss") == "tdm_feature_cosine":
        if params.get("generator_input") != "rollout_dataset_noise" or params.get("score_flow_shift") != 2.0:
            raise ValueError("Feature REFLOW requires dataset-noise rollout and score shift2")
        if runtime.get("gradient_accumulation_steps") != 1 or runtime.get("reflow_gradient_accumulation_steps") != 1:
            raise ValueError("Feature REFLOW recipe requires GA=1 in both phases")
    fake_cfg = float(params.get("fake_cfg_scale", 1.0))
    if not math.isfinite(fake_cfg) or fake_cfg != 1:
        raise ValueError("Fake is conditional-only; remove fake_cfg_scale or set it to 1")
    teacher_cfg = float(params.get("teacher_cfg_scale", 1.0))
    if not math.isfinite(teacher_cfg) or teacher_cfg < 1:
        raise ValueError("teacher_cfg_scale must be finite and >=1")
    if teacher_cfg > 1 and not data.get("negative_condition_cache"):
        raise ValueError("Teacher CFG requires an immutable same-reference negative cache")
    if runtime.get("init_reflow_from") and (
        runtime.get("resume_from")
        or runtime.get("resume_dmd_fork")
        or runtime.get("resume_refinement")
    ):
        raise ValueError(
            "Weights-only REFLOW initialization and exact resume are mutually exclusive"
        )
    if config["method"]["name"] != "dmd_full":
        raise ValueError("Qwen uses method.name=dmd_full")
    if data.get("format") != "qwen_image21_pairs":
        raise ValueError("Qwen requires data.format=qwen_image21_pairs")
    for key in ("manifest", "condition_cache", "eval_manifest"):
        if not data.get(key):
            raise ValueError(f"data.{key} is required")
    if model.get("revision") != MODEL_REVISION:
        raise ValueError("This adapter is pinned to the Qwen-Image-2.1 model revision")
    if runtime.get("data_ordering", "random") not in ("random", "bucketed_v1", "homogeneous_v2"):
        raise ValueError("Unknown data ordering")
    if runtime.get("data_ordering") == "homogeneous_v2":
        if not runtime.get("token_metadata_path"):
            raise ValueError("homogeneous_v2 requires token_metadata_path")
        for key, default in (("max_token_ratio", 1.25), ("max_resolution_ratio", 1.5)):
            value = float(runtime.get(key, default))
            if not math.isfinite(value) or value < 1:
                raise ValueError(f"{key} must be finite and >=1")
    if int(runtime.get("bucket_batches", 64)) < 1:
        raise ValueError("bucket_batches must be positive")
    if runtime.get("attention_backend", "sdpa") not in ("sdpa", "flash2_segmented", "flash3_segmented"):
        raise ValueError("Unknown attention backend")
    migration = runtime.get("allow_infra_resume_change", False)
    if isinstance(migration, str):
        if migration.lower() not in ("true", "false"):
            raise ValueError("allow_infra_resume_change must be boolean")
        runtime["allow_infra_resume_change"] = migration.lower() == "true"
    elif type(migration) is not bool:
        raise ValueError("allow_infra_resume_change must be boolean")
    if type(runtime.get("allow_condition_cache_rebuild", False)) is not bool:
        raise ValueError("allow_condition_cache_rebuild must be boolean")
    if type(runtime.get("init_allow_data_extension", False)) is not bool:
        raise ValueError("init_allow_data_extension must be boolean")
    plan = runtime.get("resume_refinement")
    if plan is not None:
        if (
            set(plan) != {"source_step", "updates", "ga", "lr_divisor", "replace_training_data"}
            or any(
                type(plan[k]) is not int or plan[k] < 1 for k in ("source_step", "updates", "ga")
            )
            or not math.isfinite(float(plan["lr_divisor"]))
            or float(plan["lr_divisor"]) <= 1
            or type(plan["replace_training_data"]) is not bool
        ):
            raise ValueError("Invalid explicit REFLOW refinement plan")
        if (
            int(params["reflow_updates"]) != plan["source_step"] + plan["updates"]
            or int(runtime["reflow_gradient_accumulation_steps"]) != plan["ga"]
            or runtime.get("reflow_checkpoint_steps") != [int(params["reflow_updates"])]
        ):
            raise ValueError("Refinement config does not match its explicit plan")
    if int(runtime.get("debug_prompt_count", 8)) != 8:
        raise ValueError("Production debug uses exactly the first 8 prompts")
    if int(runtime.get("debug_comparison_steps", 25)) not in (0, 25):
        raise ValueError("Debug comparison must be disabled or use 25 steps")
    if int(runtime.get("micro_batch_size", 0)) != 1:
        raise ValueError("Qwen variable-layout batches require micro_batch_size=1")
    if (
        type(runtime.get("gradient_accumulation_steps")) is not int
        or runtime["gradient_accumulation_steps"] < 1
    ):
        raise ValueError("DMD gradient_accumulation_steps must be a positive integer")
    if int(runtime.get("reflow_gradient_accumulation_steps", 0)) < 1:
        raise ValueError("REFLOW gradient accumulation must be positive")
    if config["distributed"].get("fsdp_backend") != "fsdp1":
        raise ValueError("This trainer implements FSDP1 only")
    for key, value in (
        ("param_dtype", "bfloat16"),
        ("reduce_dtype", "float32"),
        ("buffer_dtype", "float32"),
    ):
        if config["distributed"].get(key) != value:
            raise ValueError(f"distributed.{key} must be {value}")
    if int(params.get("nfe", 0)) != 6 or float(params.get("generator_terminal", 0)) != 0.4:
        raise ValueError("Use six Generator calls with terminal sigma=0.4")
    if params.get("sampling") != "uniform":
        raise ValueError("The implemented six-point training sampler is uniform")
    if params.get("generator_input") not in ("renoised_data", "rollout_dataset_noise"):
        raise ValueError("Generator input must be renoised_data or rollout_dataset_noise")
    if params.get("fake_initialization") not in ("reflow_generator", "hf"):
        raise ValueError("Fake initialization must be reflow_generator or hf")
    if params.get("guidance") != "conditional_only":
        raise ValueError("This recipe uses conditional-only Qwen prediction")
    if params.get("fake_loss") not in ("epsilon_via_x0_capped", "velocity_mse"):
        raise ValueError("Fake loss must be epsilon_via_x0_capped or velocity_mse")
    ratio, fake = int(params.get("fake_updates_per_outer", 0)), int(params.get("fake_updates", 0))
    if ratio != 5 or fake <= 0 or fake % ratio:
        raise ValueError("Fake updates must be positive and divisible by the fixed 5:1 ratio")
    if int(params.get("reflow_updates", -1)) < 0:
        raise ValueError("REFLOW updates must be nonnegative; zero starts DMD from HF")
    lo, hi = float(params["score_sigma_min"]), float(params["score_sigma_max"])
    if not 0 < lo < hi < 1 or int(params.get("score_grid_size", 0)) != 1000:
        raise ValueError("Score time requires a 1000-point grid and 0<min<max<1")
    for key in (
        "fake_max_weight",
        "fake_min_alpha",
        "normalization_eps",
        "fake_loss_weight",
        "generator_loss_weight",
    ):
        value = float(params[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"method.params.{key} must be finite and positive")
    if int(runtime.get("debug_every_fake_updates", 0)) <= 0:
        raise ValueError("debug_every_fake_updates must be positive")
    if int(runtime.get("debug_every_reflow_updates", 0)) < 0:
        raise ValueError("REFLOW debug interval must be nonnegative")
    checkpoints = runtime.get("reflow_checkpoint_steps", [])
    if (
        not isinstance(checkpoints, list)
        or any(type(step) is not int or step < 1 for step in checkpoints)
        or len(set(checkpoints)) != len(checkpoints)
    ):
        raise ValueError("REFLOW checkpoint steps must be unique positive integers")
    if int(runtime.get("save_every_fake_updates", 0)) <= 0:
        raise ValueError("save_every_fake_updates must be positive")
    if int(runtime["save_every_fake_updates"]) % ratio:
        raise ValueError("Save at complete outer boundaries")
    if not runtime.get("output_dir"):
        raise ValueError("runtime.output_dir is required")
    if not runtime.get("debug_prompts_csv"):
        raise ValueError("runtime.debug_prompts_csv is required")
    for phase in ("reflow", "generator", "fake_score"):
        opt = config["optimizer"][phase]
        if opt.get("type") != "adamw" or opt.get("lr_scheduler", "none") != "none":
            raise ValueError("Qwen uses standard AdamW without an LR scheduler")
        if not math.isfinite(float(opt["lr"])) or float(opt["lr"]) <= 0 or len(opt["betas"]) != 2:
            raise ValueError(f"Invalid optimizer: {phase}")
        if not all(0 <= float(b) < 1 for b in opt["betas"]):
            raise ValueError("Adam betas must be in [0,1)")
        if (
            not math.isfinite(float(opt["weight_decay"]))
            or float(opt["weight_decay"]) < 0
            or not math.isfinite(float(opt["max_grad_norm"]))
            or float(opt["max_grad_norm"]) <= 0
        ):
            raise ValueError("Invalid weight decay or gradient clipping")
    if config.get("ema", {}).get("enabled", False):
        raise ValueError("EMA is disabled in this recipe")
