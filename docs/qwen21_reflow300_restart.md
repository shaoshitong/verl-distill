# Current REFLOW300 restart: DMD2 math, clip5, debug6NFE

Use `configs/recipes/qwen_image21/reflow300_dmd2_clip5_debug6nfe.yaml`. This is the deployed environment-specific configuration; weights/data are not included.

Generator initializes from the REFLOW300-only checkpoint through `init_reflow_from`. Fake and Real initialize from HF. Both DMD optimizers start fresh; Fake/G counters begin at zero. No prior DMD checkpoint or optimizer is reused. The new run starts a new dataset cursor and RNG seed.

## DMD numerical correction

The surrogate follows official DMD2 `main/sd_guidance.py`, `compute_distribution_matching_loss`: reconstruct x0 in FP64, form p_real and p_fake, divide their difference by mean absolute p_real without a denominator clamp, apply `torch.nan_to_num` with default behavior, and compute the final detached-target MSE in FP32. Flow velocity-to-x0 reconstruction remains Qwen-specific. `normalization_eps` is retained for API compatibility but is not used by this corrected surrogate. This does not claim the entire Qwen recipe is identical to SDXL DMD2.

The prior absence of nan_to_num, added denominator clamp and final FP64 MSE were differences from the official numerical implementation. This correction alone has not been proven to explain or eliminate prior visual artifacts. Old DMD results were discarded at the user's request.

## Clip and debug behavior

Generator and Fake `max_grad_norm` are both **5.0**; their learning rates, betas and weight decay remain unchanged. REFLOW optimizer settings remain unchanged because REFLOW training is not rerun.

`runtime.debug_force_last_exit: true` forces the last of six rollout exits for actual training updates whose snapshots are saved: both the scheduled Fake update and corresponding Generator update. This deliberately changes exit sampling on those debug updates. All other updates retain uniform random exits; no extra rollout is added. `forced_debug_exit` is recorded. The setting is part of the checkpoint contract.

FA3, homogeneous batching and asynchronous logging remain enabled. The surrogate contract is `official_dmd2_fp64_reconstruction_nan_to_num_fp32_mse_v1` to prevent silent resumption with the previous loss semantics.
