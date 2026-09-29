> Superseded: the Fake200 resume described below is historical, not the current recommended run. See [the REFLOW300 restart](qwen21_reflow300_restart.md).

# Qwen-Image-2.1 performance and Fake200 resume

This update carries the Qwen implementation running from local snapshot `f5b7a67eb6831437269ba642ef78d57652fd0b73`. Unrelated model code is preserved from the remote main branch.

- Fake scoring uses one conditional forward in both Fake and Generator phases. `fake_cfg_scale` must be 1 (or omitted); it is not an embedding input. Real CFG2 keeps its conditional/unconditional combination.
- `homogeneous_v2` groups cached target/reference/text token lengths, separately by task kind and reference count. The deployed maximum token ratio is 1.10. Padding is explicit in sampler audits.
- `flash3_segmented` requires a separately built, compatible FlashAttention-3 Hopper installation; it does not silently fall back to FA2.
- Scalar JSONL writes use a bounded CPU-only worker queue. Checkpoints and completion flush pending writes; I/O failures propagate.
- FSDP1 remains block-sharded. Eliminating the extra Fake call also removes its parameter gathers. Remaining rollout forwards still gather parameters; this patch does not retain all full model weights on GPU.

## Configuration and metadata

`configs/recipes/qwen_image21/resume200_singlefake_fa3.yaml` is the exact deployed configuration, including environment-specific paths. It is not a portable download recipe. Weights, data, cached conditions, token metadata and FlashAttention binaries are not included.

To build token metadata for your own cache, call `verl_distill.data.qwen_image21.cached_token_metadata(records, cache_root)` and serialize its returned document to the configured `runtime.token_metadata_path`. The loader checks manifest/cache identities and metadata hashes.

## Explicit migration

The provided recipe resumes the specified source checkpoint containing REFLOW300, Fake200 and Generator40. It preserves model/Adam/RNG/update state. The migration validates all 32 saved cursors, preserves the unconsumed suffix, reorders it into homogeneous batches and records padding. This deliberately changes the old Fake external-CFG semantics; it does not undo updates already present in the source.

The one-time `runtime.resume_recipe_migration` field pins the source state SHA and a narrow configuration change. For a subsequent checkpoint produced by this recipe, remove that field and change `runtime.resume_from`; ordinary same-recipe resume restores the new cursor directly. Do not reuse the old source migration marker or reset the cursor.

## Measured validation

32-GPU numerical FA3 checks and full training-step runs passed before deployment. Fixed per-rank sample short tests measured FA3 Fake 25.80s (8 GPUs) / 24.25s (32 GPUs), G 34.40s / 30.78s. The 32-GPU test used asynchronous logging; only one post-warmup outer was measured, so this is not a production-throughput forecast. Native historical RoCE settings did not improve the tested collectives; default NCCL was retained, with actual channels on four 400G HCAs. Cluster addresses and SSH credentials are not part of this repository update.
