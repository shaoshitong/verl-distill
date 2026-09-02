# Configuration mapping

The OPD+GAN public recipe preserves the selected source launcher's effective
hyperparameters. Experiment identifiers and infrastructure values are
intentionally not retained.

| Source field | Public field | Value |
| --- | --- | --- |
| `model.model_path` | `model.pretrained_model` | `ZIMAGE_MODEL_PATH` |
| `model.teacher_model_path` | `model.teacher_model` | `ZIMAGE_TEACHER_MODEL_PATH` |
| `data.prompt_jsonl_path` | `data.manifest` | `TRAIN_MANIFEST` |
| `method.method_type` | `method.name` | `opd_gan` |
| Remaining `method.*` | `method.params.*` | Value preserved |
| `train.lr` | `optimizer.generator.lr` | `5e-6` |
| `train.discriminator_lr` | `optimizer.discriminator.lr` | `5e-6` |
| `train.lr_scheduler` | `optimizer.generator.lr_scheduler` | `cosine` |
| `train.lr_decay_steps` | `optimizer.generator.lr_decay_steps` | `3000` |
| `train.lr_min` | `optimizer.generator.lr_min` | `1e-6` |
| `train.grad_accumulation_steps` | `runtime.gradient_accumulation_steps` | `4` |
| `train.micro_batch_size` | `runtime.micro_batch_size` | `1` |
| `train.max_train_steps` | `runtime.max_train_steps` | `1001` |
| `train.ema_decay_rate` | `ema.decay` | `0.99` |
| discriminator checkpoint paths | `discriminator.*_checkpoint` | Required environment variables |

`max_train_steps` counts discriminator or generator phases. With the configured
3:1 phase ratio, 1001 global steps contain 250 generator updates. The cosine
scheduler advances on those generator updates only.

DMD and MeanFlow recipes currently preserve the constructor defaults exercised by
their source method test suites. A named production recipe was not bundled for
MeanFlow in the source tree, so no undocumented experiment configuration is
claimed as canonical.

Legacy discriminator component checkpoints use PyTorch distributed-checkpoint
layout. They can be converted without loading unrelated states:

```bash
PYTHONPATH=src python scripts/convert_legacy_dcp.py \
  /path/to/global_step_3500 \
  /path/to/discriminator.safetensors
```

This extracts an initialization component only. It does not produce a resumable
training checkpoint; see [Checkpoints](checkpoints.md).
