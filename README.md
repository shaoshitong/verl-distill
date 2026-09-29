# verl-distill

Diffusion distillation training for Z-Image and Qwen-Image-2.1.

## Recipes

| Model | Recipe | Training |
| --- | --- | --- |
| Qwen-Image-2.1 | [HF DMD, CFG=2](configs/recipes/qwen_image21/dmd_hf_cfg2_fsdp1.yaml) | HF initialization; 3000 Fake / 600 Generator updates; FSDP1 |
| Qwen-Image-2.1 | [REFLOW → DMD](configs/recipes/qwen_image21/reflow100_lr1e5_ga4_dmd_fsdp1.yaml) | 100 REFLOW updates followed by DMD |
| Z-Image | [ODE warmup → DMD](configs/recipes/zimage/dmd_refaligned_ode_warmup_fsdp1.yaml) | 1000 warmup iterations; FSDP1; schedule-free Fake optimizer |
| Z-Image | [MeanFlow](configs/recipes/zimage/meanflow.yaml), [OPD+GAN](configs/recipes/zimage/opd_gan.yaml) | Alternative training methods |

The Qwen HF recipe runs on 4 nodes × 8 A100 GPUs with GA=1. Generator, Fake and
Teacher load independent copies of the same HF weights. Teacher CFG is 2;
Generator and Fake use positive conditions only. Generator LR is 5e-7, Fake LR
is 2e-6, and Fake uses velocity MSE with constant weight 1. Only transformer
blocks are trained; gradient checkpointing is enabled and parameter offload is
disabled. Each cycle contains five Fake updates and one Generator update.

## Usage

- [Qwen training](docs/qwen_image21.md): pinned environment, data caches, multi-node launch, losses and checkpoints.
- [Z-Image installation](docs/installation.md) and [training](docs/training.md): setup, datasets and launch commands.
- [Z-Image checkpoints](docs/checkpoints.md) and [configuration mapping](docs/config-mapping.md).

Use separate environments for the Qwen `qwen21` extra and Z-Image `train` extra.
Keep datasets, model weights, caches and experiment outputs outside the checkout.

## Development

```bash
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
```

Qwen adapter tests require the pinned Qwen environment. The CUDA/FSDP smoke
programs under `tests/` run separately from the CPU unit tests.

## Shallow DMD Cross Ablation

Reproduce the schedule-free generator and latent/feature STE cross ablation with the [new-node quickstart](docs/dmd_cross_quickstart.md) and [implementation guide](docs/dmd_cross_reproduction_zh.md). The portable recipes are `dmd_cross_fake0_gen0to5` and `dmd_cross_fake0to5_gen0`. Models, data, and the step-1000 checkpoint must be provided separately.

Apache-2.0. Model and dataset licenses are separate.
