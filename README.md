# verl-distill

Training code for Z-Image distillation experiments. The maintained path in this
checkout is the full-model DMD recipe aligned against the reference DMD trainer.

## Current DMD Recipe

Use:

```bash
configs/recipes/zimage/dmd_refaligned_ode_warmup_fsdp1.yaml
```

This is the recipe used for the validated run:

- generator init: `ZIMAGE_MODEL_PATH`, normally `Z-Image-FM1`
- teacher model: `ZIMAGE_TEACHER_MODEL_PATH`, normally `Z-Image`
- fake score init: `ZIMAGE_FAKE_SCORE_MODEL_PATH`, normally `Z-Image-FM1`
- data: Lance image dataset at 1024 resolution
- ODE warmup: first 1000 iterations from precomputed ODE pairs
- DMD phase: starts after warmup and trains generator plus fake score
- distributed backend: FSDP1
- mixed precision: parameters `bfloat16`, gradient reduce `float32`, buffers `float32`
- gradient accumulation: 4
- debug sampling: 1024 x 1024, 4-step trajectory grids every 100 steps

Important DMD settings:

```yaml
method.params.timestep_shift: 1.0
method.params.fake_score_use_generator_timestep: false
method.params.warmup_type: ode_pair
method.params.warmup_iterations: 1000
optimizer.generator.lr: 1.0e-5
optimizer.generator.warmup_lr: 1.0e-4
optimizer.fake_score.lr: 1.0e-5
optimizer.fake_score.warmup_lr: 0.0
runtime.max_train_steps: 50000
distributed.fsdp_backend: fsdp1
```

## Install

```bash
python -m pip install -e '.[train]'
```

For development:

```bash
python -m pip install -e '.[train,dev]'
```

## Run DMD

Example launch on 8 GPUs:

```bash
cd /path/to/verl-distill

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export ZIMAGE_MODEL_PATH=/path/to/Z-Image-FM1
export ZIMAGE_TEACHER_MODEL_PATH=/path/to/Z-Image
export ZIMAGE_FAKE_SCORE_MODEL_PATH=/path/to/Z-Image-FM1
export TRAIN_LANCE_DATA_DIR=/path/to/zimage_merged_notext_turbogen_lance
export ZIMAGE_ODE_PAIR_DIR=/path/to/ode_pairs/zimage_turbo_cfg0_buckets_seq1024_lance4_text6_full_qwen_scored
export OUTPUT_DIR=/path/to/verl-distill-runs/dmd_refaligned_fsdp1_ode_warmup_50000

torchrun --standalone --nproc-per-node=8 \
  -m verl_distill.cli.train \
  --config configs/recipes/zimage/dmd_refaligned_ode_warmup_fsdp1.yaml
```

The Lance dataset path is provided through `TRAIN_LANCE_DATA_DIR`:

```text
/path/to/zimage_merged_notext_turbogen_lance
```

`TRAIN_MANIFEST` is not used by the Lance recipe.

## Outputs

Training writes to `OUTPUT_DIR`:

- `train.log`: scalar logs
- `debug_samples/step-xxxx/trajectory.png`: 4-step debug trajectory grid
- `debug_dmd_tensors/step-xxxx/`: optional tensor/image dumps when enabled
- `checkpoints/step-xxxx/`: distributed checkpoints

Generated outputs are intentionally ignored by git. Keep large runs outside the
source checkout, for example under a sibling `verl-distill-runs` directory.

## Dataset Publication

The training data for this recipe is published separately as a Hugging Face
dataset:

```text
sst12345/verl-distill-dataset
```

Expected layout:

```text
zimage_merged_notext_turbogen_lance/
ode_pairs/zimage_turbo_cfg0_buckets_seq1024_lance4_text6_full_qwen_scored/
```

## Other Recipes

The repo still includes baseline recipes for:

- `dmd`: minimal DMD recipe
- `dmd_1000_debug`: short DMD smoke/debug recipe used by probe tools
- `meanflow`: MeanFlow recipe
- `opd_gan`: OPD+GAN recipe

## Checks

```bash
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
bash scripts/check_public_tree.sh
```

Apache-2.0. Model and dataset licenses are not bundled with this repository.
