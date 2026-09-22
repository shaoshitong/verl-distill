# verl-distill

Training code for Z-Image DMD and Qwen-Image-2.1 REFLOW → DMD.
See [Qwen-Image-2.1 training](#reflow--dmd-for-qwen-image-21) for its environment,
paired data, eight-GPU launch, checkpoints and export.

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
- fake score optimizer: schedule-free AdamW
- debug sampling: 1024 x 1024, 4-step trajectory grids every 100 steps

Important DMD settings:

```yaml
method.params.timestep_shift: 1.0
method.params.fake_score_use_generator_timestep: false
method.params.warmup_type: ode_pair
method.params.warmup_iterations: 1000
optimizer.generator.lr: 1.0e-5
optimizer.generator.warmup_lr: 1.0e-4
optimizer.fake_score.type: adamw_schedule_free
optimizer.fake_score.lr: 5.0e-5
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
ode_pair_archives/zimage_turbo_cfg0_buckets_seq1024_lance4_text6_full_qwen_scored/
```

The Lance directory is uploaded directly. The ODE warmup pairs are also exposed
as archive shards under `ode_pair_archives/` to avoid pushing tens of thousands
of small files through the Hub. Reconstruct the local ODE pair directory with:

```bash
ARCHIVE_DIR=/path/to/ode_pair_archives/zimage_turbo_cfg0_buckets_seq1024_lance4_text6_full_qwen_scored
ODE_DIR=/path/to/ode_pairs/zimage_turbo_cfg0_buckets_seq1024_lance4_text6_full_qwen_scored

mkdir -p "$ODE_DIR"
cd "$ODE_DIR"

cat "$ARCHIVE_DIR"/images.tar.part-* | tar -xf -
cat "$ARCHIVE_DIR"/latents.tar.part-* | tar -xf -
cat "$ARCHIVE_DIR"/noise.tar.part-* | tar -xf -
```

## Other Recipes

The repo still includes baseline recipes for:

- `dmd`: minimal DMD recipe
- `dmd_1000_debug`: short DMD smoke/debug recipe used by probe tools
- `meanflow`: MeanFlow recipe
- `opd_gan`: OPD+GAN recipe

## REFLOW + DMD for Qwen-Image-2.1

Qwen-Image-2.1 uses a separate, pinned environment and paired latent dataset.
The current reproduction recipe is
[`reflow100_lr1e5_ga4_dmd_fsdp1.yaml`](configs/recipes/qwen_image21/reflow100_lr1e5_ga4_dmd_fsdp1.yaml).
It starts from the original HF model, trains REFLOW for 100 updates, then switches
automatically to DMD. The older `reflow_dmd_fsdp1.yaml` remains the base recipe
(1000 REFLOW updates, LR=1e-4, GA=1); select the new recipe explicitly.

| Setting | REFLOW | DMD Generator | DMD Fake |
|---|---|---|---|
| Optimizer updates | 100 | 600 | 3000 |
| AdamW learning rate | 1e-5 | 5e-7 | 5e-6 |
| Betas | (0.9, 0.999) | (0.9, 0.999) | (0.9, 0.95) |
| Weight decay | 0.01 | 0.1 | 0.1 |
| Gradient accumulation | 4 | 4 | 4 |

Each DMD cycle has five Fake updates followed by one Generator update. With eight
GPUs and one sample per GPU microbatch, the effective batch is 32. Real is frozen
at the original HF weights; this recipe initializes Fake from the REFLOW Generator.
Generator and Fake train the DiT while freezing `time_text_embed` and `txt_in`.
The shared modulation projection remains trainable. VLM and VAE are frozen.

### Environment

Use Python 3.12 and a separate environment from the Z-Image `train` extra:

```bash
python3.12 -m venv /tmp/qwen-image21-train-venv
source /tmp/qwen-image21-train-venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[qwen21,dev]'
python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
```

The extra pins Diffusers to `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`
and Transformers to 5.17.0. Use `Qwen/Qwen-Image-2.1` model revision
`b3179ad355be050328e483a9dfdd9e60cd62adfa`, including `transformer`,
`text_encoder`, `processor`, `vae`, and `scheduler`. Model weights are downloaded
separately and are not stored in this repository.

### Paired data and condition cache

Each completed production sample provides `initial_noise.pt`, `x0_latent.pt`,
`image.png`, and `complete.json` with provenance and file hashes. Noise and x0
are the actual paired endpoints from the original model's 40-step, CFG=1 rollout.
Latents are already diffusion-normalized. Both text-to-image and editing samples
are supported at their recorded dimensions, with reference images in their
original order.

Prepare a fixed snapshot and conditions before training:

```bash
export QWEN21_MODEL_PATH=/path/to/Qwen-Image-2.1
export QWEN21_REFERENCE_ROOT=/path/to/extracted_dataset
export QWEN21_DEBUG_CSV=/path/to/complex_prompt.csv
export QWEN21_TRAIN_MANIFEST=/path/to/snapshot/train.json
export QWEN21_EVAL_MANIFEST=/path/to/snapshot/eval.json
export QWEN21_CONDITION_CACHE=/path/to/condition_cache

python -m verl_distill.tools.prepare_qwen_image21 snapshot \
  --output-root /path/to/dataset_inference \
  --reference-root "$QWEN21_REFERENCE_ROOT" \
  --prompts-csv "$QWEN21_DEBUG_CSV" --snapshot-dir /path/to/snapshot

python -m verl_distill.tools.prepare_qwen_image21 cache \
  --model "$QWEN21_MODEL_PATH" \
  --manifest "$QWEN21_TRAIN_MANIFEST" --eval-manifest "$QWEN21_EVAL_MANIFEST" \
  --cache-dir "$QWEN21_CONDITION_CACHE" --device cuda:0

python -m verl_distill.tools.prepare_qwen_image21 publish \
  --manifest "$QWEN21_TRAIN_MANIFEST" --eval-manifest "$QWEN21_EVAL_MANIFEST" \
  --cache-dir "$QWEN21_CONDITION_CACHE"
```

The evaluation CSV requires 64 unique rows with `prompt_id,prompt,width,height`;
only its first eight prompts are used for debug generation. Snapshot and cache
directories must be new. To distribute cache preparation, run `cache` once per
GPU with distinct `--shard-id` values and the same `--num-shards`, then pass that
same shard count to `publish`. Existing immutable snapshots/caches can be reused
when their model identity and paths match; these preparation steps are not needed
again for every training run.

### Launch on eight GPUs

After setting the data variables above:

```bash
export OUTPUT_DIR=/path/to/runs/qwen21_reflow100_dmd
export QWEN21_PYTHON=/tmp/qwen-image21-train-venv/bin/python
export QWEN21_RESUME_FROM=""

# Configuration validation only: no model loading or training.
python -m verl_distill.cli.train --dry-run \
  --config configs/recipes/qwen_image21/reflow100_lr1e5_ga4_dmd_fsdp1.yaml

NPROC_PER_NODE=8 bash scripts/train_qwen_image21.sh \
  --config configs/recipes/qwen_image21/reflow100_lr1e5_ga4_dmd_fsdp1.yaml
```

`--config` selects the recipe instead of the launcher's legacy default. Use a new
`OUTPUT_DIR` for every launch, including resumed runs. To resume, set
`QWEN21_RESUME_FROM` to a completed checkpoint directory and keep the same GPU
count, recipe, model and data identities. A checkpoint records optimizer states,
update counters, RNG and data cursor. Special refinement or DMD-fork experiments
require their explicit configuration; changing a resume path alone is not a fork.

### What the losses train

REFLOW uses the saved noise/x0 pair. DMD resamples noise around the dataset's clean
latent. Both select **one** of six Generator sigma values per microbatch:

```text
gen_input = (1-sigma) * clean + sigma * noise
gen_velocity = Generator(gen_input, sigma, condition)

REFLOW loss = MSE(gen_velocity, saved_noise-clean)
DMD generated = gen_input - sigma*gen_velocity
```

`generated` is a single-forward x0 estimate, not a six-step rollout. The six-point
time grid inherits Qwen's resolution-dependent dynamic shift and ends with a
model call at sigma=0.4; evaluation then integrates to zero. The model predicts
velocity and receives sigma directly, without a `1-sigma` timestep reversal.

DMD independently samples `score_sigma` from the original 1000-point shifted grid,
restricted to [0.02, 0.98], and adds fresh noise to detached `generated`. Fake and
Real receive the same noisy input and score timestep. Fake trains a weighted x0
MSE with weight `min(((1-s)/max(s,1e-4))**2, 50)`; it equals epsilon MSE only when
the cap is inactive. Generator uses the detached `(fake_x0-real_x0)` direction,
normalized by `mean(abs(generated-real_x0))`, through an x0 surrogate loss. Score
forward passes do not receive gradients during the Generator update. Full formulas
and implementation details are in [the Qwen training guide](docs/qwen_image21.md).

### Infrastructure and outputs

FSDP1 wraps all 32 DiT blocks with `use_orig_params=True`. Training uses FP32 master
parameters/gradients, BF16 forward, non-reentrant gradient checkpointing and
segmented FlashAttention2 preserving Qwen's block-causal mask. The recipe enables
phase-wise weight offload: Real stays on CPU during Fake updates, Generator is
unloaded after each Fake-phase generation, and Fake/Real weights are unloaded for
Generator backward. Adam states remain on GPU. No persistent training KV cache is
used. Large multi-reference samples can still exceed device memory.

Outputs under `OUTPUT_DIR`:

- `logs/rank-*.jsonl`: losses, sigma statistics, timings and parameter-update probes.
- `checkpoints/reflow_step_*`: saved at REFLOW updates 1, 20 and 100.
- `checkpoints/fake_step_*`: early DMD saves at 5/40/45, then every 600 Fake updates
  and completion, after the corresponding Generator update.
- `debug/<event>/rollout/`: eight six-step samples, one per GPU.
- `debug/<event>/rollout_25step/`: the same samples/noise with 25 steps, the original
  scheduler and CFG=1.
- `debug/<event>/train_dmd/`: actual Fake/Real x0 predictions and differences.
- `failure-rank-*.json`: rank-local failure evidence.

Debug runs at REFLOW update 1 and every 20 REFLOW/Fake updates. When saving and
debugging coincide, checkpoint publication happens first. Completed checkpoints
have a `COMPLETE` marker. Export only Generator weights with the checkpoint's
original world size:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m verl_distill.tools.export_qwen_image21 \
  --checkpoint /path/to/checkpoints/reflow_step_000100 \
  --model "$QWEN21_MODEL_PATH" --output /path/to/exported_transformer
```

The export is a Diffusers transformer directory; VLM, VAE, processor and scheduler
remain supplied by the original model. See [the training guide](docs/qwen_image21.md)
for resume checks, export and historical validation details.

## Checks

```bash
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
bash scripts/check_public_tree.sh
```

Apache-2.0. Model and dataset licenses are not bundled with this repository.

## Shallow DMD Cross Ablation

Reproduce the schedule-free generator and latent/feature STE cross ablation with the [new-node quickstart](docs/dmd_cross_quickstart.md) and [implementation guide](docs/dmd_cross_reproduction_zh.md). The portable recipes are `dmd_cross_fake0_gen0to5` and `dmd_cross_fake0to5_gen0`. Models, data, and the step-1000 checkpoint must be provided separately.
