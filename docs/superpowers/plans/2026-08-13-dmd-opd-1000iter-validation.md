# DMD and OPD+GAN 1000-Iteration Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the migrated `verl-distill` DMD and OPD+GAN recipes run to 1000 iterations and verify that 4-NFE debug images improve by step 1000.

**Architecture:** Keep the migrated trainer architecture intact. Use the old OPD trainer tree only as a read-only source for paths, configs, and known-good launch settings; run real jobs from the current `verl-distill` package with explicit environment variables and separate output directories.

**Tech Stack:** Python 3.10+, PyTorch, torchrun/FSDP2, diffusers with Z-Image exports, safetensors, pytest, ruff, PIL debug image output.

## Global Constraints

- Old tree is read-only: `/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/OPD-Trainer-clean-a100_3_fsdp2`.
- Z-Image model path: `/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/Z-Image-Turbo`.
- DMD image manifest: `/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/data/image_train.jsonl`.
- DMD image root: `/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/data/images`.
- OPD prompt manifest: `/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/OPD-Trainer-clean-a100_3_fsdp2/data/generated/official_pickscore_train_prompts.jsonl`.
- OPD prompt key must be `prompt` for that manifest.
- OPD discriminator checkpoint candidate: `/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/data/discriminator.safetensors`.
- Debug sampling must use `method.params.sampling_steps: 4`.
- Acceptance requires debug samples under `OUTPUT_DIR/debug_samples/step-001000`.

---

### Task 1: Establish Environment and Asset Facts

**Files:**
- Read: `src/verl_distill/models/zimage/compatibility.py`
- Read: `src/verl_distill/models/zimage/checkpoints.py`
- Read: `configs/recipes/zimage/dmd.yaml`
- Read: `configs/recipes/zimage/opd_gan.yaml`

**Interfaces:**
- Consumes: filesystem paths listed in Global Constraints.
- Produces: verified shell environment for all later commands.

- [ ] **Step 1: Check CUDA and package support**

Run:

```bash
/tmp/dmd-trainer-venv/bin/python - <<'PY'
import torch
import diffusers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("diffusers", diffusers.__version__)
print("ZImagePipeline", hasattr(diffusers, "ZImagePipeline"))
print("ZImageTransformer2DModel", hasattr(diffusers, "ZImageTransformer2DModel"))
PY
```

Expected: CUDA is true, GPU count is 8, and both Z-Image attributes are true. If the Z-Image attributes are false, rebuild the venv with the old tree's `build_dmd_trainer_env.sh` before running real training.

- [ ] **Step 2: Check data and checkpoint files**

Run:

```bash
ls -ld \
  /mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/Z-Image-Turbo \
  /mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/data/image_train.jsonl \
  /mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/data/images \
  /mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/OPD-Trainer-clean-a100_3_fsdp2/data/generated/official_pickscore_train_prompts.jsonl \
  /mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/data/discriminator.safetensors
```

Expected: every path exists.

- [ ] **Step 3: Inspect OPD checkpoint compatibility**

Run:

```bash
PYTHONPATH=src /tmp/dmd-trainer-venv/bin/python - <<'PY'
from safetensors.torch import safe_open
path = "/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/data/discriminator.safetensors"
with safe_open(path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
print("num_keys", len(keys))
print("dual_keys", sum("dual_projector_multi_feature_discriminator_head" in k for k in keys))
print("frozen_keys", sum("multi_feature_discriminator_head" in k for k in keys))
print("head_keys", sum(".head." in k or k.startswith("head.") for k in keys))
print("first_keys", keys[:20])
PY
```

Expected: at least one key family can load into the OPD trainable dual head and at least one key family can load into the frozen head. If not, search deeper for the old distributed checkpoints and convert them with `scripts/convert_legacy_dcp.py`.

### Task 2: Repair Known Recipe Defaults and Run Package Tests

**Files:**
- Modify: `configs/methods/dmd.yaml`
- Modify: `configs/methods/meanflow.yaml`
- Test: `tests/equivalence/test_method_defaults.py`

**Interfaces:**
- Consumes: `load_config(path: str | Path) -> dict[str, Any]`.
- Produces: passing default-equivalence tests.

- [ ] **Step 1: Fix DMD recipe default**

Change `configs/methods/dmd.yaml`:

```yaml
timestep_shift: 5.0
```

- [ ] **Step 2: Fix MeanFlow recipe defaults**

Change `configs/methods/meanflow.yaml`:

```yaml
flow_shift: 3.0
rt_curriculum_steps: 50000
```

- [ ] **Step 3: Run targeted defaults test**

Run:

```bash
PYTHONPATH=src pytest -q tests/equivalence/test_method_defaults.py
```

Expected: all tests pass.

- [ ] **Step 4: Run relevant trainer and checkpoint tests**

Run:

```bash
PYTHONPATH=src pytest -q \
  tests/unit/test_trainers_smoke.py \
  tests/unit/test_checkpoint.py \
  tests/unit/test_component_checkpoint.py \
  tests/unit/test_zimage_compatibility.py
```

Expected: all tests pass.

### Task 3: Add Real 1000-Step Run Recipes

**Files:**
- Create: `configs/recipes/zimage/dmd_1000_debug.yaml`
- Create: `configs/recipes/zimage/opd_gan_1000_debug.yaml`

**Interfaces:**
- Consumes: current config loader default include syntax.
- Produces: stable recipe names for 1000-step validation.

- [ ] **Step 1: Create DMD validation recipe**

Create `configs/recipes/zimage/dmd_1000_debug.yaml`:

```yaml
defaults:
  - dmd.yaml

runtime:
  max_train_steps: 1000
  save_every_n_steps: 1000
  debug_every_n_steps: 50
  output_dir: ${oc.env:OUTPUT_DIR,/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/dmd_1000_debug}
```

- [ ] **Step 2: Create OPD+GAN validation recipe**

Create `configs/recipes/zimage/opd_gan_1000_debug.yaml`:

```yaml
defaults:
  - opd_gan.yaml

data:
  prompt_key: prompt

runtime:
  max_train_steps: 1000
  save_every_n_steps: 1000
  debug_every_n_steps: 50
  output_dir: ${oc.env:OUTPUT_DIR,/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/opd_gan_1000_debug}
```

- [ ] **Step 3: Validate recipe loading**

Run:

```bash
PYTHONPATH=src \
ZIMAGE_MODEL_PATH=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/Z-Image-Turbo \
TRAIN_MANIFEST=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/data/image_train.jsonl \
TRAIN_IMAGE_ROOT=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/data/images \
python -m verl_distill.cli.train --config dmd_1000_debug --dry-run
```

Expected: JSON output with `"method": "dmd"`.

- [ ] **Step 4: Validate OPD recipe loading**

Run:

```bash
PYTHONPATH=src \
ZIMAGE_MODEL_PATH=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/Z-Image-Turbo \
ZIMAGE_TEACHER_MODEL_PATH=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/Z-Image-Turbo \
TRAIN_MANIFEST=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/OPD-Trainer-clean-a100_3_fsdp2/data/generated/official_pickscore_train_prompts.jsonl \
DISCRIMINATOR_CHECKPOINT=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/data/discriminator.safetensors \
FROZEN_DISCRIMINATOR_CHECKPOINT=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/data/discriminator.safetensors \
python -m verl_distill.cli.train --config opd_gan_1000_debug --dry-run
```

Expected: JSON output with `"method": "opd_gan"`.

### Task 4: Run DMD to 1000 Iterations

**Files:**
- Runtime output: `/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/dmd_1000_debug`

**Interfaces:**
- Consumes: `verl_distill.cli.train:main`.
- Produces: DMD checkpoint and debug images at step 1000.

- [ ] **Step 1: Launch DMD**

Run:

```bash
PYTHONPATH=src \
TOKENIZERS_PARALLELISM=false \
TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=/mnt/hdfs/__MERLIN_USER_DIR__/hf_cache \
ZIMAGE_MODEL_PATH=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/Z-Image-Turbo \
ZIMAGE_TEACHER_MODEL_PATH=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/Z-Image-Turbo \
TRAIN_MANIFEST=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/data/image_train.jsonl \
TRAIN_IMAGE_ROOT=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/data/images \
OUTPUT_DIR=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/dmd_1000_debug \
/tmp/dmd-trainer-venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  -m verl_distill.cli.train \
  --config dmd_1000_debug
```

Expected: logs reach `step=1000`, checkpoint exists under `checkpoints`, and debug samples exist under `debug_samples/step-001000`.

- [ ] **Step 2: Verify DMD artifacts**

Run:

```bash
find /mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/dmd_1000_debug/debug_samples \
  -maxdepth 2 -type f | sort | tail -40
find /mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/dmd_1000_debug/checkpoints \
  -maxdepth 2 -type f | sort | tail -40
```

Expected: `step-001000/00.jpg`, `step-001000/01.jpg`, `step-001000/prompts.txt`, and step-1000 checkpoint files.

### Task 5: Run OPD+GAN to 1000 Iterations

**Files:**
- Runtime output: `/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/opd_gan_1000_debug`

**Interfaces:**
- Consumes: `verl_distill.cli.train:main`.
- Produces: OPD+GAN checkpoint and debug images at step 1000.

- [ ] **Step 1: Launch OPD+GAN**

Run:

```bash
PYTHONPATH=src \
TOKENIZERS_PARALLELISM=false \
TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=/mnt/hdfs/__MERLIN_USER_DIR__/hf_cache \
ZIMAGE_MODEL_PATH=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/Z-Image-Turbo \
ZIMAGE_TEACHER_MODEL_PATH=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/Z-Image-Turbo \
TRAIN_MANIFEST=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/OPD-Trainer-clean-a100_3_fsdp2/data/generated/official_pickscore_train_prompts.jsonl \
DISCRIMINATOR_CHECKPOINT=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/data/discriminator.safetensors \
FROZEN_DISCRIMINATOR_CHECKPOINT=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/data/discriminator.safetensors \
OUTPUT_DIR=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/opd_gan_1000_debug \
/tmp/dmd-trainer-venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  -m verl_distill.cli.train \
  --config opd_gan_1000_debug
```

Expected: logs show discriminator and generator phases, reach `step=1000`, checkpoint exists, and debug samples exist under `debug_samples/step-001000`.

- [ ] **Step 2: Verify OPD artifacts**

Run:

```bash
find /mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/opd_gan_1000_debug/debug_samples \
  -maxdepth 2 -type f | sort | tail -40
find /mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/opd_gan_1000_debug/checkpoints \
  -maxdepth 2 -type f | sort | tail -40
```

Expected: `step-001000/00.jpg`, `step-001000/01.jpg`, `step-001000/prompts.txt`, and step-1000 checkpoint files.

### Task 6: Inspect 4-NFE Debug Image Quality

**Files:**
- Read: DMD debug images under `debug_samples`
- Read: OPD debug images under `debug_samples`

**Interfaces:**
- Consumes: debug image directories produced by Tasks 4 and 5.
- Produces: final validation summary with artifact paths and quality verdict.

- [ ] **Step 1: Build contact sheets for DMD and OPD**

Run:

```bash
PYTHONPATH=src /tmp/dmd-trainer-venv/bin/python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw

roots = [
    Path("/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/dmd_1000_debug/debug_samples"),
    Path("/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/opd_gan_1000_debug/debug_samples"),
]
for root in roots:
    steps = [p for p in [root / "step-000050", root / "step-001000"] if p.is_dir()]
    images = []
    labels = []
    for step in steps:
        for image_path in sorted(step.glob("*.jpg"))[:2]:
            image = Image.open(image_path).convert("RGB").resize((384, 384))
            images.append(image)
            labels.append(f"{root.parent.name}/{step.name}/{image_path.name}")
    if not images:
        continue
    sheet = Image.new("RGB", (384 * len(images), 424), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (image, label) in enumerate(zip(images, labels)):
        x = i * 384
        sheet.paste(image, (x, 0))
        draw.text((x + 8, 392), label, fill=(0, 0, 0))
    out = root.parent / "debug_contact_sheet.jpg"
    sheet.save(out, quality=95)
    print(out)
PY
```

Expected: one contact sheet per run, comparing early and step-1000 samples.

- [ ] **Step 2: Record final verdict**

Inspect:

```text
/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/dmd_1000_debug/debug_contact_sheet.jpg
/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/opd_gan_1000_debug/debug_contact_sheet.jpg
```

Expected: step-1000 images are visibly clearer than early images. If not, record the failure mode with logs and artifact paths.
