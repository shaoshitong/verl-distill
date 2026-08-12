# DMD and OPD+GAN 1000-Iteration Validation Design

## Goal

Run the migrated `verl-distill` DMD and OPD+GAN training paths to 1000 iterations and verify that 4-NFE debug images become clearer by the 1000-iteration checkpoint.

## Source Context

The old working tree is `/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/OPD-Trainer-clean-a100_3_fsdp2`. Its launch scripts identify these default assets:

- Z-Image model: `/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/Z-Image-Turbo`
- Z-Image teacher: `/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/Z-Image-Turbo`
- Prompt data source: `/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/zimage_merged_notext_turbogen_lance`
- OPD launcher: `run_opd.sh`
- Example OPD config: `configs/zimage_task/opd/generated/zimage_s144_x0_4_4_ablate_wu05_nx00_dupr1_nofm_sharednoise_flattenpearson_share_4_a.yaml`

The current public repo uses JSONL data manifests, consolidated discriminator component checkpoints, and built-in YAML recipes under `configs/recipes/zimage`.

## Validation Criteria

DMD is accepted only when:

- The DMD trainer starts from the migrated recipe.
- Training reaches global step 1000.
- Debug samples are written at regular intervals and include `step-001000`.
- Debug sampling uses `sampling_steps: 4`.
- The step-1000 debug images are visibly clearer than early debug images for the same prompts.

OPD+GAN is accepted only when:

- The OPD+GAN trainer starts from the migrated recipe.
- Training reaches global step 1000.
- The three-discriminator-phase, one-generator-phase schedule is preserved.
- Debug samples are written at regular intervals and include `step-001000`.
- Debug sampling uses `sampling_steps: 4`.
- The step-1000 debug images are visibly clearer than early debug images for the same prompts.

## Approach

Use a two-stage validation.

First, repair and validate the migrated package locally: fix known recipe/test mismatches, run targeted unit/equivalence tests, and confirm DMD/OPD trainer smoke paths still work.

Second, discover the real training inputs from the old tree and filesystem, then run real 1000-iteration DMD and OPD+GAN jobs from this repo. The runs should write to separate output directories so checkpoints, logs, and debug images are easy to compare.

## Implementation Notes

- Do not change the old tree except for read-only inspection.
- Prefer existing `verl-distill` config, trainer, checkpoint, and debug-sampling patterns.
- Keep DMD and OPD+GAN outputs separate.
- Preserve `sampling_steps: 4` for debug sampling.
- If the old data is Lance-only, add or use a conversion path to produce JSONL manifests compatible with the migrated repo.
- If OPD discriminator checkpoints are only present as distributed checkpoints, convert them to consolidated component checkpoints before training.

## Test and Run Plan

Run these checks before long training:

- `PYTHONPATH=src pytest -q tests/equivalence/test_method_defaults.py`
- `PYTHONPATH=src pytest -q tests/unit/test_trainers_smoke.py`
- `PYTHONPATH=src pytest -q tests/unit/test_checkpoint.py tests/unit/test_component_checkpoint.py`
- `PYTHONPATH=src pytest -q`
- `ruff check src tests scripts`
- `ruff format --check src tests scripts`
- `bash scripts/check_public_tree.sh`

Run these checks for real validation:

- DMD dry run using discovered real paths.
- OPD+GAN dry run using discovered real paths and discriminator checkpoints.
- DMD training to 1000 iterations.
- OPD+GAN training to 1000 iterations.
- Debug image inspection comparing early samples with `step-001000`.

## Risks

- The current repo may not yet support the old Lance data layout directly.
- OPD checkpoints may need conversion before the public trainer can consume them.
- Real 1000-iteration runs may require multi-GPU resources and can expose environment or memory issues that unit tests do not cover.
- Image clarity is a visual acceptance criterion; logs can prove execution, but final quality needs image inspection.
