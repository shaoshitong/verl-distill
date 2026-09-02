# Training

## Inputs

All methods require:

| Variable | Required | Meaning |
| --- | --- | --- |
| `ZIMAGE_MODEL_PATH` | Yes | Complete local Z-Image pipeline |
| `ZIMAGE_TEACHER_MODEL_PATH` | No | Teacher pipeline; defaults to the student model |
| `TRAIN_MANIFEST` | Yes | JSONL training manifest |
| `OUTPUT_DIR` | No | Run output directory; defaults to `outputs` |
| `RESUME_FROM` | No | Exact checkpoint file or directory to resume |

DMD and MeanFlow also require `TRAIN_IMAGE_ROOT`. Their manifest contains image
paths relative to that directory:

```json
{"image":"000001.png","refined_prompt":"a ceramic cup on a wooden table"}
```

Paths escaping `TRAIN_IMAGE_ROOT` are rejected. OPD uses a prompt-only manifest:

```json
{"refined_prompt":"a ceramic cup on a wooden table"}
```

OPD additionally requires `DISCRIMINATOR_CHECKPOINT` and
`FROZEN_DISCRIMINATOR_CHECKPOINT`. These initialize components; they are not
training-state checkpoints.

## Start a run

Single process:

```bash
verl-distill --config dmd
verl-distill --config meanflow
verl-distill --config opd_gan
```

Single-node FSDP2:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m verl_distill.cli.train --config dmd
```

For multiple nodes, supply the rendezvous settings required by `torchrun`:

```bash
torchrun --nnodes=2 --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint=HOST:PORT \
  -m verl_distill.cli.train --config dmd
```

FSDP2 is applied only when `WORLD_SIZE` is greater than one. The built-in DMD
and MeanFlow recipes set `max_train_steps: 0`, meaning no step limit. Set a
positive value in a copied YAML file for a bounded run; the CLI intentionally
does not implement Hydra-style overrides.

## Step semantics

`gradient_accumulation_steps` is the number of micro-batches in one optimizer
step. DMD updates its fake score every global step and its generator according
to `dfake_gen_update_ratio`.

OPD uses three discriminator phases followed by one generator phase. Its
`max_train_steps: 1001` counts phases, not generator updates. The cosine
scheduler advances only on generator phases, matching the source trainer.

The OPD default accumulation is four, so the dataloader must provide at least
four batches per rank. At batch size one, the prompt manifest therefore needs
at least `4 * WORLD_SIZE` usable rows.

## Data order

Distributed samplers receive a new epoch value whenever a dataloader is
restarted. Resume restores model, optimizer, scheduler, EMA, and process RNG
state. It does not serialize worker prefetch queues; exact next-sample replay is
not guaranteed when resuming a multi-worker dataloader mid-epoch.
