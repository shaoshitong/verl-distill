# Checkpoints

## Training state

Checkpoints are written every `runtime.save_every_n_steps`:

- Single process: `OUTPUT_DIR/checkpoints/step-N.pt`
- Multi-process: `OUTPUT_DIR/checkpoints/step-N/` in PyTorch DCP format

Set `RESUME_FROM` to that exact file or directory. The formats are not
interchangeable. A distributed checkpoint includes one RNG and EMA sidecar per
rank and should be resumed with the same world size. There is no automatic
"latest" lookup. Training does not write an extra final checkpoint when the
last step is not a save interval.

MeanFlow and OPD training checkpoints contain online student parameters,
optimizer state, EMA shadows, and scheduler state where applicable. Exporting a
standalone EMA inference model is a separate operation and is not currently a
CLI command.

## Legacy discriminator components

Older OPD discriminator initializers may be stored inside a larger distributed
checkpoint. Extract only the required component with:

```bash
PYTHONPATH=src python scripts/convert_legacy_dcp.py \
  /path/to/legacy-dcp /path/to/discriminator.safetensors
```

The default state name is `teacher_discriminator_model`; use `--state-name` if
the legacy checkpoint uses another key. This conversion does not include model
training state, optimizer state, global step, or RNG and cannot be used with
`RESUME_FROM`.
