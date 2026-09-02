# Installation

## Requirements

- Linux with Python 3.10 or newer
- NVIDIA GPU with a CUDA build of PyTorch
- NCCL for multi-GPU training
- Enough aggregate GPU memory for two Z-Image transformers; the practical
  requirement depends on the method, image size, and world size

Install PyTorch and torchvision from the official PyTorch index matching the
host CUDA driver. Then install the training dependencies:

```bash
python -m pip install -e '.[train]'
```

For tests and formatting tools:

```bash
python -m pip install -e '.[train,dev]'
```

`verl-distill` requires `diffusers>=0.37.0,<0.38` because earlier releases do
not export the Z-Image pipeline and transformer classes used here.

## Model directory

`ZIMAGE_MODEL_PATH` must point to a complete local Z-Image pipeline directory.
At minimum, the loader expects the pipeline configuration, `transformer/`, VAE,
text encoder, tokenizer, and scheduler files. Downloading gated models may
require accepting the model license and authenticating with the model host.

Verify the package and built-in recipe lookup without loading model weights:

```bash
export ZIMAGE_MODEL_PATH=/path/to/Z-Image-Turbo
export TRAIN_MANIFEST=/path/to/train.jsonl
export TRAIN_IMAGE_ROOT=/path/to/images
verl-distill --config dmd --dry-run
```

The command should report `StandardDMD`. Use `meanflow` or `opd_gan` to check the
other built-in recipes.
