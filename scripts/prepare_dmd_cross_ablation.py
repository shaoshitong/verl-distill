#!/usr/bin/env python3
"""Resolve portable A/B recipes into a fresh, serial training queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import yaml

from verl_distill.config import load_config

REPO = Path(__file__).resolve().parents[1]
JOBS = {
    "A_fake0_gen0to5": "dmd_cross_fake0_gen0to5",
    "B_fake0to5_gen0": "dmd_cross_fake0to5_gen0",
}


def prepare(output_root: Path, occupier: str = "") -> None:
    root = output_root.resolve()
    if root.exists():
        raise FileExistsError(f"Choose a new output directory: {root}")
    if occupier and not Path(occupier).is_file():
        raise FileNotFoundError(f"GPU occupier script not found: {occupier}")
    configs = {name: load_config(recipe) for name, recipe in JOBS.items()}
    root.mkdir(parents=True)
    for name, config in configs.items():
        folder = root / name
        folder.mkdir()
        config["runtime"]["output_dir"] = str(folder)
        (folder / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (folder / "env.sh").write_text(
            "#!/usr/bin/env bash\n"
            f"export PYTHONPATH={shlex.quote(str(REPO / 'src'))}\n"
            "export TOKENIZERS_PARALLELISM=false\n"
            "export TORCH_NCCL_AVOID_RECORD_STREAMS=1\n"
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
            "export OMP_NUM_THREADS=1\n"
            "export HF_HUB_OFFLINE=1\n"
        )
        (folder / "train.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"source {shlex.quote(str(folder / 'env.sh'))}\n"
            f"cd {shlex.quote(str(REPO))}\n"
            f"exec {shlex.quote(sys.executable)} -m torch.distributed.run "
            "--standalone --nproc-per-node=8 -m verl_distill.cli.train "
            f"--config {shlex.quote(str(folder / 'resolved_config.yaml'))}\n"
        )
    shutil.copyfile(REPO / "scripts/run_dmd_cross_queue.py", root / "run_queue.py")
    paths = (
        list((REPO / "src").rglob("*.py"))
        + list(root.glob("*/resolved_config.yaml"))
        + list(root.glob("*/env.sh"))
        + list(root.glob("*/train.sh"))
    )
    (root / "input_hashes.json").write_text(
        json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2)
        + "\n"
    )
    (root / "queue_settings.json").write_text(
        json.dumps(
            {"gpu_occupier_script": str(Path(occupier).resolve()) if occupier else None}, indent=2
        )
        + "\n"
    )
    (root / "run_sequential.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"exec {shlex.quote(sys.executable)} -u {shlex.quote(str(root / 'run_queue.py'))} "
        f">> {shlex.quote(str(root / 'queue.log'))} 2>&1\n"
    )
    print(f"Prepared only; training has not started: {root}")
    print(
        f"Launch when all 8 GPUs are available: bash {shlex.quote(str(root / 'run_sequential.sh'))}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpu-occupier-script", default=os.environ.get("GPU_OCCUPIER_SCRIPT", ""))
    args = parser.parse_args()
    prepare(args.output_root, args.gpu_occupier_script)


if __name__ == "__main__":
    main()
