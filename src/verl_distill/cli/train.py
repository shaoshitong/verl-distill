import argparse
import json
import logging

from verl_distill.algorithms import build_algorithm
from verl_distill.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Z-Image distillation method")
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configuration and construct the algorithm without loading a model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    config = load_config(args.config)
    method = config.get("method", {})
    name = method.get("name")
    if not name:
        raise ValueError("Configuration must define method.name")
    algorithm = build_algorithm(name, method.get("params", {}))
    if args.dry_run:
        print(json.dumps({"method": name, "algorithm": type(algorithm).__name__}, indent=2))
        return
    if name in {"dmd", "dmd_full", "meanflow", "opd_gan"}:
        from verl_distill.trainers import train_dmd, train_meanflow, train_opd_gan

        trainers = {
            "dmd": train_dmd,
            "dmd_full": train_dmd,
            "meanflow": train_meanflow,
            "opd_gan": train_opd_gan,
        }
        trainer = trainers[name]
        trainer(config)
        return
    raise SystemExit(f"The end-to-end {name} trainer is still being migrated.")


if __name__ == "__main__":
    main()
