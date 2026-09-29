"""Build additional positive and same-reference negative conditions with one frozen encoder per GPU."""

import argparse
from pathlib import Path

import torch

from verl_distill.data.qwen_image21 import atomic_json, read_manifest, sha256
from verl_distill.models.qwen_image21.guidance import encode_negative_condition
from verl_distill.models.qwen_image21.modeling import (
    condition_key,
    encode_condition,
    load_condition_pipeline,
    model_identity,
    open_reused_cache,
    require_qwen_runtime,
    save_condition,
)
from verl_distill.models.qwen_image21.negative_cache import negative_key


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    for name in [
        "manifest",
        "eval-manifest",
        "model",
        "positive-output",
        "negative-output",
        "reuse-positive",
        "reuse-negative",
    ]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--reference-root", default="/tmp/qwen21-dataset")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, default=32)
    args = p.parse_args()
    require_qwen_runtime()
    torch.cuda.set_device(0)
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")
    identity = model_identity(args.model)
    hashes = {"train": sha256(args.manifest), "eval": sha256(args.eval_manifest)}
    records = read_manifest(args.manifest)["records"] + read_manifest(args.eval_manifest)["records"]

    def descriptor(root):
        return {
            "root": str(Path(root).resolve()),
            "index_sha256": sha256(Path(root) / "index.json"),
        }

    pd, nd = descriptor(args.reuse_positive), descriptor(args.reuse_negative)
    pb, nb = open_reused_cache(pd, identity), open_reused_cache(nd, identity)
    pos, neg = Path(args.positive_output), Path(args.negative_output)
    for root in [pos, neg]:
        root.mkdir(parents=True, exist_ok=True)
        if (root / "index.json").exists():
            raise FileExistsError("Published cache is immutable")
    pipe = None

    def pipeline():
        nonlocal pipe
        if pipe is None:
            pipe = load_condition_pipeline(args.model, "cuda")
        return pipe

    entries = {}
    built = 0
    layouts = {}
    assigned = records[args.shard :: args.shards]
    for i, row in enumerate(assigned):
        key = condition_key(row, identity)
        if key in pb[1]["entries"]:
            entries[key] = {**pb[1]["entries"][key], "source": "base"}
        else:
            condition = encode_condition(pipeline(), row, args.reference_root, "cuda")
            vlm_length = condition["encoder_hidden_states"].shape[1]
            text_tokens = int((~condition["img_mask"][0, :vlm_length]).sum().item())
            reference_tokens = (
                condition["reference_latents"].shape[1]
                if condition["reference_latents"] is not None
                else 0
            )
            target_tokens = row["height"] // 16 * (row["width"] // 16)
            total = target_tokens + reference_tokens + text_tokens
            category = str(len(row["reference_images"]))
            if total > layouts.get(category, {}).get("total_tokens", 0):
                layouts[category] = {
                    "record": row,
                    "total_tokens": total,
                    "text_tokens": text_tokens,
                    "reference_tokens": reference_tokens,
                    "target_tokens": target_tokens,
                }
            key, entry = save_condition(pos, row, identity, condition)
            entries[key] = entry
            built += 1
            del condition
        if (i + 1) % 10 == 0 or i + 1 == len(assigned):
            print(
                f"positive shard={args.shard} {i + 1}/{len(assigned)} newly_encoded={built}",
                flush=True,
            )
    atomic_json(
        pos / f"shard-{args.shard:05d}.json",
        {
            "schema": 1,
            "num_shards": args.shards,
            "model_identity": identity,
            "manifest_hashes": hashes,
            "entries": entries,
            "reused_cache": pd,
        },
    )
    atomic_json(pos / f"layout-shard-{args.shard:02d}.json", layouts)
    unique = {negative_key(row, identity): row for row in records}
    assigned = sorted(unique.items())[args.shard :: args.shards]
    entries = {}
    built = 0
    for i, (key, row) in enumerate(assigned):
        if key in nb[1]["entries"]:
            entries[key] = {**nb[1]["entries"][key], "source": "base"}
        else:
            condition = encode_negative_condition(
                pipeline(), {**row, "height": 32, "width": 32}, {}, args.reference_root, "cuda"
            )
            condition["img_mask"] = condition["img_mask"][:, :-1]
            condition = {
                k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in condition.items()
            }
            path = neg / key[:2] / f"{key}.pt"
            path.parent.mkdir(exist_ok=True)
            tmp = path.with_suffix(".tmp")
            torch.save({"key": key, "condition": condition}, tmp)
            tmp.replace(path)
            entries[key] = {"file": str(path.relative_to(neg)), "sha256": sha256(path)}
            built += 1
            del condition
        if (i + 1) % 10 == 0 or i + 1 == len(assigned):
            print(
                f"negative shard={args.shard} {i + 1}/{len(assigned)} newly_encoded={built}",
                flush=True,
            )
    atomic_json(
        neg / f"shard-{args.shard:02d}.json",
        {
            "identity": identity,
            "hashes": hashes,
            "shards": args.shards,
            "entries": entries,
            "reused_cache": nd,
        },
    )


if __name__ == "__main__":
    main()
