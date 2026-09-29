"""Immutable negative-text conditions with the SAME ordered reference images."""

import argparse
import json
from pathlib import Path

import torch

from verl_distill.data.qwen_image21 import (
    atomic_json,
    canonical_hash,
    read_manifest,
    sha256,
)

from .guidance import encode_negative_condition
from .modeling import (
    condition_entry_path,
    load_condition_pipeline,
    model_identity,
    open_reused_cache,
    require_qwen_runtime,
)


def negative_key(record, identity):
    return canonical_hash(
        {
            "schema": 1,
            "model": canonical_hash(identity),
            "prompt": "",
            "reference_images": record["reference_images"],
            "reference_sha256": record["reference_sha256"],
            "resolution": 2048,
        }
    )


class NegativeConditionStore:
    def __init__(self, root, identity, manifest_hashes):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text())
        if (
            self.index.get("schema") != 1
            or self.index["model_identity"] != identity
            or self.index["manifest_hashes"] != manifest_hashes
            or self.index["negative_prompt"] != ""
        ):
            raise ValueError("Negative cache contract mismatch")
        self.identity, self.verified = identity, set()
        self.reused = open_reused_cache(self.index.get("reused_cache"), identity)

    def validate_record(self, record):
        key = negative_key(record, self.identity)
        if key not in self.index["entries"]:
            raise ValueError(f"Missing negative condition for {record['id']}")
        return key

    def get(self, record, positive, device):
        key = self.validate_record(record)
        entry = self.index["entries"][key]
        path = condition_entry_path(self.root, key, entry, self.reused)
        if key not in self.verified:
            if sha256(path) != entry["sha256"]:
                raise ValueError("Negative cache payload digest mismatch")
            self.verified.add(key)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["key"] != key:
            raise ValueError("Negative payload identity mismatch")
        encoded = {}
        for name, value in payload["condition"].items():
            if isinstance(value, torch.Tensor):
                value = value.to(device)
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError("Nonfinite negative embedding")
            encoded[name] = value
        tokens = record["height"] // 16 * (record["width"] // 16) // 4
        prefix = encoded["img_mask"]
        encoded["img_mask"] = torch.cat([prefix, prefix.new_ones(1, tokens)], 1)
        # This explicitly reuses positive reference latents and image layout.
        return {**positive, **encoded}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build", "publish"])
    for name in ("manifest", "eval-manifest", "model", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--reference-root", default="/tmp/qwen21-dataset")
    p.add_argument("--reuse-cache")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=32)
    args = p.parse_args()
    require_qwen_runtime()
    identity = model_identity(args.model)
    hashes = {"train": sha256(args.manifest), "eval": sha256(args.eval_manifest)}
    descriptor = (
        {
            "root": str(Path(args.reuse_cache).resolve()),
            "index_sha256": sha256(Path(args.reuse_cache) / "index.json"),
        }
        if args.reuse_cache
        else None
    )
    reused = open_reused_cache(descriptor, identity)
    records = read_manifest(args.manifest)["records"] + read_manifest(args.eval_manifest)["records"]
    unique = {negative_key(row, identity): row for row in records}
    ordered = sorted(unique.items())
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "index.json").exists():
        raise FileExistsError("Published negative cache is immutable")
    if args.command == "build":
        if not 0 <= args.shard < args.shards:
            raise ValueError("Invalid shard")
        torch.cuda.set_device(0)
        pipe = None
        entries = {}
        assigned = ordered[args.shard :: args.shards]
        for i, (key, row) in enumerate(assigned):
            if reused and key in reused[1]["entries"]:
                entries[key] = {**reused[1]["entries"][key], "source": "base"}
                continue
            if pipe is None:
                pipe = load_condition_pipeline(args.model, "cuda")
                pipe.vae.cpu()
            # The encoder only sees reference images and empty text. Store img_mask
            # prefix independently of the target shape so all T2I rows share one entry.
            record = {**row, "height": 32, "width": 32}
            condition = encode_negative_condition(pipe, record, {}, args.reference_root, "cuda")
            condition["img_mask"] = condition["img_mask"][:, :-1]
            condition = {
                k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in condition.items()
            }
            path = root / key[:2] / f"{key}.pt"
            path.parent.mkdir(exist_ok=True)
            tmp = path.with_suffix(".tmp")
            torch.save({"key": key, "condition": condition}, tmp)
            tmp.replace(path)
            entries[key] = {"file": str(path.relative_to(root)), "sha256": sha256(path)}
            if (i + 1) % 10 == 0 or i + 1 == len(assigned):
                print(
                    f"negative shard={args.shard} {i + 1}/{len(assigned)} refs={len(row['reference_images'])}",
                    flush=True,
                )
        atomic_json(
            root / f"shard-{args.shard:02d}.json",
            {
                "identity": identity,
                "hashes": hashes,
                "shards": args.shards,
                "entries": entries,
                "reused_cache": descriptor,
            },
        )
    else:
        entries = {}
        for i in range(args.shards):
            shard = json.loads((root / f"shard-{i:02d}.json").read_text())
            if (
                shard["identity"] != identity
                or shard["hashes"] != hashes
                or shard["shards"] != args.shards
                or shard.get("reused_cache") != descriptor
            ):
                raise ValueError("Negative cache shards disagree")
            if entries.keys() & shard["entries"].keys():
                raise ValueError("Duplicate cache ownership")
            entries.update(shard["entries"])
        if set(entries) != set(unique):
            raise ValueError("Negative cache coverage mismatch")
        from concurrent.futures import ThreadPoolExecutor

        def verify(item):
            key, entry = item
            path = condition_entry_path(root, key, entry, reused)
            if entry.get("source") == "base":
                if not path.is_file():
                    raise FileNotFoundError(path)
            elif sha256(path) != entry["sha256"]:
                raise ValueError("Corrupt negative cache entry")

        with ThreadPoolExecutor(max_workers=8) as pool:
            for count, _ in enumerate(pool.map(verify, entries.items()), 1):
                if count % 250 == 0:
                    print(f"negative cache verified {count}/{len(entries)}", flush=True)
        atomic_json(
            root / "index.json",
            {
                "schema": 1,
                "model_identity": identity,
                "manifest_hashes": hashes,
                "negative_prompt": "",
                "entries": entries,
                "reused_cache": descriptor,
            },
        )
        print(
            f"Published {len(entries)} negative conditions for {len(records)} records", flush=True
        )


if __name__ == "__main__":
    main()
