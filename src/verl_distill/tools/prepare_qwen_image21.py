"""Create frozen production manifests and offline Qwen conditions; never launch training."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from verl_distill.data.qwen_image21 import (
    atomic_json, canonical_hash, read_manifest, sha256, validate_complete, within,
)
from verl_distill.models.qwen_image21.modeling import (
    condition_key, encode_condition, load_condition_pipeline, model_identity,
    require_qwen_runtime, save_condition, open_reused_cache, condition_entry_path,
)


def snapshot(args):
    out = Path(args.snapshot_dir)
    if out.exists():
        raise FileExistsError(f"Use a fresh immutable snapshot directory: {out}")
    with Path(args.prompts_csv).open(encoding="utf-8-sig", newline="") as f:
        prompts = list(csv.DictReader(f))
    if len(prompts) != 64 or len({p["prompt_id"] for p in prompts}) != 64:
        raise ValueError("Expected 64 unique evaluation prompts")
    evaluation = []
    for p in prompts:
        w, h = int(p["width"]), int(p["height"])
        if min(w, h) <= 0 or w % 32 or h % 32 or not p["prompt"]:
            raise ValueError(f"Invalid evaluation row: {p['prompt_id']}")
        seed = int(canonical_hash({"prompt_id": p["prompt_id"]})[:16], 16) % 2**63
        evaluation.append({"id": p["prompt_id"], "kind": "t2i", "prompt": p["prompt"],
                           "width": w, "height": h, "seed": seed,
                           "reference_images": [], "reference_sha256": []})
    eval_texts = {p["prompt"] for p in prompts}
    records, rejected, excluded, seen = [], [], [], set()
    root = Path(args.output_root).resolve()
    # Only production output trees; no recursive search through smoke/debug directories.
    paths = [path for kind in ("t2i", "edit")
             for path in sorted((root / kind).glob("*/*/complete.json"))]
    def validate_path(path):
        try:
            return validate_complete(path, root, args.reference_root,
                                     verify_hash=not args.fast_index), None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return None, {"path": str(path), "error": str(exc)}
    # IO-bound hashing; preserve deterministic record order with executor.map.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (r, error) in enumerate(pool.map(validate_path, paths), 1):
            if error is not None:
                rejected.append(error)
            else:
                key = (r["kind"], r["id"])
                if key in seen:
                    raise ValueError(f"Duplicate sample {key}")
                seen.add(key)
                if r["prompt"] in eval_texts:
                    excluded.append(key)
                else:
                    records.append(r)
            if index % 250 == 0 or index == len(paths):
                print(f"snapshot verified {index}/{len(paths)} rejected={len(rejected)}", flush=True)
    if not records:
        raise ValueError("No valid completed training samples")
    out.mkdir(parents=True)
    common = {"schema": 1, "output_root": str(root), "reference_root": str(Path(args.reference_root).resolve()),
              "prompts_csv_sha256": sha256(args.prompts_csv)}
    for name, purpose, rows in (("train.json", "train", records), ("eval.json", "eval", evaluation)):
        atomic_json(out / name, {**common, "purpose": purpose, "records": rows,
                                "records_sha256": canonical_hash(rows),
                                "payloads_verified": not args.fast_index})
    (out / "complex_prompt.csv").write_bytes(Path(args.prompts_csv).read_bytes())
    atomic_json(out / "summary.json", {"train": len(records), "eval": len(evaluation),
                "kind_counts": dict(Counter(r["kind"] for r in records)),
                "reference_counts": dict(Counter(str(len(r["reference_images"])) for r in records)),
                "sizes": dict(Counter(f"{r['width']}x{r['height']}" for r in records)),
                "prompt_length_max": max(len(r["prompt"]) for r in records),
                "eval_excluded": excluded, "rejected": rejected, "fast_index": args.fast_index})
    print(f"Snapshot ready: {out}; train={len(records)} rejected={len(rejected)}")


def cache(args):
    require_qwen_runtime()
    train, evaluation = read_manifest(args.manifest), read_manifest(args.eval_manifest)
    if train["purpose"] != "train" or evaluation["purpose"] != "eval":
        raise ValueError("Incorrect manifest purposes")
    root = Path(args.cache_dir)
    # Never overwrite a published cache while a training process may be reading it.
    if (root / "index.json").exists():
        raise FileExistsError("Published cache exists; choose a new cache directory")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid cache shard id/count")
    identity = model_identity(args.model)
    hashes = {"train": sha256(args.manifest), "eval": sha256(args.eval_manifest)}
    rows = train["records"] + evaluation["records"]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    descriptor = ({"root": str(Path(args.reuse_cache).resolve()),
                   "index_sha256": sha256(Path(args.reuse_cache) / "index.json")}
                  if args.reuse_cache else None)
    reused = open_reused_cache(descriptor, identity)
    pipe = None  # Reused conditions need no GPU encoding.
    entries = {}
    for index, row in enumerate(rows):
        if index % args.num_shards != args.shard_id:
            continue
        key = condition_key(row, identity)
        if reused and key in reused[1]["entries"]:
            entries[key] = {**reused[1]["entries"][key], "source": "base"}
            continue
        if pipe is None:
            pipe = load_condition_pipeline(args.model, device)
        condition = encode_condition(pipe, row, args.reference_root or train["reference_root"], device)
        key, entry = save_condition(root, row, identity, condition)
        entries[key] = entry
        print(f"cached {index+1}/{len(rows)} {row['kind']}/{row['id']}", flush=True)
    atomic_json(root / f"shard-{args.shard_id:05d}.json", {
        "schema": 1, "num_shards": args.num_shards, "model_identity": identity,
        "manifest_hashes": hashes, "entries": entries, "reused_cache": descriptor})


def publish(args):
    if args.num_shards < 1:
        raise ValueError("num_shards must be positive")
    root = Path(args.cache_dir)
    if (root / "index.json").exists():
        raise FileExistsError("Cache already published")
    train, evaluation = read_manifest(args.manifest), read_manifest(args.eval_manifest)
    hashes = {"train": sha256(args.manifest), "eval": sha256(args.eval_manifest)}
    entries, identity, descriptor = {}, None, None
    for n in range(args.num_shards):
        shard = json.loads((root / f"shard-{n:05d}.json").read_text())
        if shard["num_shards"] != args.num_shards or shard["manifest_hashes"] != hashes:
            raise ValueError("Incompatible cache shard")
        if identity is not None and identity != shard["model_identity"]:
            raise ValueError("Cache workers used different model assets")
        if n and descriptor != shard.get("reused_cache"):
            raise ValueError("Cache workers used different reuse provenance")
        descriptor = shard.get("reused_cache")
        identity = shard["model_identity"]
        if entries.keys() & shard["entries"].keys():
            raise ValueError("Duplicate condition cache ownership")
        entries.update(shard["entries"])
    expected = {condition_key(r, identity) for r in train["records"] + evaluation["records"]}
    if set(entries) != expected:
        raise ValueError("Condition cache coverage mismatch")
    reused = open_reused_cache(descriptor, identity)
    def verify_entry(item):
        key, entry = item
        path = condition_entry_path(root, key, entry, reused)
        # Base index is immutable/pinned; payload still hash-checked on first training use.
        if entry.get("source") == "base":
            if not path.is_file():
                raise FileNotFoundError(path)
            return
        if sha256(path) != entry["sha256"]:
            raise ValueError("Corrupt condition cache payload")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, _ in enumerate(pool.map(verify_entry, entries.items()), 1):
            if index % 500 == 0 or index == len(entries):
                print(f"cache verified {index}/{len(entries)}", flush=True)
    atomic_json(root / "index.json", {"schema": 1, "model_identity": identity,
                "manifest_hashes": hashes, "entries": entries, "reused_cache": descriptor})
    print(f"Published {len(entries)} conditions")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    s = commands.add_parser("snapshot")
    s.add_argument("--output-root", required=True)
    s.add_argument("--reference-root", required=True)
    s.add_argument("--prompts-csv", required=True)
    s.add_argument("--snapshot-dir", required=True)
    s.add_argument("--fast-index", action="store_true", help="Skip payload hashing at indexing only")
    for command in ("cache", "publish"):
        c = commands.add_parser(command)
        c.add_argument("--manifest", required=True)
        c.add_argument("--eval-manifest", required=True)
        c.add_argument("--cache-dir", required=True)
        c.add_argument("--num-shards", type=int, default=1)
        if command == "cache":
            c.add_argument("--model", required=True)
            c.add_argument("--reuse-cache", default=None)
            c.add_argument("--reference-root")
            c.add_argument("--device", default="cuda:0")
            c.add_argument("--shard-id", type=int, default=0)
    args = p.parse_args()
    {"snapshot": snapshot, "cache": cache, "publish": publish}[args.command](args)


if __name__ == "__main__":
    main()
