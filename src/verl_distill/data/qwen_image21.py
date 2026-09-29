"""Immutable, complete-only Qwen production snapshots and deterministic rank cursors."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from verl_distill.models.qwen_image21.configuration import MODEL_REVISION


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def within(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Path escapes data root: {relative}")
    return result


def validate_complete(path, output_root, reference_root, *, verify_hash=True):
    path = Path(path)
    meta = json.loads(path.read_text())
    required = {"num_inference_steps": 40, "true_cfg_scale": 1.0, "vae_tiling": False,
                "model_cpu_offload": True, "use_kv_cache": True}
    if MODEL_REVISION:
        required["model_revision"] = MODEL_REVISION
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise ValueError(f"{path}: {key} != {expected}")
    if meta.get("kind") not in ("t2i", "edit") or not isinstance(meta.get("prompt"), str):
        raise ValueError(f"Invalid prompt/kind: {path}")
    sample_id = str(meta["id"])
    expected_dir = within(output_root, f"{meta['kind']}/{sample_id[:2]}/{sample_id}")
    if path.parent.resolve() != expected_dir:
        raise ValueError(f"Sample path/id mismatch: {path}")
    w, h = int(meta["width"]), int(meta["height"])
    if min(w, h) <= 0 or w % 32 or h % 32:
        raise ValueError(f"Invalid aligned dimensions: {path}")
    if not isinstance(meta.get("seed"), int) or not 0 <= meta["seed"] < 2**63:
        raise ValueError(f"Invalid seed: {path}")
    payloads = ("image.png", "initial_noise.pt", "x0_latent.pt")
    if set(meta.get("sha256", {})) != set(payloads):
        raise ValueError(f"Missing payload digests: {path}")
    for name in payloads:
        f = expected_dir / name
        if not f.is_file() or f.stat().st_size <= 0:
            raise ValueError(f"Missing or empty payload: {f}")
        if verify_hash and sha256(f) != meta["sha256"][name]:
            raise ValueError(f"Payload checksum mismatch: {f}")
    refs = meta.get("reference_images", [])
    if not isinstance(refs, list) or (meta["kind"] == "edit" and not refs):
        raise ValueError(f"Invalid reference list: {path}")
    if meta["kind"] == "t2i" and refs:
        raise ValueError(f"Text-to-image sample unexpectedly has references: {path}")
    reference_hashes = []
    for ref in refs:
        f = within(reference_root, ref)
        if not f.is_file():
            raise FileNotFoundError(f)
        reference_hashes.append(sha256(f))
    return {
        "id": sample_id, "kind": meta["kind"], "prompt": meta["prompt"],
        "width": w, "height": h, "seed": meta["seed"],
        "reference_images": refs, "reference_sha256": reference_hashes,
        "payload_dir": str(expected_dir.relative_to(Path(output_root).resolve())),
        "payload_sha256": meta["sha256"], "complete_sha256": sha256(path),
        "model_revision": meta["model_revision"],
    }


def read_manifest(path):
    document = json.loads(Path(path).read_text())
    if document.get("schema") != 1 or not document.get("records"):
        raise ValueError(f"Invalid/empty Qwen manifest: {path}")
    records = document["records"]
    if canonical_hash(records) != document.get("records_sha256"):
        raise ValueError("Manifest record digest mismatch")
    keys = [(r["kind"], r["id"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate (kind,id) in manifest")
    return document


class QwenPairDataset:
    def __init__(self, manifest, output_root=None, *, verify_payloads=True):
        self.document = read_manifest(manifest)
        if self.document.get("purpose") != "train":
            raise ValueError("Expected a training snapshot")
        self.records = self.document["records"]
        self.root = Path(output_root or self.document["output_root"])
        self.verify_payloads = verify_payloads
        self.verified = set()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        directory = within(self.root, row["payload_dir"])
        if index not in self.verified:
            if sha256(directory / "complete.json") != row["complete_sha256"]:
                raise ValueError(f"Completion metadata changed: {directory}")
            if self.verify_payloads:
                for name, digest in row["payload_sha256"].items():
                    if sha256(directory / name) != digest:
                        raise ValueError(f"Training payload changed: {directory / name}")
            self.verified.add(index)
        tensors = {}
        for key, name in (("noise", "initial_noise.pt"), ("clean", "x0_latent.pt")):
            tensor = torch.load(directory / name, map_location="cpu", weights_only=True)
            shape = (1, 64, 1, row["height"] // 16, row["width"] // 16)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f"Invalid {key} latent shape: {directory}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite {key} latent: {directory}")
            tensors[key] = tensor.float().flatten(2).transpose(1, 2).contiguous()
        return {"record": row, **tensors}


def estimated_training_tokens(record):
    # References are encoded at approximately 2048^2 pixels (16384 latent tokens).
    # Prompt length is only a cheap tie-break proxy; no payload IO at sampler setup.
    return ((record["height"] // 16) * (record["width"] // 16)
            + len(record["reference_images"]) * 16384 + max(1, len(record["prompt"]) // 3))


class RankCursor:
    """Deterministic global batch bucketing, preserving coverage and resumable suffixes."""
    def __init__(self, size, rank, world_size, seed, *, costs=None, bucket_batches=64,
                 allow_ordering_migration=False):
        if size < world_size or not 0 <= rank < world_size:
            raise ValueError("Training snapshot needs at least one distinct sample per rank")
        if costs is not None and (len(costs) != size or any(c <= 0 for c in costs)):
            raise ValueError("Expected one positive sample cost per record")
        if bucket_batches < 1:
            raise ValueError("bucket_batches must be positive")
        self.size, self.rank, self.world_size, self.seed = size, rank, world_size, seed
        self.costs, self.bucket_batches = costs, bucket_batches
        self.cost_hash = canonical_hash(costs) if costs is not None else None
        self.allow_ordering_migration = allow_ordering_migration
        self.bucket_start = 0
        self.epoch = self.position = 0
        self._indices = None

    def _permutation(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(
            self.seed + self.epoch)).tolist()
        padded = ((self.size + self.world_size - 1) // self.world_size) * self.world_size
        order += order[:padded - self.size]
        if self.costs is not None:
            split = self.bucket_start * self.world_size
            prefix, pending = order[:split], order[split:]
            groups = []
            pool_size = self.world_size * self.bucket_batches
            for offset in range(0, len(pending), pool_size):
                pool = sorted(pending[offset:offset+pool_size], key=lambda i: self.costs[i])
                groups.extend(pool[i:i+self.world_size] for i in range(0, len(pool), self.world_size))
            rng = torch.Generator().manual_seed(self.seed + self.epoch + 1000003)
            # Shuffle batch order: no small-to-large curriculum or persistent rank bias.
            shuffled = []
            for i in torch.randperm(len(groups), generator=rng).tolist():
                group = groups[i]
                shuffled.extend(group[j] for j in torch.randperm(len(group), generator=rng).tolist())
            order = prefix + shuffled
        return order[self.rank::self.world_size]

    def next_index(self):
        if self._indices is None:
            self._indices = self._permutation()
        if self.position == len(self._indices):
            self.epoch += 1
            self.position = self.bucket_start = 0
            self._indices = self._permutation()
        index = self._indices[self.position]
        self.position += 1
        return index

    def state_dict(self):
        state = {k: getattr(self, k) for k in ("size", "rank", "world_size", "seed", "epoch", "position")}
        if self.costs is not None:
            state.update(ordering="bucketed_v1", cost_hash=self.cost_hash,
                         bucket_batches=self.bucket_batches, bucket_start=self.bucket_start)
        return state

    def load_state_dict(self, state):
        for k in ("size", "rank", "world_size", "seed"):
            if state[k] != getattr(self, k):
                raise ValueError(f"Data cursor {k} changed")
        ordering = state.get("ordering", "random")
        if ordering == "random" and self.costs is not None:
            if not self.allow_ordering_migration:
                raise ValueError("Random -> bucketed cursor migration must be explicit")
            self.bucket_start = state["position"]
        elif ordering == "bucketed_v1" and self.costs is not None:
            if state["cost_hash"] != self.cost_hash or state["bucket_batches"] != self.bucket_batches:
                raise ValueError("Bucket cost/pool configuration changed")
            self.bucket_start = state["bucket_start"]
        elif ordering != "random" or self.costs is not None:
            raise ValueError("Unsupported cursor ordering migration")
        self.epoch, self.position = int(state["epoch"]), int(state["position"])
        if not 0 <= self.bucket_start <= self.position <= (self.size + self.world_size - 1)//self.world_size:
            raise ValueError("Invalid restored data cursor")
        self._indices = self._permutation()
