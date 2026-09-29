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
    required = {"model_revision": MODEL_REVISION, "num_inference_steps": 40,
                "true_cfg_scale": 1.0, "vae_tiling": False,
                "model_cpu_offload": True, "use_kv_cache": True}
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


def cached_token_metadata(records, cache_root):
    """Read cache shapes and small image mask, never read large tensor storage or use CUDA.

    Returns manifest-aligned records plus index provenance. Intended as an offline
    preparation step; callers should persist this small result rather than scan at
    every training startup. Tensor content checks remain ConditionStore's job.
    """
    from verl_distill.models.qwen_image21.modeling import (
        condition_key, condition_entry_path, open_reused_cache,
    )
    root = Path(cache_root)
    index_path = root / 'index.json'
    index = json.loads(index_path.read_text())
    reused = open_reused_cache(index.get('reused_cache'), index['model_identity'])
    result = []
    for row in records:
        key = condition_key(row, index['model_identity'])
        entry = index['entries'][key]
        path = condition_entry_path(root, key, entry, reused)
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if saved['key'] != key:
            raise ValueError('Cached condition key mismatch')
        condition = saved['condition']
        shapes = condition['img_shapes'][0]
        target = (row['height'] // 16) * (row['width'] // 16)
        counts = [int(t) * int(h) * int(w) for t, h, w in shapes]
        if len(counts) != len(row['reference_images']) + 1 or counts[-1] != target:
            raise ValueError('Cached image shapes do not match manifest')
        vlm_length = int(condition['encoder_hidden_states'].shape[1])
        # mmap maps large storages lazily; only the tiny boolean mask is read.
        text = int((~condition['img_mask'][0, :vlm_length]).sum().item())
        refs = sum(counts[:-1])
        result.append({'id': row['id'], 'kind': row['kind'],
                       'width': row['width'], 'height': row['height'],
                       'reference_count': len(row['reference_images']),
                       'target_tokens': target, 'reference_tokens': refs,
                       'text_tokens': text, 'total_tokens': target + refs + text})
    return {'schema': 1, 'records_sha256': canonical_hash(records),
            'cache_index_sha256': sha256(index_path), 'metadata': result,
            'metadata_sha256': canonical_hash(result)}


class HomogeneousRankCursor:
    """Opt-in v2: global length groups with hard kind/reference-count separation.

    No random pool tails. Ratio bounds are enforced while forming groups. A
    partial group is padded using its own members only; every real slot survives.
    Padding is explicit in epoch_stats and repeated indices count toward work.
    Old checkpoints require migrate_from_legacy(), never implicit migration.
    """
    ordering = 'homogeneous_v2'

    def __init__(self, metadata, rank, world_size, seed, *, max_token_ratio=1.25,
                 max_resolution_ratio=1.5):
        if not metadata or not 0 <= rank < world_size or world_size < 1:
            raise ValueError('Invalid dataset/rank/world size')
        if max_token_ratio < 1 or max_resolution_ratio < 1:
            raise ValueError('Similarity ratios must be >= 1')
        self.metadata = metadata
        self.size, self.rank, self.world_size, self.seed = len(metadata), rank, world_size, seed
        self.max_token_ratio, self.max_resolution_ratio = max_token_ratio, max_resolution_ratio
        self.metadata_hash = canonical_hash(metadata)
        for m in metadata:
            if m['kind'] not in ('t2i', 'edit') or (m['kind'] == 't2i') != (m['reference_count'] == 0):
                raise ValueError('kind/reference count mismatch')
            if any(m[k] <= 0 for k in ('target_tokens', 'text_tokens', 'total_tokens', 'width', 'height')):
                raise ValueError('Nonpositive token/dimension metadata')
            if m['reference_tokens'] < 0 or m['total_tokens'] != m['target_tokens'] + m['reference_tokens'] + m['text_tokens']:
                raise ValueError('Invalid token metadata')
        self.epoch = self.position = 0
        self._indices = None
        self._migration_pending = None
        self.migration_audit = None
        self.epoch_stats = {}

    def _global_order(self):
        pending = list(range(self.size)) if self._migration_pending is None else list(self._migration_pending)
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        # Shuffle before stable cost sorting so equal-cost records vary by epoch.
        pending = [pending[j] for j in torch.randperm(len(pending), generator=rng).tolist()]
        partitions = {}
        for i in pending:
            m = self.metadata[i]
            partitions.setdefault((m['kind'], m['reference_count']), []).append(i)
        groups, padding = [], 0
        dims = ('target_tokens', 'reference_tokens', 'total_tokens', 'width', 'height')
        def close(group):
            nonlocal padding
            if group:
                count = (-len(group)) % self.world_size
                padding += count
                padded = group + [group[j % len(group)] for j in range(count)]
                groups.append(padded)
        import math
        def band(value, ratio):
            return value if ratio == 1 else math.floor(math.log(max(1, value), ratio))
        def sort_key(i):
            m = self.metadata[i]
            # Coarse resolution bands first avoid interleaving portrait/landscape
            # at every nearly-equal target-token value (and excessive padding).
            return (band(m['target_tokens'], self.max_token_ratio),
                    band(m['width'], self.max_resolution_ratio),
                    band(m['height'], self.max_resolution_ratio),
                    band(m['reference_tokens'], self.max_token_ratio), m['total_tokens'])
        for partition in sorted(partitions):
            ordered = sorted(partitions[partition], key=sort_key)
            group, lows, highs = [], {}, {}
            for i in ordered:
                m = self.metadata[i]
                proposed_low = {k: min(lows.get(k, m[k]), m[k]) for k in dims}
                proposed_high = {k: max(highs.get(k, m[k]), m[k]) for k in dims}
                fits = all(proposed_high[k] <= proposed_low[k] * (
                    self.max_resolution_ratio if k in ('width', 'height') else self.max_token_ratio)
                    for k in dims)
                if group and not fits:
                    close(group)
                    group, lows, highs = [], {}, {}
                group.append(i)
                lows = {k: min(lows.get(k, m[k]), m[k]) for k in dims}
                highs = {k: max(highs.get(k, m[k]), m[k]) for k in dims}
                if len(group) == self.world_size:
                    close(group)
                    group, lows, highs = [], {}, {}
            close(group)
        order = []
        for j in torch.randperm(len(groups), generator=rng).tolist():
            group = groups[j]
            order.extend(group[k] for k in torch.randperm(len(group), generator=rng).tolist())
        self.epoch_stats = {'input_slots': len(pending), 'padded_slots': padding,
                            'output_slots': len(order), 'global_batches': len(groups),
                            'padding_fraction': padding / len(order) if order else 0.0,
                            'unique_input_samples': len(set(pending)),
                            'mixed_kind_batches': 0}
        return order

    def _permutation(self):
        return self._global_order()[self.rank::self.world_size]

    def next_index(self):
        if self._indices is None:
            self._indices = self._permutation()
        if self.position == len(self._indices):
            self.epoch += 1
            self.position = 0
            self._migration_pending = None
            self._indices = self._permutation()
        value = self._indices[self.position]
        self.position += 1
        return value

    def state_dict(self):
        return {'ordering': self.ordering,
                **{k: getattr(self, k) for k in ('size', 'rank', 'world_size', 'seed', 'epoch', 'position',
                    'metadata_hash', 'max_token_ratio', 'max_resolution_ratio')},
                'migration_pending': self._migration_pending, 'migration_audit': self.migration_audit}

    def load_state_dict(self, state):
        if state.get('ordering') != self.ordering:
            raise ValueError('Legacy migration requires explicit migrate_from_legacy()')
        for k in ('size', 'rank', 'world_size', 'seed', 'metadata_hash', 'max_token_ratio', 'max_resolution_ratio'):
            if state[k] != getattr(self, k):
                raise ValueError(f'Homogeneous cursor {k} changed')
        self.epoch, self.position = int(state['epoch']), int(state['position'])
        self._migration_pending = state.get('migration_pending')
        if self._migration_pending is not None and any(type(i) is not int or not 0 <= i < self.size for i in self._migration_pending):
            raise ValueError('Invalid migration suffix indices')
        self.migration_audit = state.get('migration_audit')
        self._indices = self._permutation()
        if self.epoch < 0 or not 0 <= self.position <= len(self._indices):
            raise ValueError('Invalid homogeneous cursor position')

    def migrate_from_legacy(self, state, *, legacy_costs=None, allow=False):
        """Explicit same-world migration; reconstruct old unconsumed global suffix.

        Caller must verify all saved rank cursors have equal epoch/position and
        identical ordering contracts before calling this on each rank. Different
        world sizes or manifests require a separately reviewed migration.
        """
        if not allow:
            raise ValueError('Migration requires allow=True and audited saved cursors')
        for k in ('size', 'rank', 'world_size', 'seed'):
            if state[k] != getattr(self, k):
                raise ValueError(f'Legacy migration {k} changed')
        if state.get('ordering', 'random') not in ('random', 'bucketed_v1'):
            raise ValueError('Unsupported legacy ordering')
        if (state.get('ordering', 'random') == 'bucketed_v1') != (legacy_costs is not None):
            raise ValueError('Supply exact legacy costs only for bucketed_v1')
        suffixes = []
        for rank in range(self.world_size):
            old = RankCursor(self.size, rank, self.world_size, self.seed, costs=legacy_costs,
                             bucket_batches=state.get('bucket_batches', 64))
            old.load_state_dict({**state, 'rank': rank})
            suffixes.append(old._indices[old.position:])
        pending = [i for group in zip(*suffixes, strict=True) for i in group]
        self.epoch, self.position = state['epoch'], 0
        if len(pending) != ((self.size + self.world_size - 1)//self.world_size - state['position']) * self.world_size:
            raise ValueError('Legacy migration lost source suffix slots')
        self._migration_pending = pending
        self.migration_audit = {'source_state': state, 'pending_slots': len(pending),
                                'pending_sha256': canonical_hash(pending),
                                'consumed_slots': state['position'] * self.world_size}
        self._indices = self._permutation()
