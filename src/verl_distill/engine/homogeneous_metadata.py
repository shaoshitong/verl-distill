"""Validate the exact cached-token metadata before selecting a homogeneous sampler."""
import json
import math
from pathlib import Path
from verl_distill.data.qwen_image21 import canonical_hash, sha256


def validate_metadata(document, records, cache_index_hash):
    if type(document.get("schema")) is not int or document["schema"] != 1:
        raise ValueError("Unsupported token metadata schema")
    if document.get("records_sha256") != canonical_hash(records):
        raise ValueError("Token metadata records_sha256 mismatch")
    if document.get("cache_index_sha256") != cache_index_hash:
        raise ValueError("Token metadata cache_index_sha256 mismatch")
    metadata = document.get("metadata")
    if not isinstance(metadata, list) or len(metadata) != len(records):
        raise ValueError("Token metadata length mismatch")
    if document.get("metadata_sha256") != canonical_hash(metadata):
        raise ValueError("Token metadata metadata_sha256 mismatch")
    for row, item in zip(records, metadata, strict=True):
        if not isinstance(item, dict) or item.get("id") != row["id"]:
            raise ValueError("Token metadata IDs/order mismatch")
        expected = {k: row[k] for k in ("kind", "width", "height")}
        expected.update(reference_count=len(row["reference_images"]),
                        target_tokens=(row["height"]//16)*(row["width"]//16))
        if any(item.get(k) != v for k, v in expected.items()):
            raise ValueError("Token metadata geometry/kind mismatch")
        for key in ("width", "height", "reference_count", "target_tokens", "reference_tokens", "text_tokens", "total_tokens"):
            if type(item.get(key)) is not int:
                raise ValueError(f"Token metadata {key} must be an integer")
    return metadata


def load_metadata(path, records, cache_root):
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict):
        raise ValueError("Token metadata must be a JSON object")
    metadata = validate_metadata(document, records, sha256(Path(cache_root)/"index.json"))
    return metadata, {"metadata_file_sha256": sha256(path),
        **{k: document[k] for k in ("schema", "records_sha256", "cache_index_sha256", "metadata_sha256")}}


def validate_resume(saved, reset_data_cursor=False):
    if saved["contract"]["runtime"].get("data_ordering", "random") != "homogeneous_v2":
        raise ValueError("Legacy to homogeneous_v2 resume requires a separately audited explicit migration; automatic reset is forbidden")
    if reset_data_cursor:
        raise ValueError("homogeneous_v2 resume cannot reset the dataset cursor")


def audit_legacy_cursors(states, world_size):
    if len(states) != world_size:
        raise ValueError("Missing rank cursor")
    base = {k: v for k, v in states[0].items() if k != "rank"}
    for rank, state in enumerate(states):
        if state.get("rank") != rank or state.get("world_size") != world_size:
            raise ValueError("Cursor rank/world mismatch")
        if {k: v for k, v in state.items() if k != "rank"} != base:
            raise ValueError("Saved rank cursors differ in epoch/position/ordering contract")
    if base.get("ordering") != "bucketed_v1":
        raise ValueError("Expected bucketed_v1 source")
    return canonical_hash(states)


def migrate_legacy_checkpoint(path, cursor, legacy_costs):
    import torch
    states = [torch.load(Path(path)/f"rank-{r:05d}.pt", map_location="cpu", weights_only=False)["cursor"]
              for r in range(cursor.world_size)]
    digest = audit_legacy_cursors(states, cursor.world_size)
    cursor.migrate_from_legacy(states[cursor.rank], legacy_costs=legacy_costs, allow=True)
    cursor.migration_audit.update(all_rank_cursors_sha256=digest,
        effect="Preserve unconsumed source suffix; reorder across ranks; add homogeneous padding",
        epoch_stats=dict(cursor.epoch_stats))
    return cursor.migration_audit
