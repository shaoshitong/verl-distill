from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.utils.data import DataLoader, DistributedSampler

from verl_distill.data.image_jsonl import Text2ImageJsonlDataset
from verl_distill.data.image_lance import Text2ImageLanceDataset
from verl_distill.data.ode_pair import OdePairDataset
from verl_distill.data.prompt_jsonl import TextPromptJsonlDataset


def build_dataset(config: Mapping[str, Any]):
    kind = str(config.get("format", "image_jsonl"))
    common = {
        "jsonl_path": config.get("manifest", ""),
        "prompt_key": config.get("prompt_key", "refined_prompt"),
        "datasets_repeat": int(config.get("repeat", 1)),
    }
    if kind == "image_jsonl":
        return Text2ImageJsonlDataset(
            **common,
            image_root=config.get("image_root", ""),
            image_key=config.get("image_key", "image"),
            height=int(config.get("height", config.get("resolution", 1024))),
            width=int(config.get("width", config.get("resolution", 1024))),
            center_crop=bool(config.get("center_crop", True)),
            random_flip=bool(config.get("random_flip", False)),
        )
    if kind in {"image_lance", "lance"}:
        return Text2ImageLanceDataset(
            lance_data_dir=config.get("lance_data_dir", config.get("image_root", "")),
            height=int(config.get("height", config.get("resolution", 1024))),
            width=int(config.get("width", config.get("resolution", 1024))),
            center_crop=bool(config.get("center_crop", True)),
            random_flip=bool(config.get("random_flip", False)),
            index_path=config.get("lance_index_path"),
            datasets_repeat=int(config.get("repeat", 1)),
            aspect_buckets=config.get("aspect_buckets"),
            prompt_override_path=config.get("prompt_override_path"),
            dataset_json_path=config.get("dataset_json_path"),
        )
    if kind == "prompt_jsonl":
        return TextPromptJsonlDataset(
            **common,
            prompt_keys=config.get("prompt_keys"),
            prompt_key_probs=config.get("prompt_key_probs"),
        )
    if kind == "ode_pair":
        return OdePairDataset(
            config.get("pair_dir", ""),
            reward_weighting=config.get("reward_weighting", "source_rank"),
            reward_weights=config.get("reward_weights"),
            rank_weight_strength=float(config.get("rank_weight_strength", 1.0)),
        )
    raise ValueError(f"Unsupported data format: {kind}")


def build_dataloader(
    dataset,
    *,
    rank: int = 0,
    world_size: int = 1,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            rank=rank,
            num_replicas=world_size,
            shuffle=shuffle,
        )
    return DataLoader(
        dataset,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=True,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
