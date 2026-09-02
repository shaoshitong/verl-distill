import json

import torch
from PIL import Image

from verl_distill.data import build_dataloader, build_dataset


def test_image_jsonl_uses_standard_mapping_batch(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (12, 10), color=(255, 0, 0)).save(image_root / "one.png")
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(
        json.dumps({"image": "one.png", "refined_prompt": "a red square"}) + "\n",
        encoding="utf-8",
    )
    dataset = build_dataset(
        {
            "format": "image_jsonl",
            "manifest": str(manifest),
            "image_root": str(image_root),
            "resolution": 8,
        }
    )
    sample = dataset[0]
    assert sample["text"] == "a red square"
    assert sample["image"].shape == (3, 8, 8)
    assert torch.all(sample["image"] >= -1)
    loader = build_dataloader(dataset, batch_size=1, num_workers=0)
    batch = next(iter(loader))
    assert batch["text"] == ["a red square"]
    assert batch["image"].shape == (1, 3, 8, 8)


def test_prompt_jsonl_uses_public_schema(tmp_path):
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text('{"prompt": "a mountain"}\n', encoding="utf-8")
    dataset = build_dataset(
        {"format": "prompt_jsonl", "manifest": str(manifest), "prompt_key": "prompt"}
    )
    assert dataset[0]["text"] == "a mountain"
    assert dataset[0]["image"].numel() == 0


def test_image_loader_supports_spawn_workers(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image_root / "one.png")
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(
        json.dumps({"image": "one.png", "refined_prompt": "black"}) + "\n",
        encoding="utf-8",
    )
    dataset = build_dataset(
        {
            "format": "image_jsonl",
            "manifest": str(manifest),
            "image_root": str(image_root),
            "resolution": 8,
        }
    )
    batch = next(iter(build_dataloader(dataset, batch_size=1, num_workers=1)))
    assert batch["image"].shape == (1, 3, 8, 8)
