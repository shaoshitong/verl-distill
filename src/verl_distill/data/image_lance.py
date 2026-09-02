from __future__ import annotations

import json
import logging
import os
import random
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torchvision import transforms

logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_ASPECT_BUCKETS = [
    (1080, 1080),
    (810, 1440),
    (1440, 810),
]


def _pick_bucket(img_w: int, img_h: int, buckets: list[tuple[int, int]]) -> tuple[int, int]:
    img_ratio = img_w / max(img_h, 1)
    best = buckets[0]
    best_diff = float("inf")
    for bucket_h, bucket_w in buckets:
        bucket_ratio = bucket_w / max(bucket_h, 1)
        diff = abs(img_ratio - bucket_ratio)
        if diff < best_diff:
            best = (bucket_h, bucket_w)
            best_diff = diff
    return best


class ResizeCover:
    def __init__(self, height: int, width: int):
        self.height = int(height)
        self.width = int(width)

    def __call__(self, image: Image.Image) -> Image.Image:
        scale = max(
            self.width / max(image.width, 1),
            self.height / max(image.height, 1),
        )
        resized_w = max(self.width, int(round(image.width * scale)))
        resized_h = max(self.height, int(round(image.height * scale)))
        resample = getattr(Image, "Resampling", Image).BILINEAR
        return image.resize((resized_w, resized_h), resample)


def _identity_image(image: Image.Image) -> Image.Image:
    return image


def _make_processor(
    height: int,
    width: int,
    *,
    center_crop: bool = True,
    random_flip: bool = False,
) -> transforms.Compose:
    return transforms.Compose(
        [
            ResizeCover(height, width),
            transforms.CenterCrop((height, width))
            if center_crop
            else transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip()
            if random_flip
            else transforms.Lambda(_identity_image),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


class Text2ImageLanceDataset(torch.utils.data.Dataset):
    """Reads text/image pairs from single-file Lance shards.

    The source shard format matches the Z-Image DMD trainer: each .lance file has
    `inputs` JSON and `images` list<bytes> columns.
    """

    def __init__(
        self,
        lance_data_dir: str,
        height: int = 1024,
        width: int = 1024,
        center_crop: bool = True,
        random_flip: bool = False,
        index_path: str | None = None,
        datasets_repeat: int = 1,
        aspect_buckets: list[list[int]] | list[tuple[int, int]] | None = None,
        prompt_override_path: str | None = None,
        dataset_json_path: str | None = None,
    ):
        self.lance_data_dir = str(Path(lance_data_dir).resolve()) if lance_data_dir else ""
        self.height = int(height)
        self.width = int(width)
        self.datasets_repeat = int(datasets_repeat)
        self.prompt_map: dict[tuple[str, int], str] | None = None
        self.dataset_json_path = dataset_json_path

        manifest_samples = None
        if dataset_json_path:
            manifest_samples = self._load_manifest(dataset_json_path)

        if not self.lance_data_dir:
            raise ValueError("lance_data_dir must be set directly or in dataset_json_path")
        if not Path(self.lance_data_dir).exists():
            raise FileNotFoundError(f"lance_data_dir not found: {self.lance_data_dir}")

        self.prompt_override = None
        if prompt_override_path and Path(prompt_override_path).exists():
            with open(prompt_override_path, "r", encoding="utf-8") as handle:
                self.prompt_override = json.load(handle)
            total_overrides = sum(len(value) for value in self.prompt_override.values())
            logger.info("Loaded %d prompt overrides from %s", total_overrides, prompt_override_path)

        if aspect_buckets:
            self.aspect_buckets = [(int(h), int(w)) for h, w in aspect_buckets]
            self._bucket_processors = {
                bucket: _make_processor(
                    bucket[0],
                    bucket[1],
                    center_crop=center_crop,
                    random_flip=random_flip,
                )
                for bucket in self.aspect_buckets
            }
            self.image_processor = None
            logger.info("Lance aspect buckets enabled: %s", self.aspect_buckets)
        else:
            self.aspect_buckets = None
            self._bucket_processors = None
            self.image_processor = _make_processor(
                self.height,
                self.width,
                center_crop=center_crop,
                random_flip=random_flip,
            )

        self.samples: list[tuple[str, int]] = []
        if manifest_samples is not None:
            self.samples = list(manifest_samples)
        else:
            shards = sorted(
                path.name
                for path in Path(self.lance_data_dir).iterdir()
                if path.name.endswith(".lance")
            )
            if not shards:
                raise FileNotFoundError(f"No .lance files found in {self.lance_data_dir}")
            shard_set = set(shards)
        if manifest_samples is None and index_path:
            with open(index_path, "r", encoding="utf-8") as handle:
                index_map = json.load(handle)
            for shard_file in sorted(index_map):
                if shard_file not in shard_set:
                    continue
                for row_idx in index_map[shard_file]:
                    self.samples.append((shard_file, int(row_idx)))
        elif manifest_samples is None:
            from lance.file import LanceFileReader

            for shard_file in shards:
                reader = LanceFileReader(os.path.join(self.lance_data_dir, shard_file))
                for row_idx in range(reader.metadata().num_rows):
                    self.samples.append((shard_file, row_idx))

        if not self.samples:
            raise ValueError(f"No valid Lance samples found in {self.lance_data_dir}")

        logger.info(
            "Loaded %d samples from lance_data_dir=%s (index=%s)",
            len(self.samples),
            self.lance_data_dir,
            dataset_json_path or index_path or "none",
        )

        self._shard_cache_file: str | None = None
        self._shard_cache_reader = None
        self._shard_cache_pid = os.getpid()

    def __len__(self) -> int:
        return len(self.samples) * self.datasets_repeat

    def _read_row(self, shard_file: str, row_idx: int):
        from lance.file import LanceFileReader

        pid = os.getpid()
        if self._shard_cache_file != shard_file or self._shard_cache_pid != pid:
            self._shard_cache_reader = LanceFileReader(
                os.path.join(self.lance_data_dir, shard_file)
            )
            self._shard_cache_file = shard_file
            self._shard_cache_pid = pid
        return self._shard_cache_reader.read_range(row_idx, 1).to_table()

    def _load_manifest(self, dataset_json_path: str) -> list[tuple[str, int]]:
        with open(dataset_json_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError("Lance dataset manifest must be a JSON object")
        manifest_data_dir = manifest.get("lance_data_dir")
        if manifest_data_dir:
            self.lance_data_dir = str(Path(manifest_data_dir).resolve())
        rows = manifest.get("samples")
        if not isinstance(rows, list):
            raise ValueError("Lance dataset manifest must contain a samples list")

        samples: list[tuple[str, int]] = []
        prompt_map: dict[tuple[str, int], str] = {}
        skipped = 0
        for item in rows:
            if not isinstance(item, dict):
                skipped += 1
                continue
            shard_file = item.get("shard_file", item.get("shard"))
            row_idx = item.get("row_idx", item.get("row"))
            prompt = item.get("prompt", item.get("text"))
            if not isinstance(shard_file, str) or row_idx is None:
                skipped += 1
                continue
            sample = (shard_file, int(row_idx))
            samples.append(sample)
            if isinstance(prompt, str) and prompt.strip():
                prompt_map[sample] = prompt.strip()
        if not samples:
            raise ValueError(f"No valid samples found in manifest {dataset_json_path}")
        self.prompt_map = prompt_map
        logger.info(
            "Loaded lance dataset manifest from %s: samples=%d prompts=%d skipped=%d",
            dataset_json_path,
            len(samples),
            len(prompt_map),
            skipped,
        )
        return samples

    @staticmethod
    def _extract_caption_and_meta(inputs_json) -> tuple[str | None, dict | None]:
        try:
            parsed = json.loads(inputs_json)
        except (json.JSONDecodeError, TypeError):
            return None, None
        caption = None
        img_meta = None
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            if entry.get("type") == "text":
                caption = entry.get("text", "")
            elif entry.get("type") == "image":
                img_meta = entry
        if isinstance(caption, str) and caption.strip():
            return caption.strip(), img_meta
        return None, img_meta

    def _get_sample(self, sample_idx: int):
        shard_file, row_idx = self.samples[sample_idx]
        table = self._read_row(shard_file, row_idx)
        caption, img_meta = self._extract_caption_and_meta(table["inputs"][0].as_py())
        if self.prompt_map is not None:
            caption = self.prompt_map.get((shard_file, row_idx), caption)
        elif self.prompt_override and shard_file in self.prompt_override:
            caption = self.prompt_override[shard_file].get(str(row_idx), caption)
        caption = caption or "an image"
        images = table["images"][0].as_py()
        if not images:
            raise RuntimeError(f"No image bytes in shard={shard_file} row={row_idx}")
        image = Image.open(BytesIO(images[0])).convert("RGB")
        if self.aspect_buckets:
            if img_meta:
                orig_w = int(img_meta.get("width", image.width))
                orig_h = int(img_meta.get("height", image.height))
            else:
                orig_w, orig_h = image.width, image.height
            bucket = _pick_bucket(orig_w, orig_h, self.aspect_buckets)
            image_tensor = self._bucket_processors[bucket](image)
        else:
            image_tensor = self.image_processor(image)
        return {
            "text": caption,
            "image": image_tensor,
            "z": torch.empty(0),
        }

    def __getitem__(self, index: int):
        sample_idx = int(index) % len(self.samples)
        for attempt in range(3):
            try:
                if attempt:
                    sample_idx = random.randint(0, len(self.samples) - 1)
                return self._get_sample(sample_idx)
            except Exception as exc:
                logger.warning(
                    "Failed to load Lance sample %d (attempt %d): %s",
                    sample_idx,
                    attempt,
                    exc,
                )
        return self._get_sample(random.randint(0, len(self.samples) - 1))
