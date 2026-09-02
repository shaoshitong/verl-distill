import json
import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class Text2ImageJsonlDataset(torch.utils.data.Dataset):
    """Dataset for JSONL rows with image relative path + prompt text.

    Expected row schema (default):
    - `image`: relative path under `image_root`
    - `refined_prompt`: prompt string
    """

    def __init__(
        self,
        jsonl_path,
        image_root,
        prompt_key="refined_prompt",
        image_key="image",
        height=1024,
        width=1024,
        center_crop=True,
        random_flip=False,
        datasets_repeat=1,
        image_extensions=("jpg", "jpeg", "png", "webp", "bmp"),
    ):
        self.height = height
        self.width = width
        self.datasets_repeat = int(datasets_repeat)

        self.jsonl_path = Path(jsonl_path).resolve()
        self.image_root = Path(image_root).resolve()
        self.prompt_key = str(prompt_key)
        self.image_key = str(image_key)

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")
        if not self.image_root.exists():
            raise FileNotFoundError(f"image_root not found: {self.image_root}")

        image_extensions = tuple(f".{ext.lower()}" for ext in image_extensions)

        self.image_paths = []
        self.prompts = []

        num_invalid_json = 0
        num_missing = 0
        num_invalid_prompt = 0
        num_invalid_image = 0

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    num_invalid_json += 1
                    continue

                if not isinstance(row, dict):
                    num_missing += 1
                    continue

                rel_path = row.get(self.image_key, None)
                prompt = row.get(self.prompt_key, None)

                if not isinstance(rel_path, str) or not rel_path.strip():
                    num_missing += 1
                    continue
                if not isinstance(prompt, str) or not prompt.strip():
                    num_invalid_prompt += 1
                    continue

                image_path = self._resolve_image_path(rel_path)
                if image_path is None:
                    num_invalid_image += 1
                    continue
                if image_path.suffix.lower() not in image_extensions:
                    num_invalid_image += 1
                    continue

                self.image_paths.append(str(image_path))
                self.prompts.append(prompt.strip())

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No valid (image, prompt) pairs found in jsonl={self.jsonl_path} with image_root={self.image_root}"
            )

        logger.info(
            "Loaded %d samples from jsonl=%s (image_root=%s). skipped: invalid_json=%d missing=%d invalid_prompt=%d invalid_image=%d",
            len(self.image_paths),
            self.jsonl_path,
            self.image_root,
            num_invalid_json,
            num_missing,
            num_invalid_prompt,
            num_invalid_image,
        )

        self.image_processor = transforms.Compose(
            [
                transforms.Resize(
                    min(height, width),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop((height, width))
                if center_crop
                else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else torch.nn.Identity(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def _resolve_image_path(self, rel_path: str):
        rel_path = rel_path.strip()
        if rel_path == "":
            return None

        candidate_rel_paths = [Path(rel_path)]

        # Some JSONL files store paths like "images/xxx.png" even when image_root
        # already points to ".../images". Strip one leading "images/" in that case.
        root_name = self.image_root.name
        p = Path(rel_path)
        if len(p.parts) >= 2 and p.parts[0] == root_name:
            candidate_rel_paths.append(Path(*p.parts[1:]))

        for rel in candidate_rel_paths:
            full_path = (self.image_root / rel).resolve()
            try:
                full_path.relative_to(self.image_root)
            except ValueError:
                continue
            if full_path.exists():
                return full_path
        return None

    def __getitem__(self, index):
        data_id = index % len(self.image_paths)
        image = Image.open(self.image_paths[data_id]).convert("RGB")
        image = self.image_processor(image)
        text = self.prompts[data_id]

        if torch.any(image < -1) or torch.any(image > 1):
            logger.warning("Image values are outside the expected range [-1, 1].")

        return {"text": text, "image": image, "z": torch.empty(0)}

    def __len__(self):
        return len(self.image_paths) * self.datasets_repeat
