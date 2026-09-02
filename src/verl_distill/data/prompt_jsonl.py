import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class TextPromptJsonlDataset(torch.utils.data.Dataset):
    """Dataset for prompt-only JSONL rows used by data-free training.

    Expected row schema:
    - prompt field under `prompt_key`
    """

    def __init__(
        self,
        jsonl_path,
        prompt_key="refined_prompt",
        prompt_keys=None,
        prompt_key_probs=None,
        datasets_repeat=1,
    ):
        self.datasets_repeat = int(datasets_repeat)
        self.jsonl_path = Path(jsonl_path).resolve()
        self.prompt_key = str(prompt_key)
        self.prompt_keys = self._build_prompt_keys(prompt_key, prompt_keys)
        self.prompt_key_probs = self._build_prompt_key_probs(prompt_key_probs)

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")

        self.prompt_candidates = []
        num_invalid_json = 0
        num_missing = 0
        num_invalid_prompt = 0

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

                prompts = self._extract_prompts(row)
                if all(prompt is None for prompt in prompts):
                    num_invalid_prompt += 1
                    continue

                self.prompt_candidates.append(prompts)

        if len(self.prompt_candidates) == 0:
            raise ValueError(
                f"No valid prompts found in jsonl={self.jsonl_path} using keys={self.prompt_keys}"
            )

        logger.info(
            "Loaded %d prompt rows from jsonl=%s using keys=%s probs=%s. skipped: invalid_json=%d missing=%d invalid_prompt=%d",
            len(self.prompt_candidates),
            self.jsonl_path,
            self.prompt_keys,
            self.prompt_key_probs.tolist(),
            num_invalid_json,
            num_missing,
            num_invalid_prompt,
        )

    def _build_prompt_keys(self, prompt_key, prompt_keys):
        if prompt_keys is None:
            return [str(prompt_key)]
        keys = [str(key) for key in prompt_keys if str(key).strip()]
        if len(keys) == 0:
            raise ValueError("prompt_keys must contain at least one non-empty key")
        return keys

    def _build_prompt_key_probs(self, prompt_key_probs):
        if prompt_key_probs is None:
            probs = torch.ones(len(self.prompt_keys), dtype=torch.float32)
        else:
            probs = torch.tensor(list(prompt_key_probs), dtype=torch.float32)
        if probs.numel() != len(self.prompt_keys):
            raise ValueError("prompt_key_probs length must match prompt_keys length")
        if torch.any(probs < 0):
            raise ValueError("prompt_key_probs must be non-negative")
        if float(probs.sum().item()) <= 0:
            raise ValueError("prompt_key_probs must sum to a positive value")
        return probs / probs.sum()

    def _extract_prompts(self, row):
        prompts = []
        for key in self.prompt_keys:
            prompt = row.get(key, None)
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt.strip())
            else:
                prompts.append(None)
        return prompts

    def _sample_prompt(self, prompts):
        valid_indices = [idx for idx, prompt in enumerate(prompts) if prompt is not None]
        if len(valid_indices) == 0:
            raise ValueError("At least one prompt candidate must be available")
        if len(valid_indices) == 1:
            return prompts[valid_indices[0]]
        probs = self.prompt_key_probs[valid_indices]
        probs = probs / probs.sum()
        chosen = int(torch.multinomial(probs, num_samples=1).item())
        return prompts[valid_indices[chosen]]

    def __getitem__(self, index):
        data_id = index % len(self.prompt_candidates)
        prompt = self._sample_prompt(self.prompt_candidates[data_id])
        return {"text": prompt, "image": torch.empty(0), "z": torch.empty(0)}

    def __len__(self):
        return len(self.prompt_candidates) * self.datasets_repeat
