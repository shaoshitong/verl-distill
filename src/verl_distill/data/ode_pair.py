from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import torch

DEFAULT_ODE_REWARD_WEIGHTS = {
    "lance": {
        "aesthetic": 0.45,
        "color_harmony": 0.40,
        "instruction_following": 0.10,
        "score": 0.05,
    },
    "text_render": {
        "instruction_following": 0.75,
        "aesthetic": 0.10,
        "color_harmony": 0.10,
        "score": 0.05,
    },
    "default": {
        "aesthetic": 0.35,
        "color_harmony": 0.25,
        "instruction_following": 0.35,
        "score": 0.05,
    },
}


def _load_tensor(path: str) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(value).__name__}")
    return value.to(torch.float32)


class OdePairDataset(torch.utils.data.Dataset):
    """Dataset for ODE warmup pairs: clean latent, initial noise, prompt, reward weight."""

    def __init__(
        self,
        pair_dir: str,
        records_name: str = "records.json",
        reward_weighting: str = "source_rank",
        reward_weights: Mapping | None = None,
        rank_weight_strength: float = 1.0,
    ):
        self.pair_dir = Path(pair_dir)
        if not self.pair_dir.exists():
            raise FileNotFoundError(f"ODE pair directory does not exist: {pair_dir}")
        self.reward_weighting = str(reward_weighting).lower()
        if self.reward_weighting not in {"none", "source_rank"}:
            raise ValueError("reward_weighting must be one of: none, source_rank")
        self.rank_weight_strength = float(rank_weight_strength)
        if not math.isfinite(self.rank_weight_strength) or self.rank_weight_strength < 0.0:
            raise ValueError("rank_weight_strength must be finite and non-negative")
        self.reward_weights = self._normalize_reward_weights(
            reward_weights or DEFAULT_ODE_REWARD_WEIGHTS
        )
        self.records = self._load_records(records_name)
        if not self.records:
            raise ValueError(f"No ODE pair records found in {pair_dir}")
        self._attach_reward_weights()

    def _load_records(self, records_name: str) -> list[dict]:
        records_json = self.pair_dir / records_name
        rows = []
        if records_json.exists():
            data = json.loads(records_json.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(f"{records_json} must contain a JSON list")
            rows.extend(row for row in data if isinstance(row, dict))
        else:
            for path in sorted(self.pair_dir.glob("records_rank*.jsonl")):
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            row = json.loads(line)
                            if isinstance(row, dict):
                                rows.append(row)
        return [
            row
            for row in rows
            if row.get("noise_path") and row.get("latent_path") and row.get("text_prompt")
        ]

    def _normalize_reward_weights(self, reward_weights: Mapping) -> dict[str, dict[str, float]]:
        normalized = {}
        for source, weights in reward_weights.items():
            if not isinstance(weights, Mapping):
                continue
            values = {str(k): float(v) for k, v in weights.items()}
            total = sum(v for v in values.values() if math.isfinite(v) and v > 0.0)
            if total <= 0.0:
                continue
            normalized[str(source)] = {k: max(0.0, v) / total for k, v in values.items()}
        if "default" not in normalized:
            normalized["default"] = DEFAULT_ODE_REWARD_WEIGHTS["default"]
        return normalized

    def _reward_components(self, row: Mapping) -> dict[str, float]:
        raw = (row.get("reward") or {}).get("raw") or {}
        dims = raw.get("normalized_dimension_scores") or {}
        components = {
            "score": row.get("score", (row.get("reward") or {}).get("score", 0.0)),
            "aesthetic": dims.get("aesthetic", 0.0),
            "instruction_following": dims.get("instruction_following", 0.0),
            "color_harmony": dims.get("color_harmony", 0.0),
        }
        out = {}
        for key, value in components.items():
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                value_f = 0.0
            out[key] = value_f if math.isfinite(value_f) else 0.0
        return out

    def _composite_reward(self, row: Mapping) -> float:
        source = str(row.get("source", "default"))
        weights = self.reward_weights.get(source, self.reward_weights["default"])
        components = self._reward_components(row)
        return sum(components.get(key, 0.0) * value for key, value in weights.items())

    def _amplify_rank_weights(self, weights: Iterable[float]) -> list[float]:
        values = [
            max(0.0, 1.0 + self.rank_weight_strength * (float(weight) - 1.0)) for weight in weights
        ]
        if not values:
            return []
        mean_value = sum(values) / float(len(values))
        if mean_value <= 0.0:
            return [1.0 for _ in values]
        return [value / mean_value for value in values]

    def _attach_reward_weights(self) -> None:
        rewards = [self._composite_reward(row) for row in self.records]
        if self.reward_weighting == "none":
            for row, reward in zip(self.records, rewards, strict=True):
                row["_ode_reward"] = float(reward)
                row["_ode_weight"] = 1.0
            return

        by_source = defaultdict(list)
        for idx, row in enumerate(self.records):
            by_source[str(row.get("source", "default"))].append((idx, rewards[idx]))

        weights = [1.0 for _ in self.records]
        for items in by_source.values():
            if len(items) == 1:
                weights[items[0][0]] = 1.0
                continue
            group_weights = [1.0 for _ in items]
            group_positions = {row_idx: group_idx for group_idx, (row_idx, _) in enumerate(items)}
            sorted_items = sorted(items, key=lambda item: item[1])
            denom = float(len(sorted_items) - 1)
            pos = 0
            while pos < len(sorted_items):
                end = pos
                while (
                    end + 1 < len(sorted_items) and sorted_items[end + 1][1] == sorted_items[pos][1]
                ):
                    end += 1
                avg_rank = (pos + end) / 2.0
                weight = 2.0 * (avg_rank / denom)
                for j in range(pos, end + 1):
                    group_idx = group_positions[sorted_items[j][0]]
                    group_weights[group_idx] = weight
                pos = end + 1
            group_weights = self._amplify_rank_weights(group_weights)
            for (row_idx, _), weight in zip(items, group_weights, strict=True):
                weights[row_idx] = weight

        for row, reward, weight in zip(self.records, rewards, weights, strict=True):
            row["_ode_reward"] = float(reward)
            row["_ode_weight"] = float(max(0.0, weight))
            for key, value in self._reward_components(row).items():
                row[f"_ode_reward_{key}"] = float(value)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        clean = _load_tensor(row["latent_path"])
        noise = _load_tensor(row["noise_path"])
        if clean.ndim == 4 and clean.shape[0] == 1:
            clean = clean.squeeze(0)
        if noise.ndim == 4 and noise.shape[0] == 1:
            noise = noise.squeeze(0)
        return {
            "text": row["text_prompt"],
            "clean_latent": clean,
            "noise_latent": noise,
            "latent_path": row["latent_path"],
            "noise_path": row["noise_path"],
            "source": row.get("source", "ode_pair"),
            "sample_id": row.get("sample_id", str(idx)),
            "ode_reward": torch.tensor(row.get("_ode_reward", 0.0), dtype=torch.float32),
            "ode_weight": torch.tensor(row.get("_ode_weight", 1.0), dtype=torch.float32),
            "ode_reward_score": torch.tensor(
                row.get("_ode_reward_score", 0.0), dtype=torch.float32
            ),
            "ode_reward_aesthetic": torch.tensor(
                row.get("_ode_reward_aesthetic", 0.0), dtype=torch.float32
            ),
            "ode_reward_instruction_following": torch.tensor(
                row.get("_ode_reward_instruction_following", 0.0), dtype=torch.float32
            ),
            "ode_reward_color_harmony": torch.tensor(
                row.get("_ode_reward_color_harmony", 0.0), dtype=torch.float32
            ),
        }
