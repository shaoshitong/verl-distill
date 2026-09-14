from __future__ import annotations

from pathlib import Path

import torch.distributed.checkpoint as dcp
from torch import nn

from .discriminator import ZImageMultiFeatureDiscriminatorHead


class TeacherFeatureDiscriminator(nn.Module):
    """OPD multi-level head; the frozen teacher is owned by the score role."""

    uses_teacher_features = True

    def __init__(
        self,
        hidden_dim,
        layer_numbers=(4, 12, 20),
        transformer_layers=5,
        transformer_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.feature_layers = tuple(layer_numbers)
        self.head = ZImageMultiFeatureDiscriminatorHead(
            hidden_dim=hidden_dim,
            num_features=4,
            fusion="channel",
            norm="new",
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            mlp_ratio=mlp_ratio,
            output_dim=1,
            output_mode="tokens",
            use_time_embedding=False,
        )
        self.head.gradient_checkpointing = True

    def forward(self, features, return_features=False):
        if return_features:
            logits, prelogit = self.head(features, return_prelogit_features=True)
            return {"logits": logits, "features": prelogit}
        return self.head(features)

    def load_pretrained(self, path):
        """Load the DINO-distilled trunk, keeping only the new GAN projection fresh."""
        path = Path(path)
        if (path / "fsdp_state").is_dir():
            path = path / "fsdp_state"
        reader = dcp.FileSystemReader(path)
        metadata = reader.read_metadata().state_dict_metadata
        target = self.head.state_dict()
        fresh = {"out_mlp.2.weight", "out_mlp.2.bias"}
        prefix = "teacher_discriminator_model."
        state = {}
        for key, tensor in target.items():
            if key in fresh:
                continue
            source_key = prefix + key
            info = metadata.get(source_key)
            if info is None or tuple(info.size) != tuple(tensor.shape):
                raise ValueError(f"Pretrained discriminator missing or incompatible: {source_key}")
            state[source_key] = tensor
        # Each rank loads its CPU head before FSDP wrapping, without load collectives.
        dcp.load(state, storage_reader=reader, no_dist=True)
        loaded = {key.removeprefix(prefix): value for key, value in state.items()}
        result = self.head.load_state_dict(loaded, strict=False)
        if set(result.missing_keys) != fresh or result.unexpected_keys:
            raise RuntimeError(f"Unexpected discriminator initialization result: {result}")
        return {
            "path": str(path),
            "loaded_tensors": len(loaded),
            "loaded_parameters": sum(t.numel() for t in loaded.values()),
            "fresh_keys": sorted(fresh),
        }
