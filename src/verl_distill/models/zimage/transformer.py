import copy
import importlib.util
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import ZImageTransformer2DModel
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_z_image import (
    ADALN_EMBED_DIM,
    select_per_token,
)
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

try:
    from .discriminator import (
        ZImageDualProjectorMultiFeatureDiscriminatorHead,
        ZImageMultiFeatureDiscriminatorHead,
        ZImageSharedDualProjectorMultiFeatureDiscriminatorHead,
    )
except ImportError:
    _heads_path = Path(__file__).with_name("discriminator.py")
    _heads_spec = importlib.util.spec_from_file_location(
        "zimage_discriminator_heads_direct",
        _heads_path,
    )
    if _heads_spec is None or _heads_spec.loader is None:
        raise
    _heads_module = importlib.util.module_from_spec(_heads_spec)
    _heads_spec.loader.exec_module(_heads_module)
    ZImageMultiFeatureDiscriminatorHead = _heads_module.ZImageMultiFeatureDiscriminatorHead
    ZImageDualProjectorMultiFeatureDiscriminatorHead = (
        _heads_module.ZImageDualProjectorMultiFeatureDiscriminatorHead
    )
    ZImageSharedDualProjectorMultiFeatureDiscriminatorHead = (
        _heads_module.ZImageSharedDualProjectorMultiFeatureDiscriminatorHead
    )


def _reset_linear_deterministic_(linear: nn.Linear, scale: float) -> None:
    """Reset a linear layer without rank-local RNG."""
    with torch.no_grad():
        rows = torch.arange(
            linear.weight.shape[0],
            device=linear.weight.device,
            dtype=torch.float32,
        ).view(-1, 1)
        cols = torch.arange(
            linear.weight.shape[1],
            device=linear.weight.device,
            dtype=torch.float32,
        ).view(1, -1)
        pattern = (((rows + 1.0) * (cols + 1.0)).remainder(23.0) - 11.0) / 11.0
        linear.weight.copy_(pattern.to(dtype=linear.weight.dtype) * float(scale))
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)


class ResidualAngleBlock(nn.Module):
    """Residual MLP block used by TimeRotaryModulator."""

    def __init__(self, hidden: int, expansion: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        expanded = hidden * expansion
        self.fc1 = nn.Linear(hidden, expanded, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(expanded, hidden, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        _reset_linear_deterministic_(self.fc1, scale=float(self.fc1.weight.shape[1]) ** -0.5)
        _reset_linear_deterministic_(self.fc2, scale=float(self.fc2.weight.shape[1]) ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class TimeRotaryModulator(nn.Module):
    """Learned rotation in frequency space for conditional time embedding.

    Rotates each (cos, sin) frequency pair by a learned angle Δθₖ = MLP(tt).
    Zero-initialized so that at initialization, any tt produces no change —
    the output is identical to the unmodulated frequency embedding.

    Conceptually:
        frequency_embed(t) = [cos(ωₖt), sin(ωₖt)] for k = 0..N
        with tt:  → [cos(ωₖt + Δθₖ), sin(ωₖt + Δθₖ)]
        where Δθₖ = MLP_zero_init(tt_internal)

    This is geometrically a 2D rotation per frequency, like RoPE applied
    in time-frequency space, rather than an ad-hoc additive correction.
    """

    def __init__(
        self,
        freq_dim: int = 256,
        hidden: int = 512,
        depth: int = 4,
        expansion: int = 2,
    ):
        super().__init__()
        self.num_freqs = freq_dim // 2
        self.input_proj = nn.Linear(1, hidden, bias=True)
        self.input_act = nn.SiLU()
        self.blocks = nn.ModuleList(
            [ResidualAngleBlock(hidden, expansion=expansion) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.output_proj = nn.Linear(hidden, self.num_freqs, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        hidden = self.input_proj.weight.shape[0]
        with torch.no_grad():
            values = torch.linspace(
                -1.0,
                1.0,
                hidden,
                device=self.input_proj.weight.device,
                dtype=self.input_proj.weight.dtype,
            ).view(hidden, 1)
            self.input_proj.weight.copy_(values / float(max(1, hidden)))
            nn.init.zeros_(self.input_proj.bias)
        for block in self.blocks:
            block.reset_parameters()
        nn.init.ones_(self.output_norm.weight)
        nn.init.zeros_(self.output_norm.bias)
        # Zero init for the output projection: initial Δθ = 0, so the
        # rotation is identity while the hidden stack remains trainable.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @torch.no_grad()
    def angle_stats(self):
        """Return (mean_abs_angle, max_angle) from the last forward call, or (0, 0) if none."""
        if not hasattr(self, "_last_angles") or self._last_angles is None:
            return 0.0, 0.0
        a = self._last_angles
        return float(a.abs().mean().item()), float(a.abs().max().item())

    def forward(self, t_freq: torch.Tensor, tt_internal: torch.Tensor) -> torch.Tensor:
        """Apply per-frequency rotation.

        Args:
            t_freq: [B, freq_dim] — cos/sin pairs from timestep_embedding.
            tt_internal: [B] — normalized internal target time. For positive
                sigmas passed through GenTransformer, tt_internal equals
                1 - generator_sigma.

        Returns:
            [B, freq_dim] — rotated frequency embedding.
        """
        first_weight = self.input_proj.weight
        tt_internal = tt_internal.to(device=first_weight.device, dtype=first_weight.dtype)
        h = self.input_act(self.input_proj(tt_internal.unsqueeze(-1)))
        for block in self.blocks:
            h = block(h)
        h = self.output_norm(h)
        angles = self.output_proj(h)  # [B, num_freqs]
        self._last_angles = angles.detach()
        cos_a, sin_a = angles.cos(), angles.sin()

        # The frequency embedding stores cos halves then sin halves.
        cos_half = t_freq[:, : self.num_freqs]
        sin_half = t_freq[:, self.num_freqs :]

        # 2D rotation per frequency: (cos, sin) → (cos(θ+Δ), sin(θ+Δ))
        cos_new = cos_half * cos_a - sin_half * sin_a
        sin_new = cos_half * sin_a + sin_half * cos_a

        return torch.cat([cos_new, sin_new], dim=-1)


def _zero_module_(module: nn.Module) -> None:
    for param in module.parameters():
        nn.init.zeros_(param)


def _block_forward_with_r_modulation(
    self,
    x: torch.Tensor,
    attn_mask: torch.Tensor,
    freqs_cis: torch.Tensor,
    adaln_input: torch.Tensor | None = None,
    noise_mask: torch.Tensor | None = None,
    adaln_noisy: torch.Tensor | None = None,
    adaln_clean: torch.Tensor | None = None,
    r_adaln_input: torch.Tensor | None = None,
):
    if self.modulation:
        seq_len = x.shape[1]

        if noise_mask is not None:
            mod_noisy = self.adaLN_modulation(adaln_noisy)
            mod_clean = self.adaLN_modulation(adaln_clean)
            if r_adaln_input is not None:
                r_mod = self.r_adaLN_modulation(r_adaln_input)
                mod_noisy = mod_noisy + r_mod
                mod_clean = mod_clean + r_mod

            scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(
                4, dim=1
            )
            scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(
                4, dim=1
            )

            gate_msa_noisy, gate_mlp_noisy = (
                gate_msa_noisy.tanh(),
                gate_mlp_noisy.tanh(),
            )
            gate_msa_clean, gate_mlp_clean = (
                gate_msa_clean.tanh(),
                gate_mlp_clean.tanh(),
            )

            scale_msa_noisy, scale_mlp_noisy = (
                1.0 + scale_msa_noisy,
                1.0 + scale_mlp_noisy,
            )
            scale_msa_clean, scale_mlp_clean = (
                1.0 + scale_msa_clean,
                1.0 + scale_mlp_clean,
            )

            scale_msa = select_per_token(
                scale_msa_noisy,
                scale_msa_clean,
                noise_mask,
                seq_len,
            )
            scale_mlp = select_per_token(
                scale_mlp_noisy,
                scale_mlp_clean,
                noise_mask,
                seq_len,
            )
            gate_msa = select_per_token(
                gate_msa_noisy,
                gate_msa_clean,
                noise_mask,
                seq_len,
            )
            gate_mlp = select_per_token(
                gate_mlp_noisy,
                gate_mlp_clean,
                noise_mask,
                seq_len,
            )
        else:
            mod = self.adaLN_modulation(adaln_input)
            if r_adaln_input is not None:
                mod = mod + self.r_adaLN_modulation(r_adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa,
            attention_mask=attn_mask,
            freqs_cis=freqs_cis,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
    else:
        attn_out = self.attention(
            self.attention_norm1(x),
            attention_mask=attn_mask,
            freqs_cis=freqs_cis,
        )
        x = x + self.attention_norm2(attn_out)
        x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

    return x


def _final_layer_forward_with_r_modulation(
    self,
    x,
    c=None,
    noise_mask=None,
    c_noisy=None,
    c_clean=None,
    r_c=None,
):
    seq_len = x.shape[1]

    if noise_mask is not None:
        scale_noisy = 1.0 + self.adaLN_modulation(c_noisy)
        scale_clean = 1.0 + self.adaLN_modulation(c_clean)
        if r_c is not None:
            r_scale = self.r_adaLN_modulation(r_c)
            scale_noisy = scale_noisy + r_scale
            scale_clean = scale_clean + r_scale
        scale = select_per_token(scale_noisy, scale_clean, noise_mask, seq_len)
    else:
        assert c is not None, "Either c or (c_noisy, c_clean) must be provided"
        scale = 1.0 + self.adaLN_modulation(c)
        if r_c is not None:
            scale = scale + self.r_adaLN_modulation(r_c)
        scale = scale.unsqueeze(1)

    x = self.norm_final(x) * scale
    x = self.linear(x)
    return x


def _attach_r_modulation_to_block(block: nn.Module) -> bool:
    if not getattr(block, "modulation", False):
        return False
    if not hasattr(block, "adaLN_modulation"):
        return False
    if not hasattr(block, "r_adaLN_modulation"):
        dim = int(block.dim)
        in_features = min(dim, ADALN_EMBED_DIM)
        block.r_adaLN_modulation = nn.Sequential(nn.Linear(in_features, 4 * dim, bias=True))
        _zero_module_(block.r_adaLN_modulation)
        ref_param = next(block.adaLN_modulation.parameters())
        block.r_adaLN_modulation.to(
            device=ref_param.device,
            dtype=ref_param.dtype,
        )
    if not getattr(block, "uses_separate_r_modulation", False):
        block._forward_without_r_modulation = block.forward
        block.forward = types.MethodType(_block_forward_with_r_modulation, block)
        block.uses_separate_r_modulation = True
    return True


def _attach_r_modulation_to_final_layer(final_layer: nn.Module) -> bool:
    if not hasattr(final_layer, "adaLN_modulation"):
        return False
    if not hasattr(final_layer, "r_adaLN_modulation"):
        linear = final_layer.linear
        hidden_size = int(linear.in_features)
        final_layer.r_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )
        _zero_module_(final_layer.r_adaLN_modulation)
        ref_param = next(final_layer.adaLN_modulation.parameters())
        final_layer.r_adaLN_modulation.to(
            device=ref_param.device,
            dtype=ref_param.dtype,
        )
    if not getattr(final_layer, "uses_separate_r_modulation", False):
        final_layer._forward_without_r_modulation = final_layer.forward
        final_layer.forward = types.MethodType(
            _final_layer_forward_with_r_modulation,
            final_layer,
        )
        final_layer.uses_separate_r_modulation = True
    return True


class ZImageDiscriminatorScalarHead(nn.Module):
    """Pool a Z-Image velocity-like tensor into one realism logit per sample."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(int(in_channels), int(in_channels), kernel_size=1)
        self.norm = nn.LayerNorm(int(in_channels))
        self.act = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(int(in_channels), 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv(x)
        h = h.permute(0, 2, 3, 4, 1)
        h = self.norm(h)
        h = h.permute(0, 4, 1, 2, 3)
        h = residual + self.act(h)
        logits = self.linear(self.flatten(self.pool(h)))
        return logits.flatten()


class ZImageTransformer2DModelWrapper(ZImageTransformer2DModel):
    @register_to_config
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        siglip_feat_dim=None,  # Optional: set to enable SigLIP support for Omni
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
    ) -> None:
        super().__init__(
            all_patch_size,
            all_f_patch_size,
            in_channels,
            dim,
            n_layers,
            n_refiner_layers,
            n_heads,
            n_kv_heads,
            norm_eps,
            qk_norm,
            cap_feat_dim,
            siglip_feat_dim,
            rope_theta,
            t_scale,
            axes_dims,
            axes_lens,
        )

        # Rotary modulator for conditional time embedding (tt).
        # Replaces the old additive t_embedder_2 with a zero-init rotation
        # that starts as identity and learns through training.
        self.time_rotary = TimeRotaryModulator(
            freq_dim=self.t_embedder.frequency_embedding_size,
            hidden=256,
            depth=3,
            expansion=2,
        )
        self.t_embedder.float()
        self.time_rotary.float()
        self.separate_r_modulation = False
        self.separate_r_disable_time_rotary = False

    def enable_separate_r_modulation(
        self,
        *,
        disable_time_rotary: bool = True,
        include_final_layer: bool = True,
    ) -> int:
        attached = 0
        for block_group in (self.noise_refiner, self.layers):
            for block in block_group:
                attached += int(_attach_r_modulation_to_block(block))
        if include_final_layer:
            for final_layer in self.all_final_layer.values():
                attached += int(_attach_r_modulation_to_final_layer(final_layer))
        self.separate_r_modulation = True
        self.separate_r_disable_time_rotary = bool(disable_time_rotary)
        return attached

    def named_r_modulation_parameters(self):
        for name, param in self.named_parameters():
            if ".r_adaLN_modulation." in name:
                yield name, param

    def r_modulation_parameters(self):
        for _, param in self.named_r_modulation_parameters():
            yield param

    def init_discriminator_head(self, patch_size: int = 2, f_patch_size: int = 1):
        """Initialize the trainable discriminator projector from the original head."""
        key = f"{patch_size}-{f_patch_size}"
        if key not in self.all_final_layer:
            raise ValueError(f"Unknown Z-Image patch head key: {key}")
        self.discriminator_final_layer = copy.deepcopy(self.all_final_layer[key])
        self.discriminator_scalar_head = ZImageDiscriminatorScalarHead(int(self.config.in_channels))
        ref_param = next(self.all_final_layer[key].parameters())
        self.discriminator_scalar_head.to(
            device=ref_param.device,
            dtype=ref_param.dtype,
        )

    def init_multi_feature_discriminator_head(
        self,
        layer_numbers=(8, 16, 24),
        fusion: str = "channel",
        norm: str = "old",
        transformer_layers: int = 1,
        transformer_heads: int = 8,
        mlp_ratio: float = 4.0,
        projector_residual_mode: str = "none",
        projector_residual_init: float = 0.1,
        text_conditioning: str = "none",
        output_dim: int = 1,
        output_mode: str = "pooled",
        use_time_embedding: bool = True,
        pre_projector_only: bool = False,
        ocr_score_head: bool = False,
        ocr_feature_adapter: bool = True,
        ocr_feature_branch_layer: int = 0,
        ocr_feature_branch_output_dim: int = 0,
        ocr_text_branch_output_dim: int = 0,
        ocr_text_branch_max_tokens: int = 0,
    ):
        """Initialize a discriminator head over frozen teacher intermediate features."""
        layer_numbers = tuple(int(value) for value in layer_numbers)
        if len(layer_numbers) != 3:
            raise ValueError(
                "discriminator_multifeature_layers must contain exactly three "
                f"main-layer ids, got {layer_numbers}"
            )
        invalid = [value for value in layer_numbers if value < 1 or value > len(self.layers)]
        if invalid:
            raise ValueError(
                "discriminator_multifeature_layers must be in "
                f"[1, {len(self.layers)}], got {invalid}"
            )
        if len(set(layer_numbers)) != len(layer_numbers):
            raise ValueError("discriminator_multifeature_layers must not contain duplicates")
        self.multi_feature_discriminator_layer_numbers = layer_numbers
        self.multi_feature_discriminator_head = ZImageMultiFeatureDiscriminatorHead(
            hidden_dim=int(self.config.dim),
            num_features=len(layer_numbers) + 1,
            fusion=fusion,
            norm=norm,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            mlp_ratio=mlp_ratio,
            projector_residual_mode=projector_residual_mode,
            projector_residual_init=projector_residual_init,
            text_conditioning=text_conditioning,
            output_dim=output_dim,
            output_mode=output_mode,
            use_time_embedding=use_time_embedding,
            pre_projector_only=pre_projector_only,
            ocr_score_head=ocr_score_head,
            ocr_feature_adapter=ocr_feature_adapter,
            ocr_feature_branch_layer=ocr_feature_branch_layer,
            ocr_feature_branch_output_dim=ocr_feature_branch_output_dim,
            ocr_text_branch_output_dim=ocr_text_branch_output_dim,
            ocr_text_branch_max_tokens=ocr_text_branch_max_tokens,
        )
        ref_param = next(self.parameters())
        self.multi_feature_discriminator_head.to(
            device=ref_param.device,
            dtype=ref_param.dtype,
        )

    def init_dual_projector_multi_feature_discriminator_head(
        self,
        layer_numbers=(8, 16, 24),
        fusion: str = "channel",
        norm: str = "old",
        transformer_layers: int = 1,
        transformer_heads: int = 8,
        mlp_ratio: float = 4.0,
        projector_residual_mode: str = "none",
        projector_residual_init: float = 0.1,
        text_conditioning: str = "none",
        align_output_dim: int = 4096,
        use_time_embedding: bool = True,
    ):
        """Initialize a dual projector head over frozen teacher intermediate features."""
        layer_numbers = tuple(int(value) for value in layer_numbers)
        if len(layer_numbers) != 3:
            raise ValueError(
                "discriminator_multifeature_layers must contain exactly three "
                f"main-layer ids, got {layer_numbers}"
            )
        invalid = [value for value in layer_numbers if value < 1 or value > len(self.layers)]
        if invalid:
            raise ValueError(
                "discriminator_multifeature_layers must be in "
                f"[1, {len(self.layers)}], got {invalid}"
            )
        if len(set(layer_numbers)) != len(layer_numbers):
            raise ValueError("discriminator_multifeature_layers must not contain duplicates")
        self.multi_feature_discriminator_layer_numbers = layer_numbers
        self.dual_projector_multi_feature_discriminator_head = (
            ZImageDualProjectorMultiFeatureDiscriminatorHead(
                hidden_dim=int(self.config.dim),
                num_features=len(layer_numbers) + 1,
                fusion=fusion,
                norm=norm,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                mlp_ratio=mlp_ratio,
                projector_residual_mode=projector_residual_mode,
                projector_residual_init=projector_residual_init,
                text_conditioning=text_conditioning,
                align_output_dim=align_output_dim,
                use_time_embedding=use_time_embedding,
            )
        )
        ref_param = next(self.parameters())
        self.dual_projector_multi_feature_discriminator_head.to(
            device=ref_param.device,
            dtype=ref_param.dtype,
        )

    def discriminator_parameters(self):
        if not hasattr(self, "discriminator_final_layer") or not hasattr(
            self, "discriminator_scalar_head"
        ):
            raise ValueError("Discriminator head is not initialized")
        yield from self.time_rotary.parameters()
        yield from self.discriminator_final_layer.parameters()
        yield from self.discriminator_scalar_head.parameters()

    def multi_feature_discriminator_parameters(self):
        if not hasattr(self, "multi_feature_discriminator_head"):
            raise ValueError("Multi-feature discriminator head is not initialized")
        yield from self.multi_feature_discriminator_head.parameters()

    def dual_projector_multi_feature_discriminator_parameters(self):
        head = getattr(self, "dual_projector_multi_feature_discriminator_head", None)
        if head is None:
            head = getattr(self, "multi_feature_discriminator_head", None)
        if head is None:
            raise ValueError("Dual multi-feature discriminator head is not initialized")
        yield from head.parameters()
        if getattr(head, "is_shared_exit_discriminator", False):
            return
        exit2_head = getattr(
            self,
            "dual_projector_multi_feature_discriminator_head_exit2",
            None,
        )
        if exit2_head is not None:
            yield from exit2_head.parameters()

    def _select_dual_discriminator_head(self, discriminator_head: Optional[str]):
        primary_head = getattr(
            self,
            "dual_projector_multi_feature_discriminator_head",
            None,
        )
        if getattr(primary_head, "is_shared_exit_discriminator", False):
            return primary_head.select_exit_module(discriminator_head or "exit1")
        head_name = str(discriminator_head or "exit1").lower()
        if head_name == "exit1":
            head = primary_head
        elif head_name == "exit2":
            head = getattr(
                self,
                "dual_projector_multi_feature_discriminator_head_exit2",
                None,
            )
            if head is None:
                raise ValueError("exit-2 discriminator head is not initialized")
        else:
            raise ValueError(
                f"discriminator_head must be 'exit1' or 'exit2', got {discriminator_head!r}"
            )
        if head is None:
            raise ValueError("Dual multi-feature discriminator head is not initialized")
        return head

    def _prepare_sequence(
        self,
        feats: list[torch.Tensor],
        pos_ids: list[torch.Tensor],
        inner_pad_mask: list[torch.Tensor],
        pad_token: torch.nn.Parameter,
        noise_mask: list[list[int]] | None = None,
        device: torch.device = None,
    ):
        if feats:
            pad_token = pad_token.to(device=feats[0].device, dtype=feats[0].dtype)
        return super()._prepare_sequence(
            feats,
            pos_ids,
            inner_pad_mask,
            pad_token,
            noise_mask,
            device,
        )

    def forward(
        self,
        x: Union[List[torch.Tensor], List[List[torch.Tensor]]],
        t,
        cap_feats: Union[List[torch.Tensor], List[List[torch.Tensor]]],
        return_dict: bool = True,
        controlnet_block_samples: Optional[Dict[int, torch.Tensor]] = None,
        siglip_feats: Optional[List[List[torch.Tensor]]] = None,
        image_noise_mask: Optional[List[List[int]]] = None,
        patch_size: int = 2,
        f_patch_size: int = 1,
        target_timestep: Optional[torch.Tensor] = None,
        discriminator_mode: bool = False,
        return_discriminator_features: bool = False,
        discriminator_output: Optional[str] = None,
        discriminator_head: Optional[str] = None,
        disable_separate_r_modulation: bool = False,
        feature_layers: Optional[Tuple[int, ...]] = None,
    ):
        """
        Flow: patchify -> t_embed -> x_embed -> x_refine -> cap_embed -> cap_refine
              -> [siglip_embed -> siglip_refine] -> build_unified -> main_layers -> final_layer -> unpatchify
        """
        assert patch_size in self.all_patch_size and f_patch_size in self.all_f_patch_size
        omni_mode = isinstance(x[0], list)
        device = x[0][-1].device if omni_mode else x[0].device

        calc_dtype = torch.float64
        use_separate_r = bool(
            getattr(self, "separate_r_modulation", False)
            and target_timestep is not None
            and not bool(disable_separate_r_modulation)
        )
        r_adaln_input = None
        if omni_mode:
            # Dual embeddings: noisy (t) and clean (t=1)
            t_high = t.to(dtype=calc_dtype)
            with torch.autocast(device_type=device.type, enabled=False):
                t_noisy_freq = self.t_embedder.timestep_embedding(
                    t_high * self.t_scale,
                    self.t_embedder.frequency_embedding_size,
                )
                if target_timestep is not None:
                    target_t_high = target_timestep.to(dtype=calc_dtype)
                    if not (
                        use_separate_r and getattr(self, "separate_r_disable_time_rotary", False)
                    ):
                        t_noisy_freq = self.time_rotary(t_noisy_freq, target_t_high)
                    if use_separate_r:
                        r_freq = self.t_embedder.timestep_embedding(
                            target_t_high * self.t_scale,
                            self.t_embedder.frequency_embedding_size,
                        )
                        r_adaln_input = self.t_embedder.mlp(
                            r_freq.to(self.t_embedder.mlp[0].weight.dtype)
                        )
                t_noisy = self.t_embedder.mlp(t_noisy_freq.to(self.t_embedder.mlp[0].weight.dtype))
                t_clean = self.t_embedder(torch.ones_like(t_high) * self.t_scale)
            t_noisy = t_noisy.type_as(x[0][-1])
            t_clean = t_clean.type_as(x[0][-1])
            if r_adaln_input is not None:
                r_adaln_input = r_adaln_input.type_as(x[0][-1])
            adaln_input = None
        else:
            # Single embedding for all tokens
            t_high = t.to(dtype=calc_dtype)
            with torch.autocast(device_type=device.type, enabled=False):
                t_freq = self.t_embedder.timestep_embedding(
                    t_high * self.t_scale,
                    self.t_embedder.frequency_embedding_size,
                )
                if target_timestep is not None:
                    target_t_high = target_timestep.to(dtype=calc_dtype)
                    if not (
                        use_separate_r and getattr(self, "separate_r_disable_time_rotary", False)
                    ):
                        t_freq = self.time_rotary(t_freq, target_t_high)
                    if use_separate_r:
                        r_freq = self.t_embedder.timestep_embedding(
                            target_t_high * self.t_scale,
                            self.t_embedder.frequency_embedding_size,
                        )
                        r_adaln_input = self.t_embedder.mlp(
                            r_freq.to(self.t_embedder.mlp[0].weight.dtype)
                        )
                adaln_input = self.t_embedder.mlp(t_freq.to(self.t_embedder.mlp[0].weight.dtype))
            adaln_input = adaln_input.type_as(x[0])
            if r_adaln_input is not None:
                r_adaln_input = r_adaln_input.type_as(x[0])
            t_noisy = t_clean = None

        # Patchify
        if omni_mode:
            (
                x,
                cap_feats,
                siglip_feats,
                x_size,
                x_pos_ids,
                cap_pos_ids,
                siglip_pos_ids,
                x_pad_mask,
                cap_pad_mask,
                siglip_pad_mask,
                x_pos_offsets,
                x_noise_mask,
                cap_noise_mask,
                siglip_noise_mask,
            ) = self.patchify_and_embed_omni(
                x, cap_feats, siglip_feats, patch_size, f_patch_size, image_noise_mask
            )
        else:
            (
                x,
                cap_feats,
                x_size,
                x_pos_ids,
                cap_pos_ids,
                x_pad_mask,
                cap_pad_mask,
            ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)
            x_pos_offsets = x_noise_mask = cap_noise_mask = siglip_noise_mask = None

        # X embed & refine
        x_seqlens = [len(xi) for xi in x]
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x, dim=0))  # embed
        x, x_freqs, x_mask, _, x_noise_tensor = self._prepare_sequence(
            list(x.split(x_seqlens, dim=0)),
            x_pos_ids,
            x_pad_mask,
            self.x_pad_token,
            x_noise_mask,
            device,
        )

        for layer in self.noise_refiner:
            layer_args = (
                x,
                x_mask,
                x_freqs,
                adaln_input,
                x_noise_tensor,
                t_noisy,
                t_clean,
            )
            if r_adaln_input is not None and getattr(
                layer,
                "uses_separate_r_modulation",
                False,
            ):
                layer_args = (*layer_args, r_adaln_input)
            x = (
                self._gradient_checkpointing_func(
                    layer,
                    *layer_args,
                )
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(*layer_args)
            )

        # Cap embed & refine
        cap_seqlens = [len(ci) for ci in cap_feats]
        cap_feats = self.cap_embedder(torch.cat(cap_feats, dim=0))  # embed
        cap_feats, cap_freqs, cap_mask, _, _ = self._prepare_sequence(
            list(cap_feats.split(cap_seqlens, dim=0)),
            cap_pos_ids,
            cap_pad_mask,
            self.cap_pad_token,
            None,
            device,
        )

        for layer in self.context_refiner:
            cap_feats = (
                self._gradient_checkpointing_func(layer, cap_feats, cap_mask, cap_freqs)
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(cap_feats, cap_mask, cap_freqs)
            )
        cap_valid_mask = torch.arange(cap_feats.shape[1], device=cap_feats.device).view(
            1, -1
        ) < torch.tensor(cap_seqlens, device=cap_feats.device).view(-1, 1)

        # Siglip embed & refine
        siglip_seqlens = siglip_freqs = None
        if omni_mode and siglip_feats[0] is not None and self.siglip_embedder is not None:
            siglip_seqlens = [len(si) for si in siglip_feats]
            siglip_feats = self.siglip_embedder(torch.cat(siglip_feats, dim=0))  # embed
            siglip_feats, siglip_freqs, siglip_mask, _, _ = self._prepare_sequence(
                list(siglip_feats.split(siglip_seqlens, dim=0)),
                siglip_pos_ids,
                siglip_pad_mask,
                self.siglip_pad_token,
                None,
                device,
            )

            for layer in self.siglip_refiner:
                siglip_feats = (
                    self._gradient_checkpointing_func(
                        layer, siglip_feats, siglip_mask, siglip_freqs
                    )
                    if torch.is_grad_enabled() and self.gradient_checkpointing
                    else layer(siglip_feats, siglip_mask, siglip_freqs)
                )

        # Unified sequence
        unified, unified_freqs, unified_mask, unified_noise_tensor = self._build_unified_sequence(
            x,
            x_freqs,
            x_seqlens,
            x_noise_mask,
            cap_feats,
            cap_freqs,
            cap_seqlens,
            cap_noise_mask,
            siglip_feats,
            siglip_freqs,
            siglip_seqlens,
            siglip_noise_mask,
            omni_mode,
            device,
        )

        # Main transformer layers
        multi_feature_discriminator_head = None
        if discriminator_mode:
            dual_head = getattr(
                self,
                "dual_projector_multi_feature_discriminator_head",
                None,
            )
            frozen_head = getattr(self, "multi_feature_discriminator_head", None)
            if discriminator_output is not None and dual_head is not None:
                multi_feature_discriminator_head = self._select_dual_discriminator_head(
                    discriminator_head
                )
            elif frozen_head is not None:
                multi_feature_discriminator_head = frozen_head
            else:
                multi_feature_discriminator_head = dual_head
        use_multi_feature_discriminator = (
            multi_feature_discriminator_head is not None or feature_layers is not None
        )
        discriminator_features = (
            {} if discriminator_mode and return_discriminator_features else None
        )
        multi_feature_tensors = {} if use_multi_feature_discriminator else None
        multi_feature_layer_map = {}
        multi_feature_token_count = int(x.shape[1]) if use_multi_feature_discriminator else 0
        if use_multi_feature_discriminator:
            layer_numbers = getattr(
                self,
                "multi_feature_discriminator_layer_numbers",
                (),
            )
            if feature_layers is not None:
                layer_numbers = tuple(feature_layers)
                if (
                    not layer_numbers
                    or len(set(layer_numbers)) != len(layer_numbers)
                    or any(level < 1 or level > len(self.layers) for level in layer_numbers)
                ):
                    raise ValueError("feature_layers must select distinct valid teacher layers")
            multi_feature_layer_map = {
                int(layer_number): f"layer_{slot_idx + 1}"
                for slot_idx, layer_number in enumerate(layer_numbers)
            }
        quarter_layer_idx = (
            max(0, len(self.layers) // 4 - 1)
            if discriminator_features is not None and len(self.layers) > 0
            else None
        )
        for layer_idx, layer in enumerate(self.layers):
            layer_args = (
                unified,
                unified_mask,
                unified_freqs,
                adaln_input,
                unified_noise_tensor,
                t_noisy,
                t_clean,
            )
            if r_adaln_input is not None and getattr(
                layer,
                "uses_separate_r_modulation",
                False,
            ):
                layer_args = (*layer_args, r_adaln_input)
            unified = (
                self._gradient_checkpointing_func(
                    layer,
                    *layer_args,
                )
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(*layer_args)
            )
            if controlnet_block_samples is not None and layer_idx in controlnet_block_samples:
                unified = unified + controlnet_block_samples[layer_idx]
            if discriminator_features is not None and layer_idx == quarter_layer_idx:
                discriminator_features["quarter"] = unified.to(torch.float32).mean(dim=1)
            layer_number = layer_idx + 1
            if multi_feature_tensors is not None and layer_number in multi_feature_layer_map:
                multi_feature_tensors[multi_feature_layer_map[layer_number]] = unified[
                    :, :multi_feature_token_count
                ]

        if feature_layers is not None:
            multi_feature_tensors["pre_projector"] = unified[:, :multi_feature_token_count]
            return multi_feature_tensors

        final_layer_key = f"{patch_size}-{f_patch_size}"
        if discriminator_mode and not use_multi_feature_discriminator:
            if not hasattr(self, "discriminator_final_layer") or not hasattr(
                self, "discriminator_scalar_head"
            ):
                raise ValueError("Discriminator head is not initialized")
            final_layer = self.discriminator_final_layer
        else:
            final_layer = self.all_final_layer[final_layer_key]

        if discriminator_features is not None:
            discriminator_features["pre_projector"] = unified.to(torch.float32).mean(dim=1)

        if multi_feature_tensors is not None:
            multi_feature_tensors["pre_projector"] = unified[:, :multi_feature_token_count]
            head_uses_time = bool(
                getattr(multi_feature_discriminator_head, "use_time_embedding", True)
            )
            if target_timestep is None and head_uses_time:
                target_timestep = torch.zeros(
                    unified.shape[0],
                    device=unified.device,
                    dtype=torch.float32,
                )
            head = multi_feature_discriminator_head
            if getattr(head, "supports_dual_output", False):
                logits = head(
                    multi_feature_tensors,
                    target_timestep,
                    text_features=cap_feats,
                    text_mask=cap_valid_mask,
                    output=discriminator_output or "gan",
                )
            else:
                logits = head(
                    multi_feature_tensors,
                    target_timestep,
                    text_features=cap_feats,
                    text_mask=cap_valid_mask,
                )
            if return_discriminator_features:
                return logits, discriminator_features
            if not return_dict:
                return (logits,)
            return Transformer2DModelOutput(sample=logits)

        if omni_mode:
            final_kwargs = {
                "noise_mask": unified_noise_tensor,
                "c_noisy": t_noisy,
                "c_clean": t_clean,
            }
            if r_adaln_input is not None and getattr(
                final_layer,
                "uses_separate_r_modulation",
                False,
            ):
                final_kwargs["r_c"] = r_adaln_input
            unified = final_layer(unified, **final_kwargs)
        else:
            final_kwargs = {"c": adaln_input}
            if r_adaln_input is not None and getattr(
                final_layer,
                "uses_separate_r_modulation",
                False,
            ):
                final_kwargs["r_c"] = r_adaln_input
            unified = final_layer(unified, **final_kwargs)

        # Unpatchify
        x = self.unpatchify(
            list(unified.unbind(dim=0)), x_size, patch_size, f_patch_size, x_pos_offsets
        )

        if discriminator_mode:
            logits = self.discriminator_scalar_head(torch.stack(x, dim=0))
            if return_discriminator_features:
                return logits, discriminator_features
            if not return_dict:
                return (logits,)
            return Transformer2DModelOutput(sample=logits)

        if not return_dict:
            return (x,)

        return Transformer2DModelOutput(sample=x)
