import copy
import os
from typing import Dict, List, NamedTuple

import torch
import torch.distributed as dist
import torch.nn as nn


class DualProjectorOutput(NamedTuple):
    align: torch.Tensor
    logits: torch.Tensor


class SharedTrunkOutput(NamedTuple):
    tokens: torch.Tensor
    text_film: tuple[torch.Tensor, torch.Tensor] | None


def _normalize_common_head_args(
    *,
    hidden_dim: int,
    num_features: int,
    fusion: str,
    norm: str,
    transformer_layers: int,
    transformer_heads: int,
    mlp_ratio: float,
    projector_residual_mode: str,
    text_conditioning: str,
):
    hidden_dim = int(hidden_dim)
    num_features = int(num_features)
    transformer_layers = int(transformer_layers)
    transformer_heads = int(transformer_heads)
    mlp_ratio = float(mlp_ratio)
    fusion = str(fusion).lower()
    norm = str(norm).lower()
    projector_residual_mode = str(projector_residual_mode).lower()
    text_conditioning = str(text_conditioning).lower()
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be > 0")
    if num_features <= 0:
        raise ValueError("num_features must be > 0")
    if fusion not in {"channel", "sequence"}:
        raise ValueError("discriminator_multifeature_fusion must be 'channel' or 'sequence'")
    if norm not in {"old", "new"}:
        raise ValueError("discriminator_multifeature_norm must be 'old' or 'new'")
    if norm == "new" and fusion != "channel":
        raise ValueError("discriminator_multifeature_norm='new' requires fusion='channel'")
    if transformer_layers < 1:
        raise ValueError("transformer_layers must be >= 1")
    if transformer_heads <= 0 or hidden_dim % transformer_heads != 0:
        raise ValueError("transformer_heads must divide hidden_dim")
    if mlp_ratio <= 0.0:
        raise ValueError("mlp_ratio must be > 0")
    if projector_residual_mode not in {"none", "gated_concat"}:
        raise ValueError("projector_residual_mode must be 'none' or 'gated_concat'")
    if text_conditioning not in {"none", "film"}:
        raise ValueError("text_conditioning must be 'none' or 'film'")
    return (
        hidden_dim,
        num_features,
        fusion,
        norm,
        transformer_layers,
        transformer_heads,
        mlp_ratio,
        projector_residual_mode,
        text_conditioning,
    )


def _init_multi_feature_fusion(
    module: nn.Module,
    *,
    hidden_dim: int,
    num_features: int,
    fusion: str,
    norm: str,
    transformer_layers: int,
    transformer_heads: int,
    mlp_ratio: float,
    projector_residual_mode: str,
    projector_residual_init: float,
    text_conditioning: str,
    use_time_embedding: bool,
) -> None:
    (
        hidden_dim,
        num_features,
        fusion,
        norm,
        transformer_layers,
        transformer_heads,
        mlp_ratio,
        projector_residual_mode,
        text_conditioning,
    ) = _normalize_common_head_args(
        hidden_dim=hidden_dim,
        num_features=num_features,
        fusion=fusion,
        norm=norm,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        mlp_ratio=mlp_ratio,
        projector_residual_mode=projector_residual_mode,
        text_conditioning=text_conditioning,
    )

    module.hidden_dim = hidden_dim
    module.num_features = num_features
    module.fusion = fusion
    module.norm = norm
    module.projector_residual_mode = projector_residual_mode
    module.text_conditioning = text_conditioning
    module.use_time_embedding = bool(use_time_embedding)
    module.feature_order = tuple(
        [f"layer_{idx}" for idx in range(1, num_features)] + ["pre_projector"]
    )
    module.feature_level_embedding = nn.Parameter(torch.zeros(num_features, hidden_dim))
    if norm == "new":
        module.feature_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_features)]
        )
    else:
        module.feature_norms = None
    module.tt_embedding = (
        nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if module.use_time_embedding
        else None
    )
    if fusion == "channel" and norm == "old":
        module.channel_reduce = nn.Sequential(
            nn.LayerNorm(hidden_dim * num_features),
            nn.Linear(hidden_dim * num_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    elif fusion == "channel":
        module.channel_reduce_a = nn.Linear(hidden_dim * num_features, hidden_dim)
        module.channel_reduce_b = nn.Sequential(
            nn.Linear(hidden_dim * num_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        module.channel_reduce_norm = nn.LayerNorm(hidden_dim)
        module.channel_reduce = None
    else:
        module.channel_reduce = None

    ff_dim = max(hidden_dim, int(round(hidden_dim * mlp_ratio)))
    module.fusion_blocks = nn.ModuleList(
        [
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=transformer_heads,
                dim_feedforward=ff_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(transformer_layers)
        ]
    )
    if projector_residual_mode == "gated_concat":
        module.projector_residual_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * transformer_layers),
            nn.Linear(hidden_dim * transformer_layers, transformer_layers),
        )
        nn.init.zeros_(module.projector_residual_gate[-1].weight)
        nn.init.zeros_(module.projector_residual_gate[-1].bias)
        module.projector_residual_scale = nn.Parameter(
            torch.full((1,), float(projector_residual_init))
        )
    else:
        module.projector_residual_gate = None
        module.projector_residual_scale = None
    if text_conditioning == "film":
        module.text_conditioning_norm = nn.LayerNorm(hidden_dim)
        module.text_conditioning_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        nn.init.zeros_(module.text_conditioning_mlp[-1].weight)
        nn.init.zeros_(module.text_conditioning_mlp[-1].bias)
    else:
        module.text_conditioning_norm = None
        module.text_conditioning_mlp = None
    module._opd_mf_input_stats_call_idx = 0
    module.last_projector_residual_weights = None


def _feature_stats(tensor: torch.Tensor) -> Dict[str, float]:
    x = tensor.detach().to(torch.float32)
    count = torch.tensor(float(x.numel()), device=x.device, dtype=torch.float64)
    x64 = x.to(torch.float64)
    total = x64.sum()
    total_sq = x64.square().sum()
    min_value = x.min().to(torch.float64)
    max_value = x.max().to(torch.float64)
    if dist.is_available() and dist.is_initialized():
        for value in (count, total, total_sq):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        dist.all_reduce(min_value, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_value, op=dist.ReduceOp.MAX)
    mean = total / count.clamp_min(1.0)
    var = (total_sq / count.clamp_min(1.0) - mean.square()).clamp_min(0.0)
    return {
        "mean": float(mean.item()),
        "std": float(var.sqrt().item()),
        "min": float(min_value.item()),
        "max": float(max_value.item()),
    }


def _maybe_log_input_feature_stats(module, ordered: List[torch.Tensor]):
    if os.environ.get("OPD_LOG_MULTIFEATURE_INPUT_STATS", "0") != "1":
        return
    module._opd_mf_input_stats_call_idx += 1
    every = max(1, int(os.environ.get("OPD_LOG_MULTIFEATURE_INPUT_STATS_EVERY", "1")))
    should_log = module._opd_mf_input_stats_call_idx % every == 0
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    for name, tensor in zip(module.feature_order, ordered):
        stats = _feature_stats(tensor)
        if should_log and rank == 0:
            print(
                "opd_multifeature_input_stats "
                f"call={module._opd_mf_input_stats_call_idx} "
                f"feature={name} "
                f"shape={tuple(tensor.shape)} "
                f"mean={stats['mean']:.8f} "
                f"std={stats['std']:.8f} "
                f"min={stats['min']:.8f} "
                f"max={stats['max']:.8f}",
                flush=True,
            )


def _ordered_features(module, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    if getattr(module, "pre_projector_only", False):
        if "pre_projector" not in features:
            raise ValueError("Missing discriminator pre_projector tensor")
        pre_projector = features["pre_projector"]
        if pre_projector.ndim != 3:
            raise ValueError("pre_projector tensor must have shape [B, L, D]")
        if int(pre_projector.shape[-1]) != int(module.hidden_dim):
            raise ValueError(
                "pre_projector hidden dim mismatch: "
                f"expected {module.hidden_dim}, got {pre_projector.shape[-1]}"
            )
        return [pre_projector for _ in module.feature_order]

    missing = [name for name in module.feature_order if name not in features]
    if missing:
        raise ValueError(f"Missing discriminator multi-feature tensors: {missing}")
    ordered = [features[name] for name in module.feature_order]
    first = ordered[0]
    if first.ndim != 3:
        raise ValueError("multi-feature tensors must have shape [B, L, D]")
    batch, seq_len, hidden = first.shape
    if hidden != module.hidden_dim:
        raise ValueError(
            f"multi-feature hidden dim mismatch: expected {module.hidden_dim}, got {hidden}"
        )
    for tensor in ordered[1:]:
        if tensor.shape != (batch, seq_len, hidden):
            raise ValueError("all multi-feature tensors must have the same [B, L, D] shape")
    return ordered


def _text_film_params(
    module,
    text_features: torch.Tensor,
    text_mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    if module.text_conditioning_mlp is None:
        return None
    if text_features is None:
        raise ValueError("text_features is required when text_conditioning='film'")
    if text_features.ndim != 3:
        raise ValueError("text_features must have shape [B, L, D]")
    if text_features.shape[0] != batch_size:
        raise ValueError("text_features batch size must match multi-feature batch size")
    if text_features.shape[-1] != module.hidden_dim:
        raise ValueError(
            f"text_features hidden dim mismatch: expected {module.hidden_dim}, "
            f"got {text_features.shape[-1]}"
        )
    text_features = text_features.to(device=device, dtype=dtype)
    if text_mask is not None:
        if text_mask.shape != text_features.shape[:2]:
            raise ValueError("text_mask must have shape [B, L] matching text_features")
        weights = text_mask.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        summary = (text_features * weights.unsqueeze(-1)).sum(dim=1) / denom
    else:
        summary = text_features.mean(dim=1)
    film = module.text_conditioning_mlp(module.text_conditioning_norm(summary))
    scale, shift = film.chunk(2, dim=-1)
    return scale.view(batch_size, 1, -1), shift.view(batch_size, 1, -1)


def _fused_tokens(
    module,
    features: Dict[str, torch.Tensor],
    tt: torch.Tensor | None,
    text_features: torch.Tensor,
    text_mask: torch.Tensor,
    ref_weight: torch.Tensor,
) -> torch.Tensor:
    ordered = module._ordered_features(features)
    module._maybe_log_input_feature_stats(ordered)
    dtype = ref_weight.dtype
    device = ref_weight.device
    enriched = []
    for idx, tensor in enumerate(ordered):
        h = tensor.to(device=device, dtype=dtype)
        level = module.feature_level_embedding[idx].to(device=device, dtype=dtype)
        enriched.append(h + level.view(1, 1, -1))

    if module.feature_norms is not None:
        enriched = [norm(h) for norm, h in zip(module.feature_norms, enriched)]

    if getattr(module, "lightweight_ocr_path", False):
        h = enriched[-1]
    elif module.fusion == "channel":
        h = torch.cat(enriched, dim=-1)
        if module.norm == "new":
            h = module.channel_reduce_norm(module.channel_reduce_a(h) + module.channel_reduce_b(h))
        else:
            h = module.channel_reduce(h)
    else:
        h = torch.cat(enriched, dim=1)

    if module.tt_embedding is not None:
        if tt is None:
            tt = torch.zeros(h.shape[0], device=device, dtype=dtype)
        tt = tt.reshape(-1).to(device=device, dtype=dtype)
        if tt.numel() == 1 and h.shape[0] != 1:
            tt = tt.expand(h.shape[0])
        if tt.numel() != h.shape[0]:
            raise ValueError(
                f"tt batch size must match features batch size, got {tt.numel()} and {h.shape[0]}"
            )
        h = h + module.tt_embedding(tt.view(-1, 1)).view(
            h.shape[0],
            1,
            module.hidden_dim,
        )

    text_film = module._text_film_params(
        text_features,
        text_mask,
        batch_size=h.shape[0],
        device=device,
        dtype=dtype,
    )
    block_outputs = []
    if hasattr(module, "last_ocr_feature_branch_tokens"):
        module.last_ocr_feature_branch_tokens = None
        module.last_ocr_text_branch_tokens = None
        if (
            getattr(module, "ocr_feature_branch", None) is not None
            and module.ocr_feature_branch_layer == 0
        ):
            module.last_ocr_feature_branch_tokens = module.ocr_feature_branch(h)
    for block_idx, block in enumerate(module.fusion_blocks, start=1):
        if text_film is not None:
            scale, shift = text_film
            h = h * (1.0 + scale) + shift
        if getattr(module, "gradient_checkpointing", False) and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            h = checkpoint(block, h, use_reentrant=False)
        else:
            h = block(h)
        if (
            getattr(module, "ocr_feature_branch", None) is not None
            and block_idx == module.ocr_feature_branch_layer
        ):
            module.last_ocr_feature_branch_tokens = module.ocr_feature_branch(h)
        if module.projector_residual_gate is not None:
            block_outputs.append(h)
    if module.projector_residual_gate is not None:
        gate_input = torch.cat(
            [state.mean(dim=1) for state in block_outputs],
            dim=-1,
        )
        gate_logits = module.projector_residual_gate(gate_input)
        gate = (
            torch.sigmoid(gate_logits)
            if gate_logits.shape[-1] == 1
            else torch.softmax(gate_logits, dim=-1)
        )
        residual = torch.zeros_like(h)
        for idx, state in enumerate(block_outputs):
            residual = residual + gate[:, idx].view(-1, 1, 1).to(dtype=state.dtype) * state
        h = (
            h
            + module.projector_residual_scale.to(
                device=h.device,
                dtype=h.dtype,
            )
            * residual
        )
        module.last_projector_residual_weights = gate.detach()
    else:
        module.last_projector_residual_weights = None
    return h


class ZImageSharedDualProjectorTrunk(nn.Module):
    """Shared pre-fusion stack and lower fusion blocks for exit-specific heads."""

    def __init__(
        self,
        source: "ZImageDualProjectorMultiFeatureDiscriminatorHead",
        shared_fusion_layers: int,
    ):
        super().__init__()
        shared_fusion_layers = int(shared_fusion_layers)
        total_layers = len(source.fusion_blocks)
        if shared_fusion_layers < 1 or shared_fusion_layers > total_layers:
            raise ValueError(
                f"shared_fusion_layers must be in [1, {total_layers}], got {shared_fusion_layers}"
            )
        if source.projector_residual_gate is not None:
            raise ValueError(
                "shared exit discriminator does not support projector_residual_mode != 'none'"
            )
        self.hidden_dim = source.hidden_dim
        self.num_features = source.num_features
        self.fusion = source.fusion
        self.norm = source.norm
        self.projector_residual_mode = source.projector_residual_mode
        self.text_conditioning = source.text_conditioning
        self.use_time_embedding = bool(source.use_time_embedding)
        self.feature_order = source.feature_order
        self.feature_level_embedding = source.feature_level_embedding
        self.feature_norms = source.feature_norms
        self.tt_embedding = source.tt_embedding
        self.channel_reduce = source.channel_reduce
        if hasattr(source, "channel_reduce_a"):
            self.channel_reduce_a = source.channel_reduce_a
            self.channel_reduce_b = source.channel_reduce_b
            self.channel_reduce_norm = source.channel_reduce_norm
        else:
            self.channel_reduce_a = None
            self.channel_reduce_b = None
            self.channel_reduce_norm = None
        self.text_conditioning_norm = source.text_conditioning_norm
        self.text_conditioning_mlp = source.text_conditioning_mlp
        self.fusion_blocks = nn.ModuleList(
            [source.fusion_blocks[idx] for idx in range(shared_fusion_layers)]
        )
        self.shared_fusion_layers = shared_fusion_layers
        self._opd_mf_input_stats_call_idx = 0
        self.last_projector_residual_weights = None

    @staticmethod
    def _feature_stats(tensor: torch.Tensor) -> Dict[str, float]:
        return _feature_stats(tensor)

    def _maybe_log_input_feature_stats(self, ordered: List[torch.Tensor]):
        return _maybe_log_input_feature_stats(self, ordered)

    def _ordered_features(self, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        return _ordered_features(self, features)

    def _text_film_params(
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        return _text_film_params(
            self,
            text_features,
            text_mask,
            batch_size,
            device,
            dtype,
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        tt: torch.Tensor | None,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        ref_weight: torch.Tensor,
    ) -> SharedTrunkOutput:
        ordered = self._ordered_features(features)
        self._maybe_log_input_feature_stats(ordered)
        dtype = ref_weight.dtype
        device = ref_weight.device
        enriched = []
        for idx, tensor in enumerate(ordered):
            h = tensor.to(device=device, dtype=dtype)
            level = self.feature_level_embedding[idx].to(device=device, dtype=dtype)
            enriched.append(h + level.view(1, 1, -1))

        if self.feature_norms is not None:
            enriched = [norm(h) for norm, h in zip(self.feature_norms, enriched)]

        if self.fusion == "channel":
            h = torch.cat(enriched, dim=-1)
            if self.norm == "new":
                h = self.channel_reduce_norm(self.channel_reduce_a(h) + self.channel_reduce_b(h))
            else:
                h = self.channel_reduce(h)
        else:
            h = torch.cat(enriched, dim=1)

        if self.tt_embedding is not None:
            if tt is None:
                tt = torch.zeros(h.shape[0], device=device, dtype=dtype)
            tt = tt.reshape(-1).to(device=device, dtype=dtype)
            if tt.numel() == 1 and h.shape[0] != 1:
                tt = tt.expand(h.shape[0])
            if tt.numel() != h.shape[0]:
                raise ValueError(
                    "tt batch size must match features batch size, "
                    f"got {tt.numel()} and {h.shape[0]}"
                )
            h = h + self.tt_embedding(tt.view(-1, 1)).view(
                h.shape[0],
                1,
                self.hidden_dim,
            )

        text_film = self._text_film_params(
            text_features,
            text_mask,
            batch_size=h.shape[0],
            device=device,
            dtype=dtype,
        )
        for block in self.fusion_blocks:
            if text_film is not None:
                scale, shift = text_film
                h = h * (1.0 + scale) + shift
            h = block(h)
        self.last_projector_residual_weights = None
        return SharedTrunkOutput(tokens=h, text_film=text_film)


class ZImageDualProjectorExitTail(nn.Module):
    """Exit-specific upper fusion blocks and align/GAN projectors."""

    supports_dual_output = True

    def __init__(
        self,
        source: "ZImageDualProjectorMultiFeatureDiscriminatorHead",
        shared_fusion_layers: int,
        *,
        copy_modules: bool = False,
    ):
        super().__init__()
        shared_fusion_layers = int(shared_fusion_layers)
        total_layers = len(source.fusion_blocks)
        if shared_fusion_layers < 1 or shared_fusion_layers > total_layers:
            raise ValueError(
                f"shared_fusion_layers must be in [1, {total_layers}], got {shared_fusion_layers}"
            )
        self.hidden_dim = source.hidden_dim
        self.align_output_dim = source.align_output_dim
        tail_blocks = [
            source.fusion_blocks[idx] for idx in range(shared_fusion_layers, total_layers)
        ]
        if copy_modules:
            tail_blocks = [copy.deepcopy(block) for block in tail_blocks]
            self.align_norm = copy.deepcopy(source.align_norm)
            self.align_projector = copy.deepcopy(source.align_projector)
            self.gan_norm = copy.deepcopy(source.gan_norm)
            self.gan_projector = copy.deepcopy(source.gan_projector)
        else:
            self.align_norm = source.align_norm
            self.align_projector = source.align_projector
            self.gan_norm = source.gan_norm
            self.gan_projector = source.gan_projector
        self.fusion_blocks = nn.ModuleList(tail_blocks)

    def forward(
        self,
        h: torch.Tensor,
        text_film: tuple[torch.Tensor, torch.Tensor] | None = None,
        output: str = "both",
    ) -> torch.Tensor | DualProjectorOutput:
        output = str(output).lower()
        if output not in {"both", "align", "gan"}:
            raise ValueError("dual discriminator output must be 'both', 'align', or 'gan'")
        for block in self.fusion_blocks:
            if text_film is not None:
                scale, shift = text_film
                h = h * (1.0 + scale) + shift
            h = block(h)
        align = None
        logits = None
        if output in {"both", "align"}:
            align = self.align_projector(self.align_norm(h))
        if output in {"both", "gan"}:
            logits = self.gan_projector(self.gan_norm(h).mean(dim=1)).flatten()
        if output == "align":
            return align
        if output == "gan":
            return logits
        return DualProjectorOutput(align=align, logits=logits)


class _SharedDualProjectorExitView:
    supports_dual_output = True

    def __init__(
        self,
        parent: "ZImageSharedDualProjectorMultiFeatureDiscriminatorHead",
        discriminator_head: str,
    ):
        self.parent = parent
        self.discriminator_head = discriminator_head

    def __call__(
        self,
        features: Dict[str, torch.Tensor],
        tt: torch.Tensor | None = None,
        text_features: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        output: str = "both",
    ) -> torch.Tensor | DualProjectorOutput:
        return self.parent(
            features,
            tt=tt,
            text_features=text_features,
            text_mask=text_mask,
            output=output,
            discriminator_head=self.discriminator_head,
        )


class ZImageSharedDualProjectorMultiFeatureDiscriminatorHead(nn.Module):
    """Dual-projector head with a shared B/C trunk and exit-specific tails."""

    supports_dual_output = True
    is_shared_exit_discriminator = True

    def __init__(
        self,
        shared_trunk: ZImageSharedDualProjectorTrunk,
        exit1_tail: ZImageDualProjectorExitTail,
        exit2_tail: ZImageDualProjectorExitTail,
    ):
        super().__init__()
        self.shared_trunk = shared_trunk
        self.exit1_tail = exit1_tail
        self.exit2_tail = exit2_tail
        self.shared_fusion_layers = int(shared_trunk.shared_fusion_layers)
        self.hidden_dim = int(shared_trunk.hidden_dim)
        self.align_output_dim = int(exit1_tail.align_output_dim)
        self._exit1_view = _SharedDualProjectorExitView(self, "exit1")
        self._exit2_view = _SharedDualProjectorExitView(self, "exit2")

    @classmethod
    def from_exit1_head(
        cls,
        exit1_head: "ZImageDualProjectorMultiFeatureDiscriminatorHead",
        shared_fusion_layers: int,
    ) -> "ZImageSharedDualProjectorMultiFeatureDiscriminatorHead":
        shared_trunk = ZImageSharedDualProjectorTrunk(
            exit1_head,
            shared_fusion_layers,
        )
        exit1_tail = ZImageDualProjectorExitTail(
            exit1_head,
            shared_fusion_layers,
            copy_modules=False,
        )
        exit2_tail = ZImageDualProjectorExitTail(
            exit1_head,
            shared_fusion_layers,
            copy_modules=True,
        )
        return cls(shared_trunk, exit1_tail, exit2_tail)

    def select_exit_module(self, discriminator_head: str):
        head_name = str(discriminator_head or "exit1").lower()
        if head_name == "exit1":
            return self._exit1_view
        if head_name == "exit2":
            return self._exit2_view
        raise ValueError(
            f"discriminator_head must be 'exit1' or 'exit2', got {discriminator_head!r}"
        )

    def exit_tail_parameters(self, discriminator_head: str):
        head_name = str(discriminator_head or "exit1").lower()
        if head_name == "exit1":
            return self.exit1_tail.parameters()
        if head_name == "exit2":
            return self.exit2_tail.parameters()
        raise ValueError(
            f"discriminator_head must be 'exit1' or 'exit2', got {discriminator_head!r}"
        )

    def shared_trunk_parameters(self):
        return self.shared_trunk.parameters()

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        tt: torch.Tensor | None = None,
        text_features: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        output: str = "both",
        discriminator_head: str = "exit1",
    ) -> torch.Tensor | DualProjectorOutput:
        tail = self.exit1_tail if str(discriminator_head).lower() == "exit1" else None
        if tail is None and str(discriminator_head).lower() == "exit2":
            tail = self.exit2_tail
        if tail is None:
            raise ValueError(
                f"discriminator_head must be 'exit1' or 'exit2', got {discriminator_head!r}"
            )
        trunk_output = self.shared_trunk(
            features,
            tt=tt,
            text_features=text_features,
            text_mask=text_mask,
            ref_weight=tail.align_projector[-1].weight,
        )
        return tail(
            trunk_output.tokens,
            text_film=trunk_output.text_film,
            output=output,
        )


class ZImageMultiFeatureDiscriminatorHead(nn.Module):
    """Fuse frozen teacher token features into one discriminator logit."""

    def __init__(
        self,
        hidden_dim: int,
        num_features: int = 4,
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
        super().__init__()
        output_dim = int(output_dim)
        output_mode = str(output_mode).lower()
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if output_mode not in {"pooled", "tokens"}:
            raise ValueError("discriminator_multifeature_output_mode must be 'pooled' or 'tokens'")
        _init_multi_feature_fusion(
            self,
            hidden_dim=hidden_dim,
            num_features=num_features,
            fusion=fusion,
            norm=norm,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            mlp_ratio=mlp_ratio,
            projector_residual_mode=projector_residual_mode,
            projector_residual_init=projector_residual_init,
            text_conditioning=text_conditioning,
            use_time_embedding=use_time_embedding,
        )
        self.output_dim = output_dim
        self.output_mode = output_mode
        self.pre_projector_only = bool(pre_projector_only)
        self.ocr_score_head_enabled = bool(ocr_score_head)
        self.ocr_feature_adapter_enabled = bool(ocr_feature_adapter)
        self.ocr_feature_branch_layer = int(ocr_feature_branch_layer)
        self.ocr_text_branch_output_dim = int(ocr_text_branch_output_dim)
        self.ocr_text_branch_max_tokens = int(ocr_text_branch_max_tokens)
        self.ocr_text_branch_enabled = (
            self.ocr_score_head_enabled
            and self.ocr_text_branch_output_dim > 0
            and self.ocr_text_branch_max_tokens > 0
        )
        self.lightweight_ocr_path = self.pre_projector_only
        if self.lightweight_ocr_path:
            self.channel_reduce = None
            self.channel_reduce_a = None
            self.channel_reduce_b = None
            self.channel_reduce_norm = None
        ocr_feature_branch_output_dim = int(ocr_feature_branch_output_dim)
        self.ocr_feature_branch_requested = (
            self.ocr_feature_branch_layer > 0 or ocr_feature_branch_output_dim > 0
        )
        self.ocr_feature_branch_output_dim = (
            output_dim if ocr_feature_branch_output_dim <= 0 else ocr_feature_branch_output_dim
        )
        if self.ocr_feature_branch_requested and (
            self.ocr_feature_branch_layer < 0
            or self.ocr_feature_branch_layer > int(transformer_layers)
        ):
            raise ValueError(
                "ocr_feature_branch_layer must be in [0, transformer_layers], "
                f"got {self.ocr_feature_branch_layer} for {int(transformer_layers)} layers"
            )
        if self.ocr_feature_branch_requested and self.ocr_feature_branch_output_dim <= 0:
            raise ValueError("ocr_feature_branch_output_dim must be > 0")
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        out_hidden_dim = (
            min(self.hidden_dim, max(output_dim, 256))
            if self.lightweight_ocr_path
            else self.hidden_dim
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, out_hidden_dim),
            nn.SiLU(),
            nn.Linear(out_hidden_dim, output_dim),
        )
        self.ocr_score_prompt_dim = output_dim if self.ocr_score_head_enabled else 0
        score_input_dim = (
            output_dim
            + (self.ocr_text_branch_output_dim if self.ocr_text_branch_enabled else 0)
            + self.ocr_score_prompt_dim
        )
        score_hidden_dim = (
            min(score_input_dim, 512) if self.lightweight_ocr_path else score_input_dim
        )
        self.ocr_score_head = (
            nn.Sequential(
                nn.LayerNorm(score_input_dim * 2),
                nn.Linear(score_input_dim * 2, score_hidden_dim),
                nn.SiLU(),
                nn.Linear(score_hidden_dim, 1),
            )
            if self.ocr_score_head_enabled
            else None
        )
        if self.ocr_score_head is not None:
            nn.init.xavier_uniform_(self.ocr_score_head[-1].weight, gain=0.01)
            nn.init.zeros_(self.ocr_score_head[-1].bias)
        self.ocr_feature_adapter = (
            nn.Sequential(
                nn.LayerNorm(output_dim),
                nn.Linear(output_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            )
            if self.ocr_score_head_enabled and self.ocr_feature_adapter_enabled
            else None
        )
        feature_branch_hidden_dim = (
            min(self.hidden_dim, max(self.ocr_feature_branch_output_dim, 256))
            if self.lightweight_ocr_path
            else self.hidden_dim
        )
        self.ocr_feature_branch = (
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, feature_branch_hidden_dim),
                nn.SiLU(),
                nn.Linear(feature_branch_hidden_dim, self.ocr_feature_branch_output_dim),
            )
            if self.ocr_feature_branch_requested
            else None
        )
        text_branch_hidden_dim = (
            min(self.hidden_dim, 1024) if self.lightweight_ocr_path else self.hidden_dim
        )
        self.ocr_text_branch = (
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, text_branch_hidden_dim),
                nn.SiLU(),
                nn.Linear(text_branch_hidden_dim, self.ocr_text_branch_output_dim),
            )
            if self.ocr_text_branch_enabled
            else None
        )
        self.last_ocr_feature_branch_tokens = None
        self.last_ocr_text_branch_tokens = None
        self.last_ocr_score_debug = None

    @staticmethod
    def _feature_stats(tensor: torch.Tensor) -> Dict[str, float]:
        return _feature_stats(tensor)

    def _maybe_log_input_feature_stats(self, ordered: List[torch.Tensor]):
        return _maybe_log_input_feature_stats(self, ordered)

    def _ordered_features(self, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        return _ordered_features(self, features)

    def _text_film_params(
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        return _text_film_params(
            self,
            text_features,
            text_mask,
            batch_size,
            device,
            dtype,
        )

    def ocr_score_from_tokens(
        self,
        tokens: torch.Tensor,
        prompt_features: torch.Tensor = None,
        prompt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.ocr_score_head is None:
            raise ValueError("OCR score head is not initialized")
        if tokens.ndim != 3:
            raise ValueError("OCR score head expects tokens with shape [B, N, D]")
        if tokens.shape[-1] != self.output_dim:
            raise ValueError(
                f"OCR score head dim mismatch: expected {self.output_dim}, got {tokens.shape[-1]}"
            )
        score_tokens = [tokens]
        if self.ocr_text_branch_enabled:
            if self.last_ocr_text_branch_tokens is None:
                raise ValueError("OCR text branch tokens are not available")
            text_tokens = self.last_ocr_text_branch_tokens
            if text_tokens.ndim != 3:
                raise ValueError("OCR text branch tokens must have shape [B, N, D]")
            if text_tokens.shape[0] != tokens.shape[0]:
                raise ValueError(
                    "OCR score text branch batch mismatch: "
                    f"{text_tokens.shape[0]} vs {tokens.shape[0]}"
                )
            if text_tokens.shape[-1] != self.ocr_text_branch_output_dim:
                raise ValueError(
                    "OCR score text branch dim mismatch: expected "
                    f"{self.ocr_text_branch_output_dim}, got {text_tokens.shape[-1]}"
                )
            score_tokens.append(text_tokens)
        if self.ocr_score_prompt_dim > 0:
            if prompt_features is None:
                prompt_summary = tokens.detach().new_zeros(
                    (int(tokens.shape[0]), self.ocr_score_prompt_dim)
                )
            else:
                if prompt_features.ndim != 3:
                    raise ValueError("OCR score prompt features must have shape [B, L, D]")
                if prompt_features.shape[0] != tokens.shape[0]:
                    raise ValueError(
                        "OCR score prompt batch mismatch: "
                        f"{prompt_features.shape[0]} vs {tokens.shape[0]}"
                    )
                prompt_features = prompt_features.to(device=tokens.device, dtype=tokens.dtype)
                if prompt_mask is not None:
                    if prompt_mask.shape != prompt_features.shape[:2]:
                        raise ValueError("OCR score prompt mask must have shape [B, L]")
                    weights = prompt_mask.to(device=tokens.device, dtype=tokens.dtype).clamp(
                        0.0, 1.0
                    )
                    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
                    prompt_summary = (prompt_features * weights.unsqueeze(-1)).sum(dim=1) / denom
                else:
                    prompt_summary = prompt_features.mean(dim=1)
            if int(prompt_summary.shape[-1]) >= int(self.ocr_score_prompt_dim):
                prompt_summary = prompt_summary[:, : int(self.ocr_score_prompt_dim)]
            else:
                pad = prompt_summary.new_zeros(
                    (
                        int(prompt_summary.shape[0]),
                        int(self.ocr_score_prompt_dim) - int(prompt_summary.shape[-1]),
                    )
                )
                prompt_summary = torch.cat([prompt_summary, pad], dim=-1)
            prompt_tokens = prompt_summary.unsqueeze(1)
            score_tokens.append(prompt_tokens)
        token_mean = torch.cat([item.mean(dim=1) for item in score_tokens], dim=-1)
        token_std = torch.cat(
            [item.float().std(dim=1, unbiased=False).to(dtype=item.dtype) for item in score_tokens],
            dim=-1,
        )
        pooled = torch.cat([token_mean, token_std], dim=-1)
        output = self.ocr_score_head(pooled)
        return output.flatten()

    def ocr_feature_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.ocr_feature_branch is not None:
            if self.last_ocr_feature_branch_tokens is None:
                raise ValueError("OCR feature branch tokens are not available")
            return self.last_ocr_feature_branch_tokens
        if self.ocr_feature_adapter is None:
            return tokens
        if tokens.ndim != 3:
            raise ValueError("OCR feature adapter expects tokens with shape [B, N, D]")
        if tokens.shape[-1] != self.output_dim:
            raise ValueError(
                f"OCR feature adapter dim mismatch: expected {self.output_dim}, "
                f"got {tokens.shape[-1]}"
            )
        return self.ocr_feature_adapter(tokens)

    def ocr_text_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        del tokens
        if self.ocr_text_branch is None:
            raise ValueError("OCR text branch is not initialized")
        if self.last_ocr_text_branch_tokens is None:
            raise ValueError("OCR text branch tokens are not available")
        return self.last_ocr_text_branch_tokens

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        tt: torch.Tensor | None = None,
        text_features: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        return_prelogit_features: bool = False,
    ) -> torch.Tensor:
        h = _fused_tokens(
            self,
            features,
            tt,
            text_features,
            text_mask,
            self.out_mlp[-1].weight,
        )
        h = self.out_norm(h)
        if self.ocr_text_branch is not None:
            text_token_count = min(int(h.shape[1]), int(self.ocr_text_branch_max_tokens))
            self.last_ocr_text_branch_tokens = self.ocr_text_branch(h[:, :text_token_count])
        if self.output_mode == "tokens":
            if return_prelogit_features:
                prelogit = self.out_mlp[:-1](h)
                return self.out_mlp[-1](prelogit), prelogit
            output = self.out_mlp(h)
            return output
        output = self.out_mlp(h.mean(dim=1))
        if self.output_dim == 1:
            return output.flatten()
        return output


class ZImageDualProjectorMultiFeatureDiscriminatorHead(nn.Module):
    """Fuse frozen teacher token features into align tokens and GAN logits."""

    supports_dual_output = True

    def __init__(
        self,
        hidden_dim: int,
        num_features: int = 4,
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
        super().__init__()
        align_output_dim = int(align_output_dim)
        if align_output_dim <= 0:
            raise ValueError("align_output_dim must be > 0")
        _init_multi_feature_fusion(
            self,
            hidden_dim=hidden_dim,
            num_features=num_features,
            fusion=fusion,
            norm=norm,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            mlp_ratio=mlp_ratio,
            projector_residual_mode=projector_residual_mode,
            projector_residual_init=projector_residual_init,
            text_conditioning=text_conditioning,
            use_time_embedding=use_time_embedding,
        )
        self.align_output_dim = align_output_dim
        self.align_norm = nn.LayerNorm(self.hidden_dim)
        self.align_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, align_output_dim),
        )
        self.gan_norm = nn.LayerNorm(self.hidden_dim)
        self.gan_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    @staticmethod
    def _feature_stats(tensor: torch.Tensor) -> Dict[str, float]:
        return _feature_stats(tensor)

    def _maybe_log_input_feature_stats(self, ordered: List[torch.Tensor]):
        return _maybe_log_input_feature_stats(self, ordered)

    def _ordered_features(self, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        return _ordered_features(self, features)

    def _text_film_params(
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        return _text_film_params(
            self,
            text_features,
            text_mask,
            batch_size,
            device,
            dtype,
        )

    def _fused_tokens(
        self,
        features: Dict[str, torch.Tensor],
        tt: torch.Tensor | None = None,
        text_features: torch.Tensor = None,
        text_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        return _fused_tokens(
            self,
            features,
            tt,
            text_features,
            text_mask,
            self.align_projector[-1].weight,
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        tt: torch.Tensor | None = None,
        text_features: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        output: str = "both",
    ) -> torch.Tensor | DualProjectorOutput:
        output = str(output).lower()
        if output not in {"both", "align", "gan"}:
            raise ValueError("dual discriminator output must be 'both', 'align', or 'gan'")
        h = self._fused_tokens(
            features,
            tt=tt,
            text_features=text_features,
            text_mask=text_mask,
        )
        align = None
        logits = None
        if output in {"both", "align"}:
            align = self.align_projector(self.align_norm(h))
        if output in {"both", "gan"}:
            logits = self.gan_projector(self.gan_norm(h).mean(dim=1)).flatten()
        if output == "align":
            return align
        if output == "gan":
            return logits
        return DualProjectorOutput(align=align, logits=logits)
