import math

import torch
import torch.nn as nn


class LoRALayer(nn.Linear):
    def __init__(self, in_dim, out_dim, rank, alpha, weak_lora_alpha=0.1):
        super().__init__(in_dim, out_dim)
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.weak_lora_alpha = weak_lora_alpha
        self.use_lora = True
        self.weak_lora = False
        nn.init.normal_(self.A, mean=0.0, std=1.0 / math.sqrt(max(1, in_dim)))
        nn.init.zeros_(self.B)

    def forward(self, x):
        linear_out = super().forward(x)
        if self.use_lora:
            if self.weak_lora:
                return linear_out + self.alpha * (x @ self.A @ self.B) * self.weak_lora_alpha
            return linear_out + self.alpha * (x @ self.A @ self.B)
        if self.training:
            return linear_out + 0 * (x @ self.A @ self.B)
        return linear_out


def replace_linear_with_lora(module, rank=16, alpha=1.0, tag=0, weak_lora_alpha=1.0):
    # For Z-Image we should not rely on model-specific block names.
    # Replace all Linear layers recursively.
    _ = tag
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            param_device = getattr(child.weight, "device", None)
            use_meta = param_device is not None and param_device.type == "meta"
            if use_meta:
                with torch.device("meta"):
                    new_layer = LoRALayer(
                        child.in_features,
                        child.out_features,
                        rank,
                        alpha,
                        weak_lora_alpha=weak_lora_alpha,
                    )
            else:
                new_layer = LoRALayer(
                    child.in_features,
                    child.out_features,
                    rank,
                    alpha,
                    weak_lora_alpha=weak_lora_alpha,
                )
                new_layer.to(device=child.weight.device, dtype=child.weight.dtype)
                new_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None and new_layer.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, new_layer)
        else:
            replace_linear_with_lora(child, rank, alpha, weak_lora_alpha=weak_lora_alpha)


def _set_lora_mode(model, *, use_lora, weak_lora, alpha=None):
    for _, module in model.named_modules():
        if isinstance(module, LoRALayer):
            for n, param in module.named_parameters():
                if n in ["A", "B"]:
                    param.requires_grad_(True)
            module.use_lora = use_lora
            module.weak_lora = weak_lora
            if alpha is not None:
                module.weak_lora_alpha = alpha


def lora_false(model):
    _set_lora_mode(model, use_lora=False, weak_lora=False)


def lora_true(model, alpha=1.0):
    _set_lora_mode(model, use_lora=True, weak_lora=False, alpha=alpha)


def weak_lora(model, alpha=0.25):
    _set_lora_mode(model, use_lora=True, weak_lora=True, alpha=alpha)


def ori_lora(model):
    _set_lora_mode(model, use_lora=True, weak_lora=False)


def iter_lora_parameters(model):
    for module in model.modules():
        if isinstance(module, LoRALayer):
            yield module.A
            yield module.B


def lora_state_dict(model):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            state[f"{name}.A"] = module.A.detach().cpu()
            state[f"{name}.B"] = module.B.detach().cpu()
    return state


def load_lora_state_dict(model, state, *, strict=True):
    modules = dict(model.named_modules())
    expected = set()
    for name, module in modules.items():
        if not isinstance(module, LoRALayer):
            continue
        for parameter_name in ("A", "B"):
            key = f"{name}.{parameter_name}"
            expected.add(key)
            if key in state:
                getattr(module, parameter_name).data.copy_(
                    state[key].to(
                        device=getattr(module, parameter_name).device,
                        dtype=getattr(module, parameter_name).dtype,
                    )
                )
    missing = sorted(expected - set(state))
    unexpected = sorted(set(state) - expected)
    if strict and (missing or unexpected):
        raise KeyError(f"LoRA state mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    return {"missing": missing, "unexpected": unexpected}
