from __future__ import annotations

import weakref
from contextlib import contextmanager

import torch

try:
    from torch.distributed.tensor import DTensor, distribute_tensor
except ImportError:  # pragma: no cover - older supported Torch builds
    DTensor = None
    distribute_tensor = None


class ShardedEMA:
    """FP32 EMA shadows for regular parameters and FSDP2 DTensors."""

    def __init__(self, module: torch.nn.Module, decay: float):
        if not 0.0 <= float(decay) <= 1.0:
            raise ValueError("EMA decay must be in [0, 1]")
        self.decay = float(decay)
        self._module_ref = weakref.ref(module)
        self._param_refs = {}
        self.ema_params = {}
        self._stash = None
        for name, parameter in module.named_parameters():
            if parameter.dtype.is_floating_point:
                self._param_refs[name] = parameter
                self.ema_params[name] = self._clone(parameter, dtype=torch.float32)

    @staticmethod
    def _is_dtensor(value):
        return DTensor is not None and isinstance(value, DTensor)

    def _clone(self, parameter, *, dtype=None):
        if self._is_dtensor(parameter):
            local = parameter.to_local().detach().clone()
            if dtype is not None:
                local = local.to(dtype=dtype)
            return DTensor.from_local(
                local,
                parameter.device_mesh,
                parameter.placements,
                shape=parameter.shape,
                stride=parameter.stride(),
            )
        clone = parameter.detach().clone()
        return clone.to(dtype=dtype) if dtype is not None else clone

    def _named_parameters(self, module=None):
        module = module or self._module_ref()
        if module is None:
            raise RuntimeError("EMA module reference is stale")
        parameters = dict(module.named_parameters())
        for name, reference in self._param_refs.items():
            current = parameters.get(name)
            if current is None or tuple(current.shape) != tuple(reference.shape):
                raise RuntimeError(f"EMA parameter structure changed after initialization: {name}")
            if current is not reference:
                self._param_refs[name] = current
        return parameters

    def _to_layout(self, source, target, *, dtype):
        if self._is_dtensor(target):
            if self._is_dtensor(source):
                if (
                    source.device_mesh != target.device_mesh
                    or source.placements != target.placements
                ):
                    source = source.redistribute(target.device_mesh, target.placements)
                return source.to(dtype=dtype)
            if tuple(source.shape) == tuple(target.to_local().shape):
                local = source.to(device=target.device, dtype=dtype)
                return DTensor.from_local(
                    local,
                    target.device_mesh,
                    target.placements,
                    shape=target.shape,
                    stride=target.stride(),
                )
            if distribute_tensor is None:
                raise RuntimeError("DTensor distribution API is unavailable")
            return distribute_tensor(
                source.to(device=target.device, dtype=dtype),
                target.device_mesh,
                target.placements,
            )
        if self._is_dtensor(source):
            source = (
                source.full_tensor()
                if tuple(source.shape) == tuple(target.shape)
                else source.to_local()
            )
        return source.to(device=target.device, dtype=dtype)

    @torch.no_grad()
    def update(self, module=None):
        for name, parameter in self._named_parameters(module).items():
            if name not in self.ema_params:
                continue
            shadow = self.ema_params[name]
            current = self._to_layout(parameter.detach(), shadow, dtype=torch.float32)
            shadow.mul_(self.decay).add_(current, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, module=None):
        if self._stash is not None:
            raise RuntimeError("EMA shadow is already applied")
        self._stash = {}
        for name, parameter in self._named_parameters(module).items():
            if name not in self.ema_params:
                continue
            self._stash[name] = self._clone(parameter)
            parameter.copy_(
                self._to_layout(self.ema_params[name], parameter, dtype=parameter.dtype)
            )

    @torch.no_grad()
    def restore(self, module=None):
        if self._stash is None:
            return
        parameters = self._named_parameters(module)
        for name, original in self._stash.items():
            parameter = parameters[name]
            parameter.copy_(self._to_layout(original, parameter, dtype=parameter.dtype))
        self._stash = None

    def state_dict(self):
        state = {"decay": self.decay, "shadows": {}}
        for name, shadow in self.ema_params.items():
            value = shadow.to_local() if self._is_dtensor(shadow) else shadow
            state["shadows"][name] = value.detach().cpu()
        return state

    @torch.no_grad()
    def load_state_dict(self, state):
        if set(state["shadows"]) != set(self.ema_params):
            missing = sorted(set(self.ema_params) - set(state["shadows"]))
            unexpected = sorted(set(state["shadows"]) - set(self.ema_params))
            raise ValueError(f"EMA state mismatch: missing={missing}, unexpected={unexpected}")
        self.decay = float(state["decay"])
        for name, shadow in self.ema_params.items():
            target = shadow.to_local() if self._is_dtensor(shadow) else shadow
            target.copy_(state["shadows"][name].to(device=target.device, dtype=target.dtype))


@contextmanager
def use_ema_weights(ema: ShardedEMA | None, module: torch.nn.Module):
    if ema is None:
        yield
        return
    try:
        ema.apply_shadow(module)
        yield
    finally:
        ema.restore(module)
