"""Phase-local score shard offload for the pinned PyTorch 2.8 FSDP1 runtime.

No score autograd graph may be live. Optimizer parameter objects and GPU Adam
states stay unchanged. Only full-precision local weight shards move to CPU.
"""
from contextlib import contextmanager
import time
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


@contextmanager
def offload_score_shards(models, enabled=True):
    stats = {}
    if not enabled:
        yield stats
        return
    handles = [m._handle for model in models for m in FSDP.fsdp_modules(model)
               if m._handle is not None]
    # Validate everything before mutating any handle. FSDP lazy initialization
    # must already have happened through the preceding no-grad score forwards.
    for h in handles:
        p = h.flat_param
        if not h._use_orig_params or not h.is_sharded(p):
            raise RuntimeError("Score offload requires sharded FSDP1 use_orig_params")
        if p.grad is not None or any(q.grad is not None for q in p._params):
            raise RuntimeError("Score offload requires cleared score gradients")
        if p.device.type != "cuda" or p.data_ptr() != p._local_shard.data_ptr():
            raise RuntimeError("Score offload requires the original GPU local shard")
    device = handles[0].flat_param.device
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    started = time.monotonic()
    moved = []
    try:
        for h in handles:
            h.flat_param_to("cpu")
            h.flat_param._local_shard = h.flat_param.data
            moved.append(h)
        stats.update(weight_bytes=sum(h.flat_param.numel() * h.flat_param.element_size() for h in handles),
                     allocated_bytes_freed=before - torch.cuda.memory_allocated(device),
                     offload_seconds=time.monotonic() - started)
        yield stats
    finally:
        started = time.monotonic()
        for h in moved:
            h.flat_param_to(device)
            h.flat_param._local_shard = h.flat_param.data
        torch.cuda.synchronize(device)
        stats["reload_seconds"] = time.monotonic() - started


class PhaseShardOffload:
    """Keep inactive FSDP shards on CPU across microbatches/updates.

    Call load() before any forward/checkpoint. Lazy-init on GPU before the first
    move, including frozen Real which has not yet had a forward. Never move a
    model with a live autograd graph or outstanding gradients.
    """
    def __init__(self, model):
        from torch.distributed.fsdp._runtime_utils import _lazy_init
        _lazy_init(model, model)
        self.handles = [m._handle for m in FSDP.fsdp_modules(model) if m._handle is not None]
        self.device = self.handles[0].flat_param.device
        self.on_cpu = False

    def offload(self):
        if self.on_cpu:
            return {}
        for h in self.handles:
            p = h.flat_param
            if not h._use_orig_params or not h.is_sharded(p):
                raise RuntimeError("Phase offload requires sharded original parameters")
            if p.grad is not None or any(q.grad is not None for q in p._params):
                raise RuntimeError("Phase offload requires cleared gradients")
            if p.device != self.device or p.data_ptr() != p._local_shard.data_ptr():
                raise RuntimeError("Phase offload requires the GPU local shard")
        torch.cuda.synchronize(self.device)
        before = torch.cuda.memory_allocated(self.device)
        started = time.monotonic()
        for h in self.handles:
            h.flat_param_to("cpu")
            h.flat_param._local_shard = h.flat_param.data
        self.on_cpu = True
        return {"allocated_bytes_freed": before - torch.cuda.memory_allocated(self.device),
                "weight_bytes": sum(h.flat_param.numel()*h.flat_param.element_size() for h in self.handles),
                "offload_seconds": time.monotonic()-started}

    def load(self):
        if not self.on_cpu:
            return 0.
        started = time.monotonic()
        for h in self.handles:
            h.flat_param_to(self.device)
            h.flat_param._local_shard = h.flat_param.data
        torch.cuda.synchronize(self.device)
        self.on_cpu = False
        return time.monotonic()-started
