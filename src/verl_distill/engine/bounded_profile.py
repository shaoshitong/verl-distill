"""Opt-in observation only; never changes model tensors or optimizer state."""
import gzip
import json
import os
import shutil
import time
import tempfile
from pathlib import Path

import torch


class BoundedProfile:
    def __init__(self, output, rank, phase, before, models):
        self.output, self.rank = Path(output), rank
        self.phase, self.before = phase, dict(before)
        self.selected = (
            os.environ.get("QWEN_BOUNDED_PROFILE") == "1"
            and ((phase == "fake_score" and before["fake_updates"] == 201)
                 or (phase == "generator" and before["generator_updates"] == 40))
        )
        self.enabled = self.selected and rank in (0, 8, 16, 24)
        self.handles, self.ranges = [], []
        if not self.enabled:
            return
        self.tag = f"{phase}-fake{before['fake_updates']:06d}-gen{before['generator_updates']:06d}-rank{rank:05d}"
        self.local = Path(tempfile.mkdtemp(prefix="qwen-profile-" + self.tag + "-"))
        self.tag += "-" + self.local.name.rsplit("-", 1)[-1]
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False,
        )
        self.profiler.start()
        self.started = time.monotonic()
        for name, model in models.items():
            if model is None:
                continue
            stack = []

            def enter(module, args, label=name, stack=stack):
                scope = torch.profiler.record_function(f"model/{label}/forward")
                scope.__enter__()
                stack.append(scope)

            def leave(module, args, result, stack=stack):
                if stack:
                    stack.pop().__exit__(None, None, None)

            self.handles.append(model.register_forward_pre_hook(enter))
            self.handles.append(model.register_forward_hook(leave, always_call=True))

    def finish(self, log):
        if not self.enabled:
            return
        for handle in self.handles:
            handle.remove()
        self.profiler.stop()
        stopped = time.monotonic()
        trace = self.local / "trace.json"
        self.profiler.export_chrome_trace(str(trace))
        events = []
        for event in self.profiler.key_averages():
            events.append({
                "operator": event.key, "count": event.count,
                "cpu_total_us": event.cpu_time_total,
                "self_cpu_us": event.self_cpu_time_total,
                "device_total_us": event.device_time_total,
                "self_device_us": event.self_device_time_total,
            })
        target = self.output / "profiling" / self.tag
        target.mkdir(parents=True, exist_ok=False)
        with trace.open("rb") as source, gzip.open(target / "trace.json.gz", "wb", compresslevel=1) as dest:
            shutil.copyfileobj(source, dest)
        (target / "operators.json").write_text(json.dumps(events, indent=2))
        (target / "metadata.json").write_text(json.dumps({
            "rank": self.rank, "phase": self.phase, "before": self.before,
            "training_log": log,
            "profile_wall_seconds": stopped - self.started,
            "export_seconds": time.monotonic() - stopped,
            "warning": "Kernel sums include overlap; instrumented step is not a baseline throughput measurement.",
            "record_shapes": False, "profile_memory": False, "with_stack": False,
        }, indent=2))
        (target / "COMPLETE").write_text("ok\n")
        trace.unlink()
