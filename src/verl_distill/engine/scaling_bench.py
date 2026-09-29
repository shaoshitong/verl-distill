"""Explicit, isolated scaling-benchmark controls; no training defaults change."""
import os


def settings(config, environ=None):
    env = os.environ if environ is None else environ
    if env.get("QWEN_SCALING_BENCH") != "1":
        return None
    runtime, params = config["runtime"], config["method"]["params"]
    if runtime.get("resume_from") or runtime.get("init_reflow_from"):
        raise ValueError("Scaling benchmark requires fresh HF initialization")
    if int(params["reflow_updates"]) != 0 or params.get("fake_initialization") != "hf":
        raise ValueError("Scaling benchmark requires REFLOW=0 and Fake initialization=hf")
    if int(runtime["gradient_accumulation_steps"]) != 1 or int(runtime["micro_batch_size"]) != 1:
        raise ValueError("Scaling benchmark requires GA=1 and microbatch=1")
    if env.get("QWEN_BOUNDED_PROFILE") == "1":
        raise ValueError("Disable bounded profiler for scaling benchmark")
    index = int(env["QWEN_BENCH_INDEX"])
    if index < 0:
        raise ValueError("Benchmark sample index must be nonnegative")
    return {"sample_index": index, "exit_index": 3, "seed": int(runtime.get("seed", 42))}


def update_seed(bench, phase, before):
    # Deliberately independent of global/local rank and world size.
    phase_id = {"fake_score": 1, "generator": 2}[phase]
    return (bench["seed"] + 1000003 * phase_id + 1009 * before["fake_updates"]
            + 9176 * before["generator_updates"]) % (2**32)


def completion_name(bench):
    return "BENCHMARK_COMPLETE.json" if bench is not None else "TRAINING_COMPLETE.json"
