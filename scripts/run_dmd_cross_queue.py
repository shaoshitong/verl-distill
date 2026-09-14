#!/usr/bin/env python3
"""Run the two approved cross-ablation jobs, then occupy GPUs."""

import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = ("A_fake0_gen0to5", "B_fake0to5_gen0")
PYTHON = sys.executable
child = None
stopped = False
state = {
    "pid": os.getpid(),
    "status": "starting",
    "jobs": {name: {"status": "pending"} for name in JOBS},
}


def save():
    state["updated_at"] = time.time()
    temp = ROOT / "queue_state.tmp"
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(ROOT / "queue_state.json")


def stop(signum, frame):
    global stopped
    stopped = True
    if child is not None and child.poll() is None:
        child.terminate()


def verify_inputs():
    for path, digest in json.loads((ROOT / "input_hashes.json").read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Input changed before next job: " + path)


def verify_training(name):
    folder = ROOT / name
    with (folder / "dmd_stats.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [int(row["step"]) for row in rows] == list(range(1001, 2501)), "Incomplete step sequence"
    for row in rows:
        step = int(row["step"])
        gen = int(step % 5 == 0)
        assert int(row["generator_updated"]) == gen and int(row["score_updated"]) == 1 - gen
        assert int(row["accumulation"]) == 4
        assert int(row["generator_nonfinite_grads"]) == int(row["score_nonfinite_grads"]) == 0
        assert math.isfinite(float(row["generator_loss"])) and math.isfinite(
            float(row["score_loss"])
        )
        if gen:
            selected = (
                ("latent",)
                if name.startswith("B")
                else ("latent", "layer_1", "layer_2", "layer_3", "layer_4", "layer_5")
            )
            total = sum(float(row["gen/teacher_feature_loss_" + key]) for key in selected)
            assert math.isclose(total, float(row["generator_loss"]), rel_tol=1e-5, abs_tol=1e-6)
    checkpoint = folder / "checkpoints/step-2500"
    assert (checkpoint / ".metadata").is_file(), "Final checkpoint missing"
    return {
        "checkpoint": str(checkpoint),
        "fake_updates": 1200,
        "gen_updates": 300,
        "last_step": 2500,
    }


def execute(command, log_path):
    global child
    if stopped:
        raise InterruptedError("Queue stopped")
    with log_path.open("ab") as log:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state["child_pid"] = child.pid
        save()
        stop_since = None
        while child.poll() is None:
            if stopped:
                stop_since = time.time() if stop_since is None else stop_since
                if time.time() - stop_since > 60:
                    os.killpg(child.pid, signal.SIGKILL)
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        code = child.returncode
    child = None
    state.pop("child_pid", None)
    if stopped:
        raise InterruptedError("Queue stopped")
    return code


if __name__ == "__main__":
    with (ROOT / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / "queue_state.json").exists():
            raise RuntimeError("Queue already has state; refusing duplicate launch")
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        (ROOT / "queue.pid").write_text(str(os.getpid()) + "\n")
        save()
        try:
            for name in JOBS:
                verify_inputs()
                if (ROOT / name / "dmd_stats.tsv").exists():
                    raise RuntimeError("Existing training output: " + name)
                gpu_pids = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
                )
                if gpu_pids.strip():
                    raise RuntimeError("GPUs occupied before " + name + ": " + gpu_pids.strip())
                state.update(status="training", active_job=name)
                state["jobs"][name].update(status="training", started_at=time.time())
                save()
                print("Starting " + name, flush=True)
                code = execute(["bash", str(ROOT / name / "train.sh")], ROOT / name / "train.log")
                (ROOT / name / "train.exit_code").write_text(str(code) + "\n")
                if code:
                    raise RuntimeError(name + " training failed: " + str(code))
                summary = verify_training(name)
                state["jobs"][name].update(status="complete", finished_at=time.time(), **summary)
                save()
                print("Verified complete: " + name, flush=True)
            settings = json.loads((ROOT / "queue_settings.json").read_text())
            occupier = settings.get("gpu_occupier_script")
            if not occupier:
                state.update(status="complete", active_job=None, training_completed_at=time.time())
                save()
                sys.exit(0)
            state.update(status="occupying", active_job=None, training_completed_at=time.time())
            save()
            print("Both jobs complete; starting GPU occupier", flush=True)
            code = execute([PYTHON, "-u", occupier], ROOT / "occupy.log")
            state.update(status="occupier_exited", occupier_exit_code=code)
            save()
        except Exception as exc:
            state.update(status="stopped" if stopped else "failed", error=str(exc))
            active = state.get("active_job")
            if active:
                state["jobs"][active]["status"] = state["status"]
            save()
            raise
