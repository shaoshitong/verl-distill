#!/usr/bin/env python3
"""Run an explicit, immutable DMD experiment manifest sequentially; fail closed."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_inputs(manifest):
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Experiment input changed after queue creation: {name}")


def training_summary(job):
    rows = list(csv.DictReader((Path(job["output_dir"]) / "dmd_stats.tsv").open(), delimiter="\t"))
    expected = list(range(job["start_step"] + 1, job["end_step"] + 1))
    if [int(row["step"]) for row in rows] != expected:
        raise RuntimeError(f"Incomplete or duplicated training steps: {job['name']}")
    checkpoint = Path(job["output_dir"]) / "checkpoints" / f"step-{job['end_step']}"
    if not (checkpoint / ".metadata").is_file():
        raise RuntimeError(f"Final checkpoint missing: {checkpoint}")
    for i, row in enumerate(rows):
        g = int(i % 6 == 5)
        if (
            int(row["generator_updated"]) != g
            or int(row["score_updated"]) != 1 - g
            or int(row["discriminator_updated"])
            or int(row["accumulation"]) != 4
        ):
            raise RuntimeError(f"Unexpected update schedule at step {row['step']}")
        if any(float(row[key]) for key in ("score_nonfinite_grads", "generator_nonfinite_grads")):
            raise RuntimeError(f"Nonfinite gradients recorded at step {row['step']}")
    result = {"name": job["name"], "checkpoint": str(checkpoint)}
    for phase, flag, loss_key in (
        ("fake", "score_updated", "score_loss"),
        ("generator", "generator_updated", "generator_loss"),
    ):
        selected = [r for r in rows if int(r[flag])]
        tail = [r for r in selected if int(r["step"]) > job["end_step"] - 100]
        result[phase] = {
            "updates": len(selected),
            "last_100_steps_loss_mean": sum(float(r[loss_key]) for r in tail) / len(tail),
        }
    return result


def run_queue(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    lock_name = hashlib.sha256(str(root).encode()).hexdigest()[:24]
    state = {
        "pid": os.getpid(),
        "status": "running",
        "started_at": time.time(),
        "jobs": [
            {"name": job["name"], "output_dir": job["output_dir"], "status": "pending"}
            for job in manifest["jobs"]
        ],
    }
    child = None
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.terminate()

    def execute(command, log_path, job_state, stage):
        nonlocal child
        if stopped:
            raise InterruptedError("Queue stopped")
        verify_inputs(manifest)
        if manifest.get("require_idle_gpus"):
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                raise RuntimeError(
                    "GPUs occupied before next stage; queue will not preempt other jobs"
                )
        job_state.update(stage=stage, status="running", stage_started_at=time.time())
        with Path(log_path).open("ab") as log:
            child = subprocess.Popen(
                command,
                cwd=manifest["cwd"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            job_state["child_pid"] = child.pid
            print(f"{job_state['name']}: {stage}, pid={child.pid}, log={log_path}", flush=True)
            write_json(root / "queue_state.json", state)
            stop_since = None
            while child.poll() is None:
                if stopped:
                    child.terminate()
                    stop_since = time.time() if stop_since is None else stop_since
                    if time.time() - stop_since > 60:
                        os.killpg(child.pid, signal.SIGKILL)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    continue
            code = child.returncode
        child = None
        if stopped:
            raise InterruptedError("Queue stopped")
        if code:
            raise RuntimeError(f"{job_state['name']} {stage} exited with code {code}")

    with (Path(tempfile.gettempdir()) / f"dmd-ablation-{lock_name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "queue_state.json").exists():
            raise RuntimeError("Queue state already exists; refusing an implicit restart")
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in handlers:
            signal.signal(sig, stop)
        try:
            verify_inputs(manifest)
            write_json(root / "queue_state.json", state)
            for job, item in zip(manifest["jobs"], state["jobs"], strict=True):
                output = Path(job["output_dir"])
                if (output / "dmd_stats.tsv").exists():
                    raise RuntimeError(f"Existing training output; refusing to overwrite: {output}")
                execute(job["train_command"], output / "train.log", item, "training")
                item["training_summary"] = training_summary(job)
                write_json(root / "queue_state.json", state)
                execute(job["eval_command"], output / "eval.log", item, "evaluation_images")
                images = list(Path(job["eval_images_dir"]).glob("case_*_seed*.png"))
                if len(images) != job["expected_images"] or any(
                    p.stat().st_size == 0 for p in images
                ):
                    raise RuntimeError(f"Evaluation images incomplete: {job['name']}")
                item.update(status="complete", images=len(images), finished_at=time.time())
                write_json(root / "queue_state.json", state)
                write_json(
                    root / "training_summary.json",
                    [j["training_summary"] for j in state["jobs"] if "training_summary" in j],
                )
            state["status"] = "complete"
        except BaseException as exc:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            state.update(status="stopped" if stopped else "failed", error=str(exc))
            for item in state["jobs"]:
                if item["status"] == "running":
                    item["status"] = state["status"]
            raise
        finally:
            state["updated_at"] = time.time()
            write_json(root / "queue_state.json", state)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    run_queue(parser.parse_args().manifest)
