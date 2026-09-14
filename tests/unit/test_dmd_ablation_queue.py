import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from verl_distill.algorithms.dmd.full_model import FullModelDMD
from verl_distill.config import load_config

spec = importlib.util.spec_from_file_location(
    "ablation_queue", Path(__file__).parents[2] / "scripts/run_dmd_ablation_queue.py"
)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


@pytest.mark.parametrize(
    "name,layers,latent",
    [
        ("0_5", [5], True),
        ("0_5_15", [5, 15], True),
        ("0_5_15_25", [5, 15, 25], True),
        ("25", [25], False),
    ],
)
def test_matched_ablation_config(monkeypatch, name, layers, latent):
    for key in (
        "ZIMAGE_MODEL_PATH",
        "ZIMAGE_TEACHER_MODEL_PATH",
        "ZIMAGE_FAKE_SCORE_MODEL_PATH",
        "TRAIN_LANCE_DATA_DIR",
    ):
        monkeypatch.setenv(key, "/placeholder")
    cfg = load_config(f"configs/recipes/zimage/dmd_teacher_ablation_{name}.yaml")
    method = FullModelDMD(**cfg["method"]["params"])
    assert method.teacher_feature_layers == tuple(layers)
    assert method.teacher_feature_include_latent == latent
    assert not method.teacher_feature_include_last and method.teacher_feature_normalize
    expected = {f"layer_{layer}": 1.0 for layer in layers}
    if latent:
        expected["latent"] = 1.0
    assert (
        method.generator_teacher_feature_weights == method.score_teacher_feature_weights == expected
    )
    assert method.score_objective == "teacher_feature_mse" and method.score_loss_target == "x0"
    assert not method.score_use_weighting
    assert cfg["runtime"]["max_train_steps"] == 2500
    assert cfg["runtime"]["gradient_accumulation_steps"] == 4
    assert cfg["optimizer"]["generator"]["lr"] == 5e-5
    assert cfg["optimizer"]["generator"]["max_grad_norm"] == 2.83
    assert cfg["optimizer"]["fake_score"]["lr"] == 1e-5


def manifest_fixture(root, fail=False):
    jobs = []
    for name in ("A", "B"):
        out = root / name
        out.mkdir()
        checkpoint = out / "checkpoints/step-1006"
        train = (
            "import csv; from pathlib import Path; "
            f"p=Path({str(out)!r}); c=Path({str(checkpoint)!r}); c.mkdir(parents=True); (c/'.metadata').touch(); "
            "rows=[dict(step=1001+i,generator_updated=int(i==5),score_updated=int(i<5),"
            "discriminator_updated=0,accumulation=4,score_nonfinite_grads=0,generator_nonfinite_grads=0,"
            "score_loss=1.,generator_loss=2.) for i in range(6)]; "
            "f=(p/'dmd_stats.tsv').open('w'); w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t'); "
            "w.writeheader(); w.writerows(rows); f.close()"
        )
        evaluation = f"from pathlib import Path; Path({str(out / 'case_000001_seed0.png')!r}).write_bytes(b'fixture')"
        jobs.append(
            dict(
                name=name,
                output_dir=str(out),
                start_step=1000,
                end_step=1006,
                train_command=[
                    sys.executable,
                    "-c",
                    "raise SystemExit(3)" if fail and name == "A" else train,
                ],
                eval_command=[sys.executable, "-c", evaluation],
                eval_images_dir=str(out),
                expected_images=1,
            )
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(dict(cwd=str(root), jobs=jobs, sha256={})))
    return manifest


def test_queue_runs_serially_and_checks_final_artifacts(tmp_path):
    manifest = manifest_fixture(tmp_path)
    queue.run_queue(manifest)
    state = json.loads((tmp_path / "queue_state.json").read_text())
    assert state["status"] == "complete"
    assert [job["status"] for job in state["jobs"]] == ["complete", "complete"]
    assert all(job["training_summary"]["fake"]["updates"] == 5 for job in state["jobs"])
    assert state["jobs"][0]["finished_at"] <= state["jobs"][1]["stage_started_at"]
    with pytest.raises(RuntimeError, match="already exists"):
        queue.run_queue(manifest)


def test_queue_stops_on_failure_without_starting_next_job(tmp_path):
    queue_path = manifest_fixture(tmp_path, fail=True)
    with pytest.raises(RuntimeError, match="exited with code 3"):
        queue.run_queue(queue_path)
    state = json.loads((tmp_path / "queue_state.json").read_text())
    assert state["status"] == "failed"
    assert state["jobs"][1]["status"] == "pending"
    assert not (tmp_path / "B/train.log").exists()


def test_queue_rejects_changed_inputs(tmp_path):
    p = tmp_path / "input.yaml"
    p.write_text("original")
    manifest = {"sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()}}
    queue.verify_inputs(manifest)
    p.write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        queue.verify_inputs(manifest)
