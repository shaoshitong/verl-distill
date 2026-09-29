import argparse
import csv
import json

import pytest

from verl_distill.tools import prepare_qwen_image21


@pytest.mark.parametrize("kind", ["t2i", "edit"])
def test_snapshot_accepts_a_single_task_directory(tmp_path, monkeypatch, kind):
    root = tmp_path / "production"
    sample = root / kind / "aa" / "sample"
    sample.mkdir(parents=True)
    (sample / "complete.json").write_text("{}")
    (sample.parent / "unfinished").mkdir()
    csv_path = tmp_path / "prompts.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prompt_id", "prompt", "width", "height"])
        writer.writerows((str(i), f"evaluation {i}", 32, 32) for i in range(64))
    record = {
        "id": "sample",
        "kind": kind,
        "prompt": "training sample",
        "height": 32,
        "width": 32,
        "reference_images": [],
    }
    checked = []

    def validate(path, output_root, reference_root, *, verify_hash):
        assert path == sample / "complete.json"
        assert output_root == root and verify_hash
        checked.append(path)
        return record

    monkeypatch.setattr(prepare_qwen_image21, "validate_complete", validate)
    output = tmp_path / "snapshot"
    prepare_qwen_image21.snapshot(
        argparse.Namespace(
            snapshot_dir=str(output),
            output_root=str(root),
            reference_root=str(tmp_path),
            prompts_csv=str(csv_path),
            fast_index=False,
        )
    )
    manifest = json.loads((output / "train.json").read_text())
    assert manifest["records"] == [record]
    assert manifest["payloads_verified"] is True
    assert len(checked) == 1
