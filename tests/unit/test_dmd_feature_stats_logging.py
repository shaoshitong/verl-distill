import csv
from types import SimpleNamespace

from verl_distill.trainers.dmd import _append_stats_row, _tracked_stats_keys


def test_fake_first_log_retains_later_generator_feature_columns(tmp_path):
    # Include a layer outside the old fixed list to cover arbitrary selections.
    representations = ("latent", "layer_1", "layer_2", "layer_3", "layer_4", "layer_5", "layer_7")
    method = SimpleNamespace(teacher_feature_representation_keys=lambda: representations)
    path = tmp_path / "stats.tsv"
    fake = {"step": 1001, "score/teacher_feature_loss_layer_1": 0.125}
    gen = {"step": 1005}
    for i, rep in enumerate(representations, 1):
        gen[f"gen/teacher_feature_loss_{rep}"] = float(i)
        gen[f"gen/teacher_feature_denom_{rep}"] = i / 10
    for row in (fake, gen):
        for key in _tracked_stats_keys(method):
            row.setdefault(key, 0.0)
        _append_stats_row(path, row)
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert float(rows[0]["score/teacher_feature_loss_layer_1"]) == 0.125
    for i, rep in enumerate(representations, 1):
        assert float(rows[1][f"gen/teacher_feature_loss_{rep}"]) == i
        assert float(rows[1][f"gen/teacher_feature_denom_{rep}"]) == i / 10
