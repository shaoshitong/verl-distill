"""Score-time diagnostics, with explicit per-metric counts for distributed reduction."""
import torch

SCORE_BIN_EDGES = (0., .1, .2, .4, .6, .8, .9, 1.)
SCORE_BIN_METRICS = ("velocity_mse", "x0_mse", "epsilon_mse", "loss_weighted", "direction_rms")


def score_bin_sums(rows, device="cpu"):
    # columns: sample count, then sum/count for each diagnostic (not all phases provide all).
    values = torch.zeros(len(SCORE_BIN_EDGES) - 1, 1 + 2 * len(SCORE_BIN_METRICS),
                         dtype=torch.float64)
    for row in rows:
        sigma = row["score_sigma"]
        if not 0 <= sigma <= 1:
            raise ValueError("Score sigma outside [0,1]")
        index = min(sum(sigma >= edge for edge in SCORE_BIN_EDGES[1:]), len(values) - 1)
        values[index, 0] += 1
        for metric_index, name in enumerate(SCORE_BIN_METRICS):
            if name in row:
                values[index, 1 + 2 * metric_index] += row[name]
                values[index, 2 + 2 * metric_index] += 1
    return values.to(device)


def score_bin_report(values):
    values = values.cpu().tolist()
    return [{"sigma_min": SCORE_BIN_EDGES[i], "sigma_max": SCORE_BIN_EDGES[i + 1],
             "count": int(row[0]),
             "means": {name: row[1 + 2 * j] / row[2 + 2 * j] if row[2 + 2 * j] else None
                       for j, name in enumerate(SCORE_BIN_METRICS)}}
            for i, row in enumerate(values)]
