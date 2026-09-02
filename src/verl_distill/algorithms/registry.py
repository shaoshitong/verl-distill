from collections.abc import Mapping
from typing import Any

from verl_distill.algorithms.dmd import FullModelDMD, StandardDMD
from verl_distill.algorithms.meanflow import ZImageMeanFlow
from verl_distill.algorithms.opd_gan import DualDistilledDiscriminatorOPD

ALGORITHMS = {
    "dmd": StandardDMD,
    "dmd_full": FullModelDMD,
    "meanflow": ZImageMeanFlow,
    "opd_gan": DualDistilledDiscriminatorOPD,
}


def build_algorithm(name: str, config: Mapping[str, Any]):
    try:
        algorithm_type = ALGORITHMS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unknown algorithm '{name}'. Expected one of: {choices}") from exc
    return algorithm_type(**dict(config))
