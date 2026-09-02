import pytest

from verl_distill.algorithms import build_algorithm
from verl_distill.algorithms.dmd import StandardDMD
from verl_distill.algorithms.meanflow import ZImageMeanFlow
from verl_distill.algorithms.opd_gan import DualDistilledDiscriminatorOPD


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("dmd", StandardDMD),
        ("meanflow", ZImageMeanFlow),
        ("opd_gan", DualDistilledDiscriminatorOPD),
    ],
)
def test_build_algorithm(name, expected):
    assert isinstance(build_algorithm(name, {}), expected)


def test_build_algorithm_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        build_algorithm("unknown", {})
