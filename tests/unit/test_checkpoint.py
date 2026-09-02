import random

import numpy as np
import torch

from verl_distill.engine.checkpoint import load_training_state, save_training_state


def test_checkpoint_restores_rng_state(tmp_path):
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    path = tmp_path / "state.pt"
    save_training_state(path, {"step": 12})
    expected = (random.random(), np.random.rand(), torch.rand(()))

    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    state = load_training_state(path)
    actual = (random.random(), np.random.rand(), torch.rand(()))

    assert state == {"step": 12}
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
