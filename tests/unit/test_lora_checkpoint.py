import torch

from verl_distill.algorithms.dmd.lora import (
    load_lora_state_dict,
    lora_state_dict,
    replace_linear_with_lora,
)


def test_lora_state_round_trip():
    source = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    target = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    replace_linear_with_lora(source, rank=2)
    replace_linear_with_lora(target, rank=2)
    for parameter in source.parameters():
        parameter.data.uniform_(-0.5, 0.5)
    result = load_lora_state_dict(target, lora_state_dict(source))
    assert result == {"missing": [], "unexpected": []}
    for key, value in lora_state_dict(source).items():
        torch.testing.assert_close(lora_state_dict(target)[key], value)
