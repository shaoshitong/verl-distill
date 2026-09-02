import torch

from verl_distill.engine.ema import ShardedEMA, use_ema_weights


def test_ema_update_and_temporary_shadow():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(1.0)
    ema = ShardedEMA(model, decay=0.5)
    model.weight.data.fill_(3.0)
    ema.update(model)
    with use_ema_weights(ema, model):
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 2.0))
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 3.0))


def test_ema_state_dict_round_trip():
    source = torch.nn.Linear(2, 2, bias=False)
    target = torch.nn.Linear(2, 2, bias=False)
    ema = ShardedEMA(source, decay=0.9)
    source.weight.data.fill_(3.0)
    ema.update(source)

    restored = ShardedEMA(target, decay=0.1)
    restored.load_state_dict(ema.state_dict())

    assert restored.decay == 0.9
    torch.testing.assert_close(restored.ema_params["weight"], ema.ema_params["weight"])


def test_ema_rebinds_equivalent_parameter_objects():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(3.0)
    ema = ShardedEMA(model, decay=0.5)
    ema.ema_params["weight"].fill_(2.0)

    with use_ema_weights(ema, model):
        replacement = torch.nn.Parameter(model.weight.detach().clone())
        model.weight = replacement
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 2.0))

    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 3.0))
