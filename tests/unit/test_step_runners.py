from types import SimpleNamespace

import torch

from verl_distill.engine.steps import DMDStepRunner, opd_phase_for_step


class _Method:
    def __init__(self):
        self.steps = []

    def set_train_step(self, step):
        self.steps.append(("set", step))

    def score_loss(self, **kwargs):
        del kwargs
        self.steps.append(("score", None))
        return self.score.square(), {"score": 1}

    def generator_loss(self, **kwargs):
        del kwargs
        self.steps.append(("generator", None))
        return self.generator.square(), {"generator": 1}

    def should_update_generator(self, step):
        return step % 3 == 2


def test_dmd_runner_preserves_score_then_generator_order():
    method = _Method()
    method.score = torch.nn.Parameter(torch.tensor(2.0))
    method.generator = torch.nn.Parameter(torch.tensor(3.0))
    score_model = torch.nn.Linear(1, 1)
    generator_model = torch.nn.Linear(1, 1)
    score_optimizer = torch.optim.SGD([method.score], lr=0.1)
    generator_optimizer = torch.optim.SGD([method.generator], lr=0.1)
    runner = DMDStepRunner(method, generator_optimizer, score_optimizer)
    runner.run(
        step=2,
        generator_model=generator_model,
        score_model=score_model,
        x_real=torch.zeros(1, 1),
        c=[],
        e=[],
    )
    assert method.steps == [("set", 2), ("score", None), ("generator", None)]


def test_opd_phase_schedule_preserves_three_discriminator_one_generator_cycle():
    method = SimpleNamespace(
        discriminator_update_ratio=3,
        phase_schedule_switch_step=None,
        phase_schedule_after_pattern=None,
        _any_generator_objective_enabled=lambda: False,
        _discriminator_adv_enabled=lambda: True,
    )
    assert [opd_phase_for_step(method, i) for i in range(8)] == [
        "discriminator",
        "discriminator",
        "discriminator",
        "generator",
        "discriminator",
        "discriminator",
        "discriminator",
        "generator",
    ]
