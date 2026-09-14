from types import SimpleNamespace

import pytest
import torch
from torch import nn

import verl_distill.trainers.dmd as dmd_trainer
import verl_distill.trainers.meanflow as meanflow_trainer
import verl_distill.trainers.opd_gan as opd_trainer
from verl_distill.algorithms.dmd import StandardDMD


class _Loader(list):
    pass


class _Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = _Wrapper()

    def encode_prompt(self, text, do_cfg=True):
        del do_cfg
        batch = len(text)
        embeds = torch.ones(batch, 1, 1)
        mask = torch.ones(batch, 1)
        return embeds, mask, -embeds, mask

    def pixels_to_latents(self, image):
        return image


class _DMDMethod:
    def set_train_step(self, step):
        self.step = step

    def score_loss(self, score_model, **kwargs):
        del kwargs
        score = score_model["fake"] if isinstance(score_model, dict) else score_model
        return score.transformer.weight.square().sum(), {}

    def should_update_generator(self, step):
        return step == 1

    def generator_loss(self, generator_model, **kwargs):
        del kwargs
        return generator_model.transformer.weight.square().sum(), {}


class _DMDGANMethod(_DMDMethod):
    def uses_gan_objective(self):
        return True

    def should_update_generator(self, step):
        del step
        return False

    def discriminator_loss(self, discriminator_model, **kwargs):
        del kwargs
        parameter = next(discriminator_model.parameters())
        return parameter.square().sum(), {}

    def generator_loss(self, generator_model, discriminator_model=None, **kwargs):
        del discriminator_model, kwargs
        return generator_model.transformer.weight.square().sum(), {}


class _MeanFlowMethod:
    def training_step(self, model, x, **kwargs):
        del x, kwargs
        return model.transformer.weight.square().sum(), {}


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Linear(1, 1, bias=False)

    def forward(self, value):
        return self.transformer(value)


class _OPDMethod:
    discriminator_update_ratio = 0
    phase_schedule_switch_step = None
    phase_schedule_after_pattern = None

    def _any_generator_objective_enabled(self):
        return True

    def _discriminator_adv_enabled(self):
        return True

    def training_step(self, student_model, discriminator_model, phase, **kwargs):
        del kwargs
        model = student_model if phase == "generator" else discriminator_model
        parameter = next(model.parameters())
        return parameter.square().sum(), {}


class _Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.dual_projector_multi_feature_discriminator_head = nn.Linear(
            1, 1, bias=False
        )


class _ConstantFlowGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow = nn.Parameter(torch.zeros(()))

    def forward(self, x_t, t, c=None):
        del t, c
        return self.flow.expand_as(x_t)


class _MeanLogitDiscriminator(nn.Module):
    def forward(self, latents):
        return latents.flatten(1).mean(dim=1)


def _context():
    return SimpleNamespace(
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        is_main_process=True,
    )


def _batch():
    return {"text": ["prompt"], "image": torch.ones(1, 1), "z": torch.empty(0)}


def _base_config(name):
    return {
        "distributed": {"backend": "gloo"},
        "runtime": {
            "seed": 42,
            "micro_batch_size": 1,
            "num_workers": 0,
            "max_train_steps": 1,
            "save_every_n_steps": 0,
        },
        "method": {"name": name, "params": {}},
        "data": {},
        "optimizer": {
            "generator": {"lr": 0.1, "max_grad_norm": 1.0},
            "fake_score": {"lr": 0.1, "max_grad_norm": 1.0},
            "discriminator": {"lr": 0.1, "max_grad_norm": 1.0},
        },
    }


def test_dmd_trainer_runs_one_optimizer_step(monkeypatch):
    student = _Student()
    score = {"real": _Wrapper(), "fake": _Wrapper()}
    monkeypatch.setattr(dmd_trainer, "initialize_distributed", lambda backend: _context())
    monkeypatch.setattr(dmd_trainer, "cleanup_distributed", lambda: None)
    monkeypatch.setattr(dmd_trainer, "build_algorithm", lambda name, config: _DMDMethod())
    monkeypatch.setattr(dmd_trainer, "_build_models", lambda config, device: (student, score))
    monkeypatch.setattr(dmd_trainer, "build_dataset", lambda config: object())
    monkeypatch.setattr(
        dmd_trainer, "build_dataloader", lambda *args, **kwargs: _Loader([_batch()])
    )
    before = student.transformer.transformer.weight.detach().clone()
    assert dmd_trainer.train_dmd(_base_config("dmd")) == 1
    assert not torch.equal(before, student.transformer.transformer.weight)


def test_dmd_trainer_runs_one_schedule_free_fake_score_step(monkeypatch):
    student = _Student()
    score = {"real": _Wrapper(), "fake": _Wrapper()}
    monkeypatch.setattr(dmd_trainer, "initialize_distributed", lambda backend: _context())
    monkeypatch.setattr(dmd_trainer, "cleanup_distributed", lambda: None)
    monkeypatch.setattr(dmd_trainer, "build_algorithm", lambda name, config: _DMDMethod())
    monkeypatch.setattr(dmd_trainer, "_build_models", lambda config, device: (student, score))
    monkeypatch.setattr(dmd_trainer, "build_dataset", lambda config: object())
    monkeypatch.setattr(
        dmd_trainer, "build_dataloader", lambda *args, **kwargs: _Loader([_batch()])
    )
    config = _base_config("dmd")
    config["distributed"]["fsdp_backend"] = "fsdp1"
    config["optimizer"]["fake_score"]["type"] = "adamw_schedule_free"
    before = score["fake"].transformer.weight.detach().clone()
    assert dmd_trainer.train_dmd(config) == 1
    assert not torch.equal(before, score["fake"].transformer.weight)


def test_dmd_trainer_runs_one_schedule_free_generator_step(monkeypatch):
    student = _Student()
    score = {"real": _Wrapper(), "fake": _Wrapper()}
    monkeypatch.setattr(dmd_trainer, "initialize_distributed", lambda backend: _context())
    monkeypatch.setattr(dmd_trainer, "cleanup_distributed", lambda: None)
    monkeypatch.setattr(dmd_trainer, "build_algorithm", lambda name, config: _DMDMethod())
    monkeypatch.setattr(dmd_trainer, "_build_models", lambda config, device: (student, score))
    monkeypatch.setattr(dmd_trainer, "build_dataset", lambda config: object())
    monkeypatch.setattr(
        dmd_trainer, "build_dataloader", lambda *args, **kwargs: _Loader([_batch()])
    )
    config = _base_config("dmd")
    config["distributed"]["fsdp_backend"] = "fsdp1"
    config["optimizer"]["generator"]["type"] = "adamw_schedule_free"
    before = student.transformer.transformer.weight.detach().clone()
    assert dmd_trainer.train_dmd(config) == 1
    assert not torch.equal(before, student.transformer.transformer.weight)


@pytest.mark.parametrize("teacher_features", [False, True])
def test_dmd_trainer_runs_gan_optimizer_cycle(monkeypatch, teacher_features):
    student = _Student()
    score = {"real": _Wrapper(), "fake": _Wrapper()}
    discriminator = _Wrapper()
    discriminator.feature_layers = (4, 12, 20)
    monkeypatch.setattr(dmd_trainer, "initialize_distributed", lambda backend: _context())
    monkeypatch.setattr(dmd_trainer, "cleanup_distributed", lambda: None)
    method = _DMDGANMethod()
    method.dfake_gen_update_ratio = 5
    method.warmup_iterations = 0
    monkeypatch.setattr(method, "uses_gan_objective", lambda: not teacher_features)
    monkeypatch.setattr(
        method, "uses_teacher_feature_objective", lambda: teacher_features, raising=False
    )
    calls = []
    for name in ("score_loss", "discriminator_loss", "generator_loss"):
        original = getattr(method, name)

        def tracked_loss(*args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(method, name, tracked_loss)
    monkeypatch.setattr(dmd_trainer, "build_algorithm", lambda name, config: method)
    monkeypatch.setattr(dmd_trainer, "_build_models", lambda config, device: (student, score))

    def build_discriminator(config, student, device):
        assert not teacher_features, (
            "Frozen teacher feature objective must not build a discriminator"
        )
        return discriminator

    monkeypatch.setattr(
        dmd_trainer,
        "_build_feature_discriminator",
        build_discriminator,
    )
    monkeypatch.setattr(dmd_trainer, "build_dataset", lambda config: object())
    monkeypatch.setattr(
        dmd_trainer, "build_dataloader", lambda *args, **kwargs: _Loader([_batch()])
    )
    config = _base_config("dmd")
    cycle = ["fake_score"] * 5 + ([] if teacher_features else ["discriminator"]) + ["generator"]
    config["runtime"]["max_train_steps"] = 2 * len(cycle)
    config["runtime"]["gradient_accumulation_steps"] = 4
    updates = []
    build_optimizer = dmd_trainer._build_optimizer

    def tracked_optimizer(parameters, options, *, role, **kwargs):
        optimizer = build_optimizer(parameters, options, role=role, **kwargs)
        original_step = optimizer.step

        def step(*args, **kwargs):
            updates.append(role)
            return original_step(*args, **kwargs)

        optimizer.step = step
        return optimizer

    monkeypatch.setattr(dmd_trainer, "_build_optimizer", tracked_optimizer)
    before = discriminator.transformer.weight.detach().clone()
    assert dmd_trainer.train_dmd(config) == 2 * len(cycle)
    assert torch.equal(before, discriminator.transformer.weight) == teacher_features
    assert updates == cycle * 2
    assert (
        calls
        == (
            ["score_loss"] * 20
            + ([] if teacher_features else ["discriminator_loss"] * 4)
            + ["generator_loss"] * 4
        )
        * 2
    )


def test_dmd_gan_generator_loss_pushes_fake_logit_up():
    method = StandardDMD(
        generator_objective="gan",
        min_step_percent=0.5,
        max_step_percent=0.5,
        backward_simulation=False,
    )
    generator = _ConstantFlowGenerator()
    discriminator = _MeanLogitDiscriminator()
    x_real = torch.zeros(2, 1, 2, 2)
    c = [torch.ones(2, 1, 1), torch.ones(2, 1)]

    loss, stats = method.generator_loss(
        generator_model=generator,
        score_model={"real": _ConstantFlowGenerator(), "fake": _ConstantFlowGenerator()},
        discriminator_model=discriminator,
        x_real=x_real,
        c=c,
        e=None,
        latent_shape=x_real.shape,
    )
    loss.backward()

    assert generator.flow.grad is not None
    assert generator.flow.grad.item() > 0
    assert "gan_gen_fake_logit" in stats


def test_schedule_free_fake_score_optimizer_requires_fsdp1():
    parameter = nn.Parameter(torch.ones(()))
    config = {"type": "adamw_schedule_free", "lr": 0.1}

    try:
        dmd_trainer._build_optimizer(
            [parameter],
            config,
            role="fake_score",
            fsdp_backend="fsdp2",
        )
    except ValueError as exc:
        assert "requires FSDP1" in str(exc)
    else:
        raise AssertionError("expected schedule-free fake_score optimizer to require FSDP1")


def test_schedule_free_generator_optimizer_requires_fsdp1():
    parameter = nn.Parameter(torch.ones(()))
    config = {"type": "adamw_schedule_free", "lr": 0.1}

    try:
        dmd_trainer._build_optimizer(
            [parameter],
            config,
            role="generator",
            fsdp_backend="fsdp2",
        )
    except ValueError as exc:
        assert "requires FSDP1" in str(exc)
    else:
        raise AssertionError("expected schedule-free generator optimizer to require FSDP1")


def test_schedule_free_optimizer_eval_context_restores_train_mode():
    parameter = nn.Parameter(torch.ones(()))
    optimizer = dmd_trainer._build_optimizer(
        [parameter],
        {"type": "adamw_schedule_free", "lr": 0.1},
        role="fake_score",
        fsdp_backend="fsdp1",
    )

    try:
        optimizer.step()
    except RuntimeError as exc:
        assert "requires optimizer.train" in str(exc)
    else:
        raise AssertionError("expected schedule-free step to require train mode")

    dmd_trainer._optimizer_train(optimizer)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert optimizer.param_groups[0]["train_mode"] is True

    with dmd_trainer._optimizer_eval_context(optimizer):
        assert optimizer.param_groups[0]["train_mode"] is False

    assert optimizer.param_groups[0]["train_mode"] is True


def test_dmd_model_only_resume_skips_optimizer_state(monkeypatch):
    student = _Student()
    score = {"real": _Wrapper(), "fake": _Wrapper()}
    loaded = {}

    def fake_load_state(path, *, map_location):
        del path, map_location
        loaded["called"] = True
        return {
            "step": 0,
            "generator": student.transformer.state_dict(),
            "fake_score": score["fake"].state_dict(),
            "generator_optimizer": {"unexpected": "old"},
            "score_optimizer": {"unexpected": "old"},
        }

    monkeypatch.setattr(dmd_trainer, "initialize_distributed", lambda backend: _context())
    monkeypatch.setattr(dmd_trainer, "cleanup_distributed", lambda: None)
    monkeypatch.setattr(dmd_trainer, "build_algorithm", lambda name, config: _DMDMethod())
    monkeypatch.setattr(dmd_trainer, "_build_models", lambda config, device: (student, score))
    monkeypatch.setattr(dmd_trainer, "build_dataset", lambda config: object())
    monkeypatch.setattr(
        dmd_trainer, "build_dataloader", lambda *args, **kwargs: _Loader([_batch()])
    )
    monkeypatch.setattr(dmd_trainer, "load_training_state", fake_load_state)
    config = _base_config("dmd")
    config["runtime"]["resume_from"] = "/tmp/checkpoint.pt"
    config["runtime"]["resume_optimizer_state"] = False

    assert dmd_trainer.train_dmd(config) == 1
    assert loaded["called"] is True


def test_meanflow_trainer_runs_one_optimizer_step(monkeypatch):
    student = _Student()
    teacher = _Wrapper()
    monkeypatch.setattr(meanflow_trainer, "initialize_distributed", lambda backend: _context())
    monkeypatch.setattr(meanflow_trainer, "cleanup_distributed", lambda: None)
    monkeypatch.setattr(meanflow_trainer, "build_algorithm", lambda name, config: _MeanFlowMethod())
    monkeypatch.setattr(
        meanflow_trainer, "_build_models", lambda config, device: (student, teacher)
    )
    monkeypatch.setattr(meanflow_trainer, "build_dataset", lambda config: object())
    monkeypatch.setattr(
        meanflow_trainer, "build_dataloader", lambda *args, **kwargs: _Loader([_batch()])
    )
    config = _base_config("meanflow")
    config["runtime"]["gradient_accumulation_steps"] = 1
    before = student.transformer.transformer.weight.detach().clone()
    assert meanflow_trainer.train_meanflow(config) == 1
    assert not torch.equal(before, student.transformer.transformer.weight)


def test_opd_gan_trainer_runs_generator_phase(monkeypatch):
    student = _Student()
    teacher = _Wrapper()
    discriminator = _Discriminator()
    monkeypatch.setattr(opd_trainer, "initialize_distributed", lambda backend: _context())
    monkeypatch.setattr(opd_trainer, "cleanup_distributed", lambda: None)
    monkeypatch.setattr(opd_trainer, "build_algorithm", lambda name, config: _OPDMethod())
    monkeypatch.setattr(
        opd_trainer,
        "_build_models",
        lambda config, device: (
            student,
            teacher,
            discriminator,
            teacher,
            list(discriminator.parameters()),
        ),
    )
    monkeypatch.setattr(opd_trainer, "build_dataset", lambda config: object())
    monkeypatch.setattr(
        opd_trainer, "build_dataloader", lambda *args, **kwargs: _Loader([_batch()])
    )
    config = _base_config("opd_gan")
    config["runtime"]["gradient_accumulation_steps"] = 1
    before = student.transformer.transformer.weight.detach().clone()
    assert opd_trainer.train_opd_gan(config) == 1
    assert not torch.equal(before, student.transformer.transformer.weight)


def test_opd_cosine_scheduler_preserves_recipe_parameters():
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=5e-6)
    scheduler = opd_trainer._scheduler(
        optimizer,
        {"lr_scheduler": "cosine", "lr_decay_steps": 3000, "lr_min": 1e-6},
    )

    assert scheduler.T_max == 3000
    assert scheduler.eta_min == 1e-6
