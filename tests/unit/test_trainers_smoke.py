from types import SimpleNamespace

import torch
from torch import nn

import verl_distill.trainers.dmd as dmd_trainer
import verl_distill.trainers.meanflow as meanflow_trainer
import verl_distill.trainers.opd_gan as opd_trainer


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
