"""Run with torchrun --standalone --nproc-per-node=2 and PYTHONPATH=src."""

import os
import runpy
from pathlib import Path

import torch
import torch.distributed as dist

from verl_distill.algorithms.dmd import StandardDMD
from verl_distill.algorithms.dmd.full_model import FullModelDMD
from verl_distill.engine.fsdp1 import apply_zimage_fsdp1
from verl_distill.models.zimage.feature_discriminator import TeacherFeatureDiscriminator
from verl_distill.trainers.dmd import _gradient_sync_context, _temporarily_requires_grad


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        tiny_teacher = runpy.run_path(
            str(Path(__file__).parent / "unit/test_teacher_feature_discriminator.py")
        )["tiny_teacher"]
        teacher = tiny_teacher().cuda(rank)
        teacher.enable_gradient_checkpointing()
        teacher = apply_zimage_fsdp1(
            teacher, no_split_modules=("ZImageTransformerBlock",), local_rank=rank
        )
        disc = TeacherFeatureDiscriminator(
            64, (1, 2, 3), transformer_layers=2, transformer_heads=4
        ).cuda(rank)
        disc = apply_zimage_fsdp1(
            disc, no_split_modules=("TransformerEncoderLayer",), local_rank=rank
        )
        parameters = list(disc.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        method = StandardDMD(generator_objective="gan", gan_r1_weight=1.0)
        c = [torch.randn(1, 4, 32, device=rank), torch.ones(1, 4, device=rank)]
        before = [p.detach().clone() for p in parameters]
        for i in range(4):
            with _gradient_sync_context(disc, i == 3):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x = torch.randn(1, 4, 8, 8, device=rank)
                    fake, hf = method._call_discriminator(
                        disc, x, score_model={"real": teacher}, c=c, return_features=True
                    )
                    real, hr = method._call_discriminator(
                        disc, x + 0.2, score_model={"real": teacher}, c=c, return_features=True
                    )
                    _, hpr = method._call_discriminator(
                        disc,
                        x + 0.2 + 0.01 * torch.randn_like(x),
                        score_model={"real": teacher},
                        c=c,
                        return_features=True,
                    )
                    _, hpf = method._call_discriminator(
                        disc,
                        x + 0.01 * torch.randn_like(x),
                        score_model={"real": teacher},
                        c=c,
                        return_features=True,
                    )
                    r1 = method._gan_feature_r1(hr, hpr, hf, hpf)
                    assert torch.isfinite(r1) and r1 > 0
                    loss = (
                        torch.nn.functional.softplus(fake).mean()
                        + torch.nn.functional.softplus(-real).mean()
                        + r1
                    ) / 4
                loss.backward()
        assert torch.isfinite(disc.clip_grad_norm_(1.0))
        optimizer.step()
        assert any(not torch.equal(a, p) for a, p in zip(before, parameters))
        optimizer.zero_grad(set_to_none=True)
        with _temporarily_requires_grad(parameters, False):
            x = torch.randn(1, 4, 8, 8, device=rank, requires_grad=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.softplus(
                    -method._call_discriminator(disc, x, score_model={"real": teacher}, c=c)
                ).mean()
            loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        assert all(p.grad is None for p in teacher.parameters())
        assert all(p.grad is None for p in parameters)
        with _temporarily_requires_grad(parameters, False):
            x = torch.randn(1, 4, 8, 8, device=rank, requires_grad=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    _, fake_h = method._call_discriminator(
                        disc, x.detach(), score_model={"real": teacher}, c=c, return_features=True
                    )
                    _, real_h = method._call_discriminator(
                        disc,
                        x.detach() + 0.2,
                        score_model={"real": teacher},
                        c=c,
                        return_features=True,
                    )
                _, h = method._call_discriminator(
                    disc, 0.5 * x, score_model={"real": teacher}, c=c, return_features=True
                )
                feature_loss = method._gan_feature_ste_loss(h, real_h, fake_h)
            assert feature_loss.dtype == torch.float64 and torch.isfinite(feature_loss)
            feature_loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        assert all(p.grad is None for p in teacher.parameters())
        assert all(p.grad is None for p in parameters)
        print(
            f"rank={rank} PASS FSDP1 frozen-D feature STE; loss={feature_loss.item():.6f}",
            flush=True,
        )
        method = StandardDMD(
            generator_objective="teacher_feature_ste", teacher_feature_layers=[1, 2, 3]
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = torch.randn(1, 4, 8, 8, device=rank, requires_grad=True)
            with torch.no_grad():
                real_features = method._call_teacher_features(teacher, x.detach() + 0.2, c)
                fake_features = method._call_teacher_features(teacher, x.detach() - 0.2, c)
            features = method._call_teacher_features(teacher, x, c)
            loss = sum(
                method._teacher_feature_ste_losses(features, real_features, fake_features).values()
            )
        loss.backward()
        assert (
            torch.isfinite(loss)
            and x.grad is not None
            and torch.isfinite(x.grad).all()
            and x.grad.abs().sum() > 0
        )
        assert all(p.grad is None for p in teacher.parameters())
        print(
            f"rank={rank} PASS FSDP1 four-layer raw teacher feature STE input gradient; loss={loss.item():.6f}",
            flush=True,
        )
        method = FullModelDMD(
            generator_objective="teacher_feature_ste",
            score_objective="teacher_feature_mse",
            teacher_feature_layers=[1, 2, 3],
            teacher_feature_include_last=False,
            teacher_feature_include_latent=True,
            teacher_feature_normalize=True,
            real_guidance_scale=0.0,
        )
        trainable = []
        for _ in range(2):
            model = tiny_teacher().cuda(rank).train().requires_grad_(True)
            model.enable_gradient_checkpointing()
            trainable.append(
                apply_zimage_fsdp1(
                    model, no_split_modules=("ZImageTransformerBlock",), local_rank=rank
                )
            )
        generator, fake_model = trainable
        gopt = torch.optim.AdamW(generator.parameters(), lr=1e-4)
        fopt = torch.optim.AdamW(fake_model.parameters(), lr=1e-4)
        score = {"real": teacher, "fake": fake_model}
        fake_before = [p.detach().clone() for p in fake_model.parameters()]
        for _ in range(5):
            fopt.zero_grad(set_to_none=True)
            for i in range(4):
                with _gradient_sync_context(fake_model, i == 3):
                    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                        x = torch.randn(1, 4, 8, 8, device=rank)
                        loss, _ = method.score_loss(generator, score, x, c, None)
                    (loss / 4).backward()
            assert torch.isfinite(loss)
            assert torch.isfinite(fake_model.clip_grad_norm_(1.0))
            fopt.step()
            assert all(p.grad is None for p in generator.parameters())
            assert all(p.grad is None for p in teacher.parameters())
        assert any(not torch.equal(a, p) for a, p in zip(fake_before, fake_model.parameters()))
        fopt.zero_grad(set_to_none=True)
        g_before = [p.detach().clone() for p in generator.parameters()]
        for i in range(4):
            with _gradient_sync_context(generator, i == 3):
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    x = torch.randn(1, 4, 8, 8, device=rank)
                    loss, _ = method.generator_loss(generator, score, x, c, None)
                (loss / 4).backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(generator.clip_grad_norm_(2.83))
        gopt.step()
        assert any(not torch.equal(a, p) for a, p in zip(g_before, generator.parameters()))
        assert all(p.grad is None for p in fake_model.parameters())
        assert all(p.grad is None for p in teacher.parameters())
        print(
            f"rank={rank} PASS FSDP1 GC GA4 latent+3layer fake MSE x5 -> normalized G STE; G loss={loss.item():.6f}",
            flush=True,
        )
        print(
            f"rank={rank} PASS FSDP1 GA4 R1 D update and frozen-teacher GC input gradient; R1={r1.item():.6f}",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
