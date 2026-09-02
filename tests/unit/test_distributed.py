import torch

from verl_distill.engine.distributed import initialize_distributed


def test_single_process_context(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    context = initialize_distributed()
    assert context.rank == 0
    assert context.world_size == 1
    assert context.is_main_process
    assert context.device.type in {"cpu", "cuda"}
    if context.device.type == "cuda":
        assert context.device == torch.device("cuda", 0)
