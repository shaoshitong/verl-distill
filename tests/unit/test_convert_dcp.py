import torch
import torch.distributed.checkpoint as dcp

from verl_distill.tools.convert_dcp import convert_dcp_component


def test_convert_dcp_component_reads_only_named_state(tmp_path):
    checkpoint = tmp_path / "fsdp_state"
    source = {
        "teacher_discriminator_model": {
            "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "bias": torch.tensor([1.0, 2.0]),
        },
        "optimizer": {"step": torch.tensor(9)},
    }
    dcp.save(source, checkpoint_id=str(checkpoint))
    output = tmp_path / "head.pt"
    result = convert_dcp_component(checkpoint, output)
    converted = torch.load(output, weights_only=True)
    assert result["tensor_count"] == 2
    assert set(converted) == {"weight", "bias"}
    torch.testing.assert_close(converted["weight"], source["teacher_discriminator_model"]["weight"])
