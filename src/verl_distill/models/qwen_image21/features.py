"""Frozen-teacher image-token features with input gradients for TDM REFLOW."""
import types
import torch
import torch.nn.functional as F


def install_feature_forward(model):
    original = model.forward

    def forward(self, *args, return_features=False, **kwargs):
        if not return_features:
            return original(*args, **kwargs)
        captured = []
        # Hook the final block, before norm_out/proj_out (including unpatch).
        # Return through the root forward so FSDP registers input-gradient backward.
        def capture(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError("Expected Qwen21 block tensor output")
            captured.append(output)
        handle = self.transformer_blocks[-1].register_forward_hook(capture)
        try:
            original(*args, **kwargs)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"Expected one final-block output, got {len(captured)}")
        return (captured[0],)

    model.forward = types.MethodType(forward, model)
    return model


def predict_features(model, target_latents, sigma, condition):
    hidden = target_latents.to(torch.bfloat16)
    reference = condition['reference_latents']
    if reference is not None:
        hidden = torch.cat([reference.to(hidden), hidden], dim=1)
    kwargs = {k: v for k, v in condition.items() if k != 'reference_latents'}
    features = model(hidden_states=hidden, timestep=sigma.float().reshape(-1),
                     **kwargs, return_dict=False, return_features=True)[0]
    # Qwen21 appends target image tokens last; text/reference tokens are excluded.
    return features[:, -target_latents.shape[1]:]


def feature_cosine_loss(predicted, target):
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError('Expected matching [batch,target_tokens,channels] features')
    return (1.0 - F.cosine_similarity(predicted.float(), target.detach().float(), dim=-1, eps=1e-8)).mean()
