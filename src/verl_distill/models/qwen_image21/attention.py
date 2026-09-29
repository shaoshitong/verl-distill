"""Exact block-causal segmentation using FlashAttention 2's bottom-right causal mask.

Qwen/Diffusers revision is pinned by require_qwen_runtime. No model parameters or
cache semantics change. Genuine padding masks retain the official SDPA path.
"""
from importlib.metadata import version
from functools import lru_cache

import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AttnProcessor, _qwenimage21_prepare_qkv,
)


@lru_cache(maxsize=1)
def require_flash2():
    from packaging.version import Version
    if Version(version('flash-attn')) < Version('2.1'):
        raise RuntimeError('Qwen segmented attention needs flash-attn>=2.1 bottom-right causality')
    from flash_attn import flash_attn_func
    return flash_attn_func


@lru_cache(maxsize=1)
def require_flash3():
    """Official Hopper interface, intentionally separate from the FA2 package."""
    import inspect
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError('flash3_segmented requires a Hopper SM90 CUDA device')
    try:
        from flash_attn_interface import flash_attn_func
    except ImportError as exc:
        raise RuntimeError('flash3_segmented requires the isolated official Hopper FA3 build') from exc
    signature = inspect.signature(flash_attn_func)
    if 'causal' not in signature.parameters or 'return_attn_probs' not in signature.parameters:
        raise RuntimeError(f'Unsupported Hopper FA3 interface: {signature}')

    def flash3(q, k, v, *, dropout_p=0.0, causal=False):
        if dropout_p != 0.0:
            raise ValueError('Hopper FA3 adapter supports zero dropout only')
        # FA3 has no dropout_p keyword. Default return_attn_probs=False returns a Tensor.
        result = flash_attn_func(q, k, v, causal=causal, return_attn_probs=False)
        if not isinstance(result, torch.Tensor):
            raise RuntimeError('Unexpected Hopper FA3 result; expected Tensor')
        return result
    return flash3


class QwenImage21Flash2Processor(QwenImage21AttnProcessor):
    """An image block sees its full prefix; a text block sees prefix + causal self.

    For text queries [start:end] and keys [:end], FA2 permits j <= i + (K-Q)
    = i + start. This is exactly [ones(Q,start), tril(ones(Q,Q))].
    """
    def flash_kernel(self):
        return require_flash2()

    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None,
                 layer_cache=None, kv_cache_mode=None, cache_write_slice=None,
                 segments=None, key_valid=None):
        if (key_valid is not None or attention_mask is not None
                or hidden_states.device.type != 'cuda'
                or hidden_states.dtype not in (torch.float16, torch.bfloat16)):
            return super().__call__(attn, hidden_states, attention_mask, rotary_emb,
                                    layer_cache, kv_cache_mode, cache_write_slice, segments, key_valid)
        flash = self.flash_kernel()
        q, k, v, seq_len_q = _qwenimage21_prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice)
        if segments is None:
            output = flash(q, k, v, dropout_p=0.0, causal=False)
        else:
            outputs = [flash(q[:, start:end], k[:, :end], v[:, :end],
                             dropout_p=0.0, causal=is_text)
                       for start, end, is_text in segments]
            prefix = segments[-1][1] if segments else 0
            outputs.append(flash(q[:, prefix:], k, v, dropout_p=0.0, causal=False))
            output = torch.cat(outputs, dim=1)
        output = output[:, :seq_len_q].flatten(2, 3).type_as(q)
        return attn.to_out[1](attn.to_out[0](output))


class QwenImage21Flash3Processor(QwenImage21Flash2Processor):
    """Identical segmentation, QKV/cache handling and SDPA padding fallback; FA3 kernels."""
    def flash_kernel(self):
        return require_flash3()


def configure_attention(model, backend):
    if backend == 'sdpa':
        processor = QwenImage21AttnProcessor
    elif backend == 'flash2_segmented':
        require_flash2()  # Fail at startup, rather than silently miss the requested backend.
        processor = QwenImage21Flash2Processor
    elif backend == 'flash3_segmented':
        require_flash3()
        processor = QwenImage21Flash3Processor
    else:
        raise ValueError(f'Unknown Qwen attention backend: {backend}')
    for block in model.transformer_blocks:
        block.attn.set_processor(processor())
    model._qwen_attention_backend = backend
    return model
