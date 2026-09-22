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


class QwenImage21Flash2Processor(QwenImage21AttnProcessor):
    """An image block sees its full prefix; a text block sees prefix + causal self.

    For text queries [start:end] and keys [:end], FA2 permits j <= i + (K-Q)
    = i + start. This is exactly [ones(Q,start), tril(ones(Q,Q))].
    """
    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None,
                 layer_cache=None, kv_cache_mode=None, cache_write_slice=None,
                 segments=None, key_valid=None):
        if (key_valid is not None or attention_mask is not None
                or hidden_states.device.type != 'cuda'
                or hidden_states.dtype not in (torch.float16, torch.bfloat16)):
            return super().__call__(attn, hidden_states, attention_mask, rotary_emb,
                                    layer_cache, kv_cache_mode, cache_write_slice, segments, key_valid)
        flash = require_flash2()
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


def configure_attention(model, backend):
    if backend == 'sdpa':
        processor = QwenImage21AttnProcessor
    elif backend == 'flash2_segmented':
        require_flash2()  # Fail at startup, rather than silently miss the requested backend.
        processor = QwenImage21Flash2Processor
    else:
        raise ValueError(f'Unknown Qwen attention backend: {backend}')
    for block in model.transformer_blocks:
        block.attn.set_processor(processor())
    model._qwen_attention_backend = backend
    return model
