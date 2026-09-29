import sys
import types
import unittest
from unittest.mock import patch
import torch
from verl_distill.models.qwen_image21 import attention as a

class Flash3Adapter(unittest.TestCase):
    def tearDown(self): a.require_flash3.cache_clear()

    def test_hardware_failfast(self):
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'SM90'): a.require_flash3()

    def test_interface_no_dropout_and_gradient(self):
        calls = []
        def kernel(q, k, v, causal=False, return_attn_probs=False):
            calls.append((causal, return_attn_probs)); return q + k + v
        module = types.ModuleType('flash_attn_interface'); module.flash_attn_func = kernel
        with patch.dict(sys.modules, {'flash_attn_interface': module}), \
             patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'get_device_capability', return_value=(9, 0)):
            flash = a.require_flash3()
            q, k, v = [torch.ones(2, requires_grad=True) for _ in range(3)]
            flash(q,k,v,dropout_p=0,causal=True).sum().backward()
            self.assertEqual(calls, [(True, False)])
            for x in (q,k,v): torch.testing.assert_close(x.grad, torch.ones(2))
            with self.assertRaises(ValueError): flash(q,k,v,dropout_p=.1)

    def test_cpu_padding_fallback_never_uses_flash(self):
        with patch.object(a.QwenImage21AttnProcessor, '__call__', return_value='fallback') as original, \
             patch.object(a, 'require_flash3', side_effect=AssertionError('unexpected kernel')):
            result=a.QwenImage21Flash3Processor()(object(),torch.ones(1,2,3),key_valid=torch.ones(1,2))
            self.assertEqual(result,'fallback'); original.assert_called_once()

if __name__ == '__main__': unittest.main()
