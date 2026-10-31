"""CPU-only checks of the preflight; these do not verify a T4 or other GPU."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from chat_recovery_data import sha
from device_preflight import run_preflight
from test_chat_recovery import fixture


class DevicePreflightTests(unittest.TestCase):
    def test_real_cpu_parity_preserves_checkpoint_rng_and_flags_and_caches(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            _, checkpoint = fixture(root)
            state = torch.load(checkpoint, weights_only=True)
            state['tokenizer_sha256'] = sha(root/'tokenizer.model')
            torch.save(state, checkpoint)
            original = sha(checkpoint)
            rng = torch.get_rng_state().clone()
            flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            result = run_preflight(checkpoint, root/'tokenizer.model', root/'preflight.json', 'cpu')
            self.assertTrue(result['passed'], result)
            self.assertFalse(result['gpu_tested'])
            self.assertIn('GPU not tested', result['scope'])
            self.assertEqual(result['forward_max_error'], 0.)
            self.assertEqual(result['gradient_max_error'], 0.)
            self.assertEqual(sha(checkpoint), original)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(flags, (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
            cached = run_preflight(checkpoint, root/'tokenizer.model', root/'preflight.json', 'cpu')
            self.assertTrue(cached['cache_hit'])
            # Cache is invalidated when source contents change; invalid config cannot pass.
            state['config']['heads'] = 3
            torch.save(state, checkpoint)
            invalid = run_preflight(checkpoint, root/'tokenizer.model', root/'preflight.json', 'cpu')
            self.assertFalse(invalid['passed'])
            self.assertFalse(invalid['cache_hit'])
            self.assertIn('divisible', invalid['error'])
            self.assertTrue(invalid['checkpoint_unchanged'])

    def test_unavailable_cuda_is_explicit_failure(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            # Failure occurs before attempting checkpoint deserialization.
            checkpoint = root/'source.pt'; checkpoint.write_bytes(b'fixture')
            tokenizer = root/'tokenizer.model'; tokenizer.write_bytes(b'fixture')
            with patch.object(torch.cuda, 'is_available', return_value=False):
                result = run_preflight(checkpoint, tokenizer, root/'report.json', 'cuda')
            self.assertFalse(result['passed'])
            self.assertFalse(result['gpu_tested'])
            self.assertIn('unavailable', result['error'])
            self.assertTrue(result['checkpoint_unchanged'])
            self.assertFalse(json.loads((root/'report.json').read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
