"""Offline runner checks; fixture success is not evidence about the user's model."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import torch
import tiny_learning_check as tiny
from test_chat_recovery import fixture
from chat_recovery_data import sha


def main():
    torch.set_num_threads(2)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        tok, checkpoint = fixture(root)
        state = torch.load(checkpoint, weights_only=True)
        state['tokenizer_sha256'] = sha(root / 'tokenizer.model')
        torch.save(state, checkpoint)
        original = sha(checkpoint)
        def args(folder, steps):
            return SimpleNamespace(checkpoint=str(checkpoint), data_dir=str(root), run_dir=str(root / folder),
                                   steps=steps, eval_every=1, device='cpu')
        # Keep generation bounded for this mechanical test. Real runner uses 96.
        original_decode = tiny.decode
        with patch.object(tiny, 'decode', lambda model, tokenizer, q: original_decode(model, tokenizer, q, 3)):
            first = tiny.run(args('resume', 1))
            assert first['steps'] == 1 and first['first_update_max_delta'] > 0
            assert len(first['training_evaluation']['samples']) == 16
            assert len(first['unseen_wordings']) == 4
            tiny.run(args('resume', 2))
            tiny.run(args('full', 2))
            a = torch.load(root / 'resume/latest.pt', weights_only=True)
            b = torch.load(root / 'full/latest.pt', weights_only=True)
            assert all(torch.equal(v, b['state_dict'][k]) for k,v in a['state_dict'].items())
            (root / 'resume/STOP').touch()
            stopped = tiny.run(args('resume', 3))
            assert stopped['status'] == 'stopped' and stopped['steps'] == 2
            assert sha(checkpoint) == original
            progress = json.loads((root / 'resume/progress.json').read_text())
            assert progress['last_evaluation_step'] == progress['saved_step'] == 2
            # A changed source must not be mixed into a resumable diagnostic.
            state['step'] += 1
            torch.save(state, checkpoint)
            try:
                tiny.run(args('resume', 3))
            except RuntimeError as e:
                assert 'different source' in str(e)
            else:
                raise AssertionError('Source change was accepted')
    print('PASS: masking/roundtrip, forward-cache/causality, finite update, exact resume, STOP, source preservation and source-change guard')

if __name__ == '__main__':
    main()
