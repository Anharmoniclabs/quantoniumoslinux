"""Synthetic end-to-end check; does not test the user's inaccessible weights."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import sentencepiece as spm
import torch

import coherence_optimizer_trial as trial
import moe_train
from moe_model import PhiStateMoELM, config_for


def main():
    torch.set_num_threads(2)
    for piece in ('▁', '<0x09>', '<0x0A>', '<0x0D>'):
        assert trial.whitespace_piece(piece), piece
    for piece in ('▁return', '<0x41>', '<|code|>'):
        assert not trial.whitespace_piece(piece), piece
    with TemporaryDirectory() as directory:
        root = Path(directory)
        text = ('def add(a, b):\n    return a + b\n'
                'Quantum physics describes matter and energy.\n') * 60
        (root / 'text.txt').write_text(text)
        spm.SentencePieceTrainer.train(
            input=str(root / 'text.txt'), model_prefix=str(root / 'tokenizer'),
            model_type='bpe', vocab_size=320, hard_vocab_limit=False,
            byte_fallback=True, normalization_rule_name='identity',
            remove_extra_whitespaces=False, add_dummy_prefix=False,
            user_defined_symbols=['<|code|>', '<|general|>'], minloglevel=2)
        sp = spm.SentencePieceProcessor(model_file=str(root / 'tokenizer.model'))
        cfg = config_for('mini', sp.vocab_size())
        model = PhiStateMoELM(cfg)
        settings = dict(optimizer='adafactor', lr=.0003, weight_decay=.05,
                        context=16, batch=1, accum=1, code_share=.55, clip=1., seed=20261030)
        optimizer = moe_train.optimizer_for(SimpleNamespace(**settings), model)
        ids = sp.encode(text)
        x = torch.tensor([ids[:16]])
        y = torch.tensor([ids[1:17]])
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab), y.flatten())
            (loss + model.router_penalty()[0]).backward()
            optimizer.step()
        bins = {}
        for domain in ('code', 'general'):
            for split in ('train', 'val'):
                name = split + '_' + domain
                filename = name + '.bin'
                np.asarray(ids, dtype=np.uint16).tofile(root / filename)
                bins[name] = dict(file=filename, tokens=len(ids), sha256=trial.sha(root / filename))
        manifest = dict(format='PhiMoECorpus-v1', dtype='uint16', bins=bins,
                        data_fingerprint='synthetic-test-only',
                        whitespace_repair={'tokenizer_sha256': trial.sha(root / 'tokenizer.model')})
        (root / 'manifest.json').write_text(json.dumps(manifest))
        checkpoint = root / 'source.pt'
        torch.save(dict(format='PhiStateMoE-v1', config=vars(cfg), step=65000,
                        state_dict=model.state_dict(), train_args=settings,
                        trained_tokens=65000*16, tokenizer='bpe',
                        optimizer_state=optimizer.state_dict(),
                        data_fingerprint=manifest['data_fingerprint']), checkpoint)
        original_hash = trial.sha(checkpoint)
        output = root / 'trial'
        trial.run(SimpleNamespace(checkpoint=str(checkpoint), data_dir=str(root),
                                  output=str(output), device='cpu', steps=2, windows=2))
        report = json.loads((output / 'report.json').read_text())
        assert report['status'] == 'complete' and report['identical_batches']
        assert trial.sha(checkpoint) == original_hash
        assert report['tokens_per_arm'] == 32
        saved = {}
        for arm in report['arms']:
            saved[arm] = torch.load(output / (arm + '.pt'), weights_only=True)
            assert saved[arm]['step'] == 65002
            assert saved[arm]['trained_tokens'] == 65000*16 + 32
            assert not torch.equal(saved[arm]['state_dict']['embedding.weight'], model.embedding.weight)
        control_steps = [int(s['step']) for s in saved['adafactor_control']['optimizer_state']['state'].values()]
        adam_steps = [int(s['step']) for s in saved['adamw_candidate']['optimizer_state']['state'].values()]
        assert max(control_steps) == 5, control_steps
        assert max(adam_steps) == 2, adam_steps
        try:
            trial.run(SimpleNamespace(checkpoint=str(checkpoint), data_dir=str(root),
                                      output=str(output), device='cpu', steps=2, windows=2))
        except RuntimeError as error:
            assert 'Output exists' in str(error)
        else:
            raise AssertionError('Existing output must be refused')
    print('PASS: matched batches, separate optimizer histories, source preservation, byte whitespace, output guard')


if __name__ == '__main__':
    main()
