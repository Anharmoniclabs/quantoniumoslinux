"""Offline integration tests for complete answers, data splits, resume, and C++ export."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from collections import Counter

import numpy as np
import sentencepiece as spm
import torch

import chat_recovery_data as data
import chat_recovery_train as train
import chat_recovery_cpp as cpp
import moe_train
from moe_model import PhiStateMoELM, config_for


def fixture(root):
    text = ('User:\nPlease explain numbers and Python.\nAssistant:\n'
            'Numbers are useful.\n```python\ndef add(a, b):\n    return a + b\n```\n') * 200
    (root / 'text.txt').write_text(text)
    spm.SentencePieceTrainer.train(input=str(root / 'text.txt'), model_prefix=str(root / 'tokenizer'),
        model_type='bpe', vocab_size=400, hard_vocab_limit=False, byte_fallback=True,
        normalization_rule_name='identity', add_dummy_prefix=False, remove_extra_whitespaces=False,
        user_defined_symbols=['<|code|>', '<|general|>'], minloglevel=2)
    tokenizer = spm.SentencePieceProcessor(model_file=str(root / 'tokenizer.model'))
    ids = tokenizer.encode(text)
    bins = {}
    for domain in ('code', 'general'):
        for split in ('train', 'val'):
            key = split + '_' + domain
            np.asarray(ids, dtype=np.uint16).tofile(root / (key + '.bin'))
            bins[key] = dict(file=key + '.bin', tokens=len(ids), sha256=data.sha(root / (key + '.bin')))
    manifest = dict(format='PhiMoECorpus-v1', tokenizer='bpe', dtype='uint16', vocab=tokenizer.vocab_size(),
                    data_fingerprint='synthetic-repaired-fixture', bins=bins,
                    whitespace_repair={'tokenizer_sha256': data.sha(root / 'tokenizer.model')})
    (root / 'manifest.json').write_text(json.dumps(manifest))
    cfg = config_for('mini', tokenizer.vocab_size())
    model = PhiStateMoELM(cfg)
    args = dict(optimizer='adafactor', lr=.0003, weight_decay=.05, seed=20261030)
    optimizer = moe_train.optimizer_for(SimpleNamespace(**args), model)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = model(torch.tensor([ids[:16]]))
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), torch.tensor(ids[1:17]))
        (loss + model.router_penalty()[0]).backward()
        optimizer.step()
    checkpoint = root / 'source.pt'
    torch.save(dict(format='PhiStateMoE-v1', config=vars(cfg), state_dict=model.state_dict(),
                    step=65000, train_args=args, optimizer_state=optimizer.state_dict(),
                    trained_tokens=65000*512, data_fingerprint=manifest['data_fingerprint']), checkpoint)
    return tokenizer, checkpoint


def adapter(source):
    def rows(path, counts):
        for i in range(40):
            counts['raw'] += 1
            yield dict(source=source, group=f'{source}-group-{i}', messages=[
                {'role': 'user', 'content': f'Please explain {source} example number {i}.'},
                {'role': 'assistant', 'content': f'Example {i} uses numbers and Python.'}])
    return rows


def main():
    torch.set_num_threads(2)
    with TemporaryDirectory() as folder:
        root = Path(folder)
        tokenizer, checkpoint = fixture(root)
        initial_hash = data.sha(checkpoint)
        unicode_jsonl = root / 'unicode.jsonl'
        unicode_jsonl.write_text(json.dumps({'text': 'line\u2028separator'}, ensure_ascii=False) + '\n')
        assert list(data.read_jsonl(unicode_jsonl)) == [{'text': 'line\u2028separator'}]
        messages = [{'role': 'user', 'content': 'What is Python?'},
                    {'role': 'assistant', 'content': 'Python is a programming language.'}]
        x, y = data.encode_example(messages, tokenizer, 256)
        prefix = tokenizer.encode(data.prompt_text(messages[:-1]))
        assert all(v == -100 for v in y[:len(prefix)-1])
        assert y[len(prefix)-1] != -100 and y[-1] == tokenizer.eos_id()
        assert tokenizer.decode(x) == data.prompt_text(messages[:-1]) + messages[-1]['content']
        logits = torch.randn(len(y), tokenizer.vocab_size(), requires_grad=True)
        torch.nn.functional.cross_entropy(logits, torch.tensor(y), ignore_index=-100).backward()
        assert not logits.grad[:len(prefix)-1].count_nonzero()
        assert logits.grad[-1].count_nonzero()
        try:
            data.encode_example(messages, tokenizer, len(x)-1)
        except ValueError:
            pass
        else:
            raise AssertionError('Long answers must be rejected, not truncated and labelled EOS')
        # Distinct group IDs with the same final question must stay together.
        dup = [dict(source='python', group='a', messages=messages),
               dict(source='python', group='b', messages=messages)]
        grouped = data.group_split(dup)
        assert grouped[0]['split_group'] == grouped[1]['split_group']
        bad_python = root / 'bad_python.json'
        bad_python.write_text(json.dumps([[dict(language='Python', instruction='Add.',
            code_solution='python\ndef add(a,b):\n    return a+b'),
            dict(language='Python', instruction='Bad.', code_solution='import \n')]]))
        counts = Counter()
        valid = list(data.python_rows(bad_python, counts))
        assert len(valid) == 1 and counts['invalid_python'] == 1
        stub = root / 'upstream.md'
        stub.write_text('Synthetic fixture only')
        def download(name, cache):
            return stub, data.sha(stub), stub
        sft = root / 'sft'
        with patch.object(data, 'download_source', download), \
             patch.object(data, 'conversation_rows', adapter('conversation')), \
             patch.object(data, 'python_rows', adapter('python')), \
             patch.object(data, 'support_rows', adapter('support')), \
             patch.object(data, 'math_rows', adapter('math')):
            data.build(root, sft, root / 'cache', 256)
        pools = train.load_rows(sft, tokenizer, 256)
        assert all(len(pools['train'][s]) >= 10 and pools['val'][s] for s in data.SOURCES)
        def args(run_dir, steps):
            return SimpleNamespace(run_dir=str(run_dir), checkpoint=str(checkpoint), sft_dir=str(sft),
                data_dir=str(root), steps=steps, context=256, accum=1, device='cpu',
                eval_every=1, save_every=1, val_examples=1, patience=6, clear_stop=False)
        with patch.object(train, 'PROBES', ['What is Python?']):
            full, resumed = root / 'full', root / 'resumed'
            assert train.run(args(full, 2)) == 'complete'
            assert train.run(args(resumed, 1)) == 'complete'
            assert train.run(args(resumed, 2)) == 'complete'
            a = torch.load(full / 'latest.pt', weights_only=True)
            b = torch.load(resumed / 'latest.pt', weights_only=True)
            for key in a['state_dict']:
                assert torch.equal(a['state_dict'][key], b['state_dict'][key]), key
            assert a['phase_tokens'] == b['phase_tokens'] and a['orders'] == b['orders']
            assert max(int(s['step']) for s in a['optimizer_state']['state'].values()) == 5
            (resumed / 'STOP').touch()
            assert train.run(args(resumed, 3)) == 'stopped'
            assert torch.load(resumed / 'latest.pt', weights_only=True)['sft_steps'] == 2
            assert data.sha(checkpoint) == initial_hash
            output = cpp.export(full, sft)
            assert json.loads((output / 'report.json').read_text())['status'] == 'passed'
            cpp.chat(full, prompt='What is Python?', count=8)
            # A retention regression must stop without replacing the baseline best.
            initial = {'val_mean_bpt': 6., 'val_code_bpt': 5., 'val_general_bpt': 7.,
                       'val_count_tokens': 4096, 'val_elapsed_seconds': 0.}
            regressed = {**initial, 'val_mean_bpt': 7.}
            guarded = root / 'guarded'
            with patch.object(moe_train, 'evaluate_both', side_effect=[initial, regressed]):
                assert train.run(args(guarded, 1)) == 'retention_stopped'
            assert torch.load(guarded / 'best.pt', weights_only=True)['sft_steps'] == 0
            assert json.loads((guarded / 'progress.json').read_text())['status'] == 'retention_stopped'
        # Prepared-data tampering must fail before training.
        with (sft / 'train.jsonl').open('a') as f:
            f.write('\n')
        try:
            data.verify_prepared(sft, data.sha(root / 'tokenizer.model'), 256)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Modified data was accepted')
    print('PASS: complete-answer masking, EOS, grouped splits, syntax rejection, optimizer preservation, exact resume, STOP, source preservation, C++ chat parity and tamper guard')


if __name__ == '__main__':
    main()
