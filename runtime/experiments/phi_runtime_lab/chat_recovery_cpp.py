"""Export/verify the chat checkpoint, then interact using the native C++ model."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import sentencepiece as spm
from chat_recovery_data import TEMPLATE, prompt_text, sha


def export(run_dir, sft_dir, checkpoint_name='best.pt'):
    import torch
    import numpy as np
    from moe_model import PhiStateMoELM, MoEConfig
    from native_cpp_trial import compile_native, export_model, native_run, parity
    from chat_recovery_train import PROBES, atomic_json, run_lock
    run_dir, sft_dir = Path(run_dir), Path(sft_dir)
    with run_lock(run_dir):
        if checkpoint_name not in ('best.pt', 'candidate.pt'):
            raise ValueError('Export best.pt or candidate.pt from the selected run')
        checkpoint = run_dir / checkpoint_name
        source_hash = sha(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if sha(checkpoint) != source_hash or state.get('chat_template') != TEMPLATE:
            raise RuntimeError('Checkpoint changed or chat template mismatch')
        if state['tokenizer_sha256'] != sha(sft_dir / 'tokenizer.model'):
            raise RuntimeError('Chat tokenizer mismatch')
        name = f'native_sft_{state["sft_steps"]:05d}_{source_hash[:10]}_decode1'
        output = run_dir / name
        if output.exists():
            report = json.loads((output / 'report.json').read_text())
            if (report.get('status') != 'passed' or report.get('source_sha256') != source_hash
                    or report.get('decoding_api') != 1):
                raise RuntimeError('Existing native export is not verified')
            for file, expected in report['files'].items():
                if sha(output / file) != expected:
                    raise RuntimeError('Existing native export changed')
            atomic_json(run_dir / 'native_latest.json', {'directory': str(output.resolve())})
            print('NATIVE READY', output, flush=True)
            return output
        torch.set_num_threads(2)
        os.environ['RFTMW_THREADS'] = '2'
        model = PhiStateMoELM(MoEConfig(**state['config']))
        model.load_state_dict(state['state_dict'], strict=True)
        model.eval()
        sp = spm.SentencePieceProcessor(model_file=str(sft_dir / 'tokenizer.model'))
        report = dict(status='checking', source_sha256=source_hash,
                      decoding_api=1,
                      sft_step=state['sft_steps'], step=state['step'], template=TEMPLATE,
                      scope='Native FP32 CPU inference parity, not a coherence or training guarantee', cases=[])
        del state
        with tempfile.TemporaryDirectory(prefix='phistate_chat_native_') as folder:
            folder = Path(folder)
            binary, weights = folder / 'phistate_moe_cli', folder / 'model.phimoe'
            compile_native(binary)
            export_model(model, weights)
            for prompt in PROBES:
                ids = sp.encode(prompt_text([{'role': 'user', 'content': prompt}]))
                _, stats = native_run(binary, weights, ids, model.cfg.vocab, folder, 96, sp.eos_id())
                trajectory = ids + stats['generated_ids']
                logits, _ = native_run(binary, weights, trajectory, model.cfg.vocab, folder)
                checked = parity(model, trajectory, logits)
                row = dict(prompt=prompt, answer=sp.decode(stats['generated_ids']), parity=checked, **stats)
                report['cases'].append(row)
                if not checked['passed']:
                    atomic_json(run_dir / 'native_failure.json', report)
                    raise RuntimeError('Native chat parity failed; export withheld')
                print('\nC++ USER:', prompt, '\nC++ ASSISTANT:', repr(row['answer']), flush=True)
            if sha(checkpoint) != source_hash:
                raise RuntimeError('Selected checkpoint changed during export')
            stage = Path(tempfile.mkdtemp(prefix=name + '.pending_', dir=run_dir))
            for path in (binary, weights, sft_dir / 'tokenizer.model'):
                shutil.copyfile(path, stage / path.name)
            report.update(status='passed', files={p.name: sha(p) for p in stage.iterdir()})
            atomic_json(stage / 'report.json', report)
            stage.rename(output)
        atomic_json(run_dir / 'native_latest.json', {'directory': str(output.resolve())})
        print('NATIVE READY', output, flush=True)
        return output


def chat(run_dir, prompt=None, count=160, decode_config=None):
    import subprocess
    import numpy as np
    from native_cpp_trial import decode_arguments
    decode_args = decode_arguments(decode_config)
    if not 1 <= count <= 256:
        raise ValueError('count must be 1..256')
    run_dir = Path(run_dir)
    output = Path(json.loads((run_dir / 'native_latest.json').read_text())['directory'])
    report = json.loads((output / 'report.json').read_text())
    if report['status'] != 'passed' or report['template'] != TEMPLATE:
        raise RuntimeError('Use a verified chat export')
    if decode_args and report.get('decoding_api') != 1:
        raise RuntimeError('Re-export this checkpoint with the current exporter to enable native sampling')
    for file, expected in report['files'].items():
        if sha(output / file) != expected:
            raise RuntimeError('Native artifact changed: ' + file)
    tokenizer = spm.SentencePieceProcessor(model_file=str(output / 'tokenizer.model'))
    mode = ('sampled decoding ' + json.dumps(decode_config)
            if decode_config and float(decode_config.get('temperature', 0)) > 0 else 'greedy decoding')
    print('Native C++ chat | saved SFT step', report['sft_step'], '|', mode, flush=True)
    history = []
    with tempfile.TemporaryDirectory(prefix='phistate_chat_') as tmp:
        tmp = Path(tmp)
        binary = tmp / 'phistate_moe_cli'
        shutil.copyfile(output / 'phistate_moe_cli', binary)
        binary.chmod(0o755)
        while True:
            question = prompt if prompt is not None else input('\nYOU (blank stops, /reset clears history): ')
            if not question.strip():
                break
            if question.strip() == '/reset':
                history = []
                if prompt is not None:
                    break
                continue
            messages = history + [{'role': 'user', 'content': question}]
            ids = tokenizer.encode(prompt_text(messages))
            while len(ids) + count > 512 and len(messages) > 1:
                messages = messages[2:]
                ids = tokenizer.encode(prompt_text(messages))
            if len(ids) + count > 512:
                print('Prompt too long for this 512-token chat format; shorten it.', flush=True)
                if prompt is not None:
                    break
                continue
            np.asarray(ids, dtype='<u4').tofile(tmp / 'tokens.u32')
            result = subprocess.run([str(binary), str(output / 'model.phimoe'), str(tmp / 'tokens.u32'),
                                     str(tmp / 'logits.f32'), str(count), str(tokenizer.eos_id())] + decode_args,
                                    env=dict(os.environ, RFTMW_THREADS='2'), capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr or result.stdout)
            stats = json.loads(result.stdout)
            answer = tokenizer.decode(stats['generated_ids'])
            print('\nASSISTANT:', answer if answer.strip() else '[whitespace-only output]', flush=True)
            print(f'[{len(stats["generated_ids"])} tokens | {stats["seconds"]:.2f}s]', flush=True)
            history = messages + [{'role': 'assistant', 'content': answer}]
            if prompt is not None:
                break


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('export')
    q.add_argument('--run-dir', required=True)
    q.add_argument('--sft-dir', required=True)
    q.add_argument('--checkpoint-name', choices=['best.pt', 'candidate.pt'], default='best.pt')
    q = sub.add_parser('chat')
    q.add_argument('--run-dir', required=True)
    q.add_argument('--prompt')
    q.add_argument('--new-tokens', type=int, default=160)
    q.add_argument('--temperature', type=float, default=0.)
    q.add_argument('--top-p', type=float, default=1.)
    q.add_argument('--top-k', type=int, default=0)
    q.add_argument('--repetition-penalty', type=float, default=1.)
    q.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.action == 'export':
        export(a.run_dir, a.sft_dir, a.checkpoint_name)
    else:
        if not 1 <= a.new_tokens <= 256:
            p.error('new-tokens must be 1..256')
        config = dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                      repetition_penalty=a.repetition_penalty, seed=a.seed)
        # An all-default command remains compatible with existing greedy exports.
        if config == dict(temperature=0., top_p=1., top_k=0, repetition_penalty=1., seed=0):
            config = None
        chat(a.run_dir, a.prompt, a.new_tokens, config)
