"""Bounded curriculum training selected by generated development answers, with replay."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import signal
from types import SimpleNamespace

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

import moe_train
from chat_recovery_data import SOURCES, TEMPLATE, sha, verify_prepared
from chat_recovery_train import atomic_json, load_rows, run_lock, tensors
from generalization_curriculum import build, fingerprint, score
from tiny_learning_check import row, decode, audit


@torch.inference_mode()
def evaluate(model, tokenizer, items, label):
    model.eval()
    buckets = defaultdict(lambda: {'passed':0, 'total':0})
    results = []
    for i, item in enumerate(items):
        limit = 96 if item['kind']=='code' else (16 if item['kind']=='arithmetic' else 64)
        answer, eos = decode(model, tokenizer, item['prompt'], limit)
        passed = score(item, answer, eos)
        result = dict(**item, generated=answer, eos=eos, passed=passed)
        results.append(result)
        buckets[item['bucket']]['total'] += 1
        buckets[item['bucket']]['passed'] += int(passed)
        if (i+1)%12 == 0:
            print(label, 'generated', i+1, '/', len(items), flush=True)
    for b in buckets.values():
        b['rate'] = b['passed']/b['total']
    return dict(macro_score=sum(b['rate'] for b in buckets.values())/len(buckets),
                buckets=dict(buckets), samples=results)


def brief(label, results):
    print(label, json.dumps({k:v for k,v in results.items() if k!='samples'}), flush=True)
    for bucket in results['buckets']:
        sample = next(r for r in results['samples'] if r['bucket']==bucket)
        print('QUESTION:', sample['prompt'], '\nANSWER:', repr(sample['generated']),
              '| passed:', sample['passed'], flush=True)


def run(a):
    output, source, data, sft = map(lambda p:Path(p).resolve(), (a.run_dir,a.checkpoint,a.data_dir,a.sft_dir))
    if output in (source.parent, data, sft):
        raise RuntimeError('Use a separate curriculum output directory')
    with run_lock(output), run_lock(source.parent):
        return train(a, output, source, data, sft)


def train(a, output, source, data, sft):
    if not 1 <= a.steps <= 2400 or a.eval_every < 1 or a.save_every < 1:
        raise ValueError('Invalid bounded run settings')
    moe_train.setup(2, 20261031)
    device = moe_train.resolve_device(a.device)
    source_hash = sha(source)
    base = torch.load(source, map_location='cpu', weights_only=True)
    if sha(source) != source_hash:
        raise RuntimeError('Source changed while loading')
    corpus, replay, _ = moe_train.load_data(str(data))
    token_hash = sha(data/'tokenizer.model')
    prepared = verify_prepared(sft, token_hash, 512)
    if (base.get('format')!='PhiStateMoE-v1' or base.get('training_stage')!='chat_recovery_v1' or
        base['data_fingerprint']!=corpus['data_fingerprint'] or base.get('tokenizer_sha256')!=token_hash or
        prepared['base_data_fingerprint']!=corpus['data_fingerprint'] or
        corpus.get('whitespace_repair',{}).get('tokenizer_sha256')!=token_hash or
        base.get('train_args',{}).get('optimizer')!='adafactor' or 'optimizer_state' not in base):
        raise RuntimeError('Use the unchanged chat recovery latest.pt and its prepared data/tokenizer')
    for key, spec in corpus['bins'].items():
        if key.startswith(('train_', 'val_')) and sha(data/spec['file'])!=spec['sha256']:
            raise RuntimeError('Replay corpus hash mismatch: '+key)
    tokenizer = spm.SentencePieceProcessor(model_file=str(data/'tokenizer.model'))
    if tokenizer.vocab_size()!=base['config']['vocab']:
        raise RuntimeError('Vocabulary mismatch')
    splits = build()
    curriculum_hash = fingerprint(splits)
    encoded = {}
    for item in splits['train']:
        x,y = row(tokenizer,item['prompt'],item['answer'])
        encoded[item['prompt']] = dict(input_ids=x,labels=y)
    pools = load_rows(sft,tokenizer,512)['train']
    # Filter exact evaluation questions from existing SFT replay, if any occur there.
    reserved = {' '.join(r['prompt'].lower().split()) for s in ('dev','final') for r in splits[s]}
    pools = {s:[r for r in rows if all(' '.join(m['content'].lower().split()) not in reserved
                                    for m in r['messages'] if m['role']=='user')] for s,rows in pools.items()}
    if any(not rows for rows in pools.values()):
        raise RuntimeError('An SFT replay source became empty')
    groups = defaultdict(list)
    for item in splits['train']:
        groups[item['kind']].append(encoded[item['prompt']])
    model = moe_train.PhiStateMoELM(moe_train.MoEConfig(**base['config'])).to(device)
    model.load_state_dict(base['state_dict'],strict=True)
    train_args = base['train_args']
    optimizer = moe_train.optimizer_for(SimpleNamespace(**train_args),model)
    optimizer.load_state_dict(base['optimizer_state'])
    parent, parent_sft = base['step'],base['sft_steps']
    base_tokens = base.get('trained_tokens',0)
    del base
    settings = dict(source_sha256=source_hash,tokenizer_sha256=token_hash,curriculum_sha256=curriculum_hash,
        sft_fingerprint=prepared['data_fingerprint'],accum=4,seed=20261031,curriculum_share=.6,
        conversation_share=.2,replay_share=.2,replay_context=128,eval_every=a.eval_every,
        save_every=a.save_every,optimizer='preserved_adafactor',retention_limit=.5)
    rng = random.Random(settings['seed'])
    np_rng = np.random.default_rng(settings['seed'])
    step, processed, best_score, best_step, bad_checks = 0,0,-1.,0,0
    latest = output/'latest.pt'
    if latest.exists():
        state = torch.load(latest,map_location='cpu',weights_only=True)
        if state.get('generalization_settings')!=settings:
            raise RuntimeError('Existing run source/settings differ')
        model.load_state_dict(state['state_dict'],strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        step,processed = state['phase_step'],state['phase_tokens']
        best_score,best_step,bad_checks = state['best_score'],state['best_phase_step'],state['bad_checks']
        baseline,last,retention_base = state['baseline'],state['last_evaluation'],state['retention_baseline']
        rng.setstate(state['python_rng'])
        np_rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        del state
        if (output/'report.json').exists():
            report = json.loads((output/'report.json').read_text())
            if report['target_steps']!=a.steps:
                raise RuntimeError('Final evaluation already opened. Keep this experiment fixed.')
            print('ALREADY COMPLETE:',output/'report.json',flush=True)
            return report
        print('RESUMED curriculum step',step,flush=True)
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError('Output contains files without a resumable checkpoint')
        output.mkdir(parents=True,exist_ok=True)
        atomic_json(output/'curriculum.json',splits)
        print('DATA:',{k:len(v) for k,v in splits.items()},flush=True)
        print('AUDIT:',audit(model,row(tokenizer,splits['train'][0]['prompt'],splits['train'][0]['answer'])),flush=True)
        baseline = evaluate(model,tokenizer,splits['dev'],'BASELINE')
        brief('BASELINE',baseline)
        retention_base = moe_train.evaluate_both(model,replay,128,1,16)
        last = dict(step=0,development=baseline,replay_validation=retention_base)
        best_score = baseline['macro_score']
    stop = [False]
    handlers = {s:signal.signal(s,lambda *_:stop.__setitem__(0,True)) for s in (signal.SIGINT,signal.SIGTERM)}

    def save(status, improved=False):
        record = dict(format='PhiStateMoE-v1',training_stage='generalization_v1',config=vars(model.cfg),
            state_dict=model.state_dict(),optimizer_state=optimizer.state_dict(),step=parent+step,
            sft_steps=parent_sft+step,phase_step=step,phase_tokens=processed,trained_tokens=base_tokens+processed,
            data_fingerprint=corpus['data_fingerprint'],tokenizer_sha256=token_hash,chat_template=TEMPLATE,
            train_args=train_args,generalization_settings=settings,last_evaluation=last,baseline=baseline,
            retention_baseline=retention_base,best_score=best_score,best_phase_step=best_step,bad_checks=bad_checks,
            python_rng=rng.getstate(),numpy_rng=np_rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        moe_train.atomic_checkpoint(latest,record)
        if improved:
            record.pop('optimizer_state')
            moe_train.atomic_checkpoint(output/'best.pt',record)
        atomic_json(output/'progress.json',dict(status=status,saved_step=step,target_steps=a.steps,
            best_development_score=best_score,best_phase_step=best_step,last_evaluation=last))
        print('SAVED',status,step,'/',a.steps,flush=True)

    try:
        if not latest.exists(): save('prepared',True)
        status = 'training'
        for index in range(step+1,a.steps+1):
            if stop[0] or (output/'STOP').exists():
                status='stopped'
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            mean_loss=0.
            for _ in range(4):
                u=rng.random()
                if u<.6:
                    kind=rng.choice(sorted(groups))
                    x,y=tensors(rng.choice(groups[kind]),device)
                elif u<.8:
                    name=rng.choices(list(SOURCES),weights=[s['weight'] for s in SOURCES.values()],k=1)[0]
                    x,y=tensors(rng.choice(pools[name]),device)
                else:
                    domain='code' if rng.random()<.55 else 'general'
                    x,y=moe_train.sample_batch(replay['train_'+domain],1,128,np_rng)
                    x,y=x.to(device),y.to(device)
                logits=model(x)
                loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
                penalty,_=model.router_penalty()
                if not torch.isfinite(loss+penalty): raise RuntimeError('Nonfinite training loss')
                ((loss+penalty)/4).backward()
                processed+=x.numel()
                mean_loss+=float(loss.detach())/4
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            step=index
            if step%25==0: print('TRAIN',step,'/',a.steps,'| mixed BPT',round(mean_loss/math.log(2),4),'| grad',round(float(grad),3),flush=True)
            improved=False
            if step%a.eval_every==0 or step==a.steps:
                dev=evaluate(model,tokenizer,splits['dev'],'DEV')
                retention=moe_train.evaluate_both(model,replay,128,1,16)
                last=dict(step=step,development=dev,replay_validation=retention)
                atomic_json(output/f'development_{step:04d}.json',last)
                brief('DEV',dev)
                retained=retention['val_mean_bpt']<=retention_base['val_mean_bpt']+.5
                improved=retained and dev['macro_score']>best_score+1e-9
                if improved: best_score,best_step,bad_checks=dev['macro_score'],step,0
                else: bad_checks+=1
                if not retained: status='retention_stopped'
                elif step>=600 and bad_checks>=4: status='plateau_stopped'
            if stop[0] or (output/'STOP').exists(): status='stopped'
            elif step==a.steps and status=='training': status='complete'
            if step%a.save_every==0 or step%a.eval_every==0 or status!='training': save(status,improved)
            if status!='training': break
        if status=='training': status='complete' if step>=a.steps else 'stopped'
        save(status)
        if status=='stopped':
            print('Saved. Rerun the same cell to resume; remove STOP if you created it.',flush=True)
            return dict(status=status)
        # Development selects weights; final prompts are opened only after training.
        best=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(best['state_dict'],strict=True)
        selected_step=best['phase_step']
        del best
        final=evaluate(model,tokenizer,splits['final'],'FINAL UNSEEN')
        brief('FINAL UNSEEN',final)
        general_samples=[]
        for question in ['Hello! What can you help me with?','What is the difference between RAM and storage?',
                         'Why does a computer need both a CPU and RAM?']:
            answer,eos=decode(model,tokenizer,question,96)
            general_samples.append(dict(question=question,generated=answer,eos=eos))
            print('OPEN QUESTION:',question,'\nANSWER:',repr(answer),flush=True)
        if sha(source)!=source_hash: raise RuntimeError('Original checkpoint changed externally')
        # Operational export gate only; this is not a broad-capability certification.
        export_ready=(best_score>=baseline['macro_score']+.1 and final['macro_score']>=.5 and
                      final['buckets']['knowledge']['rate']>=.5 and final['buckets']['code']['rate']>=.5)
        report=dict(status=status,target_steps=a.steps,completed_steps=step,selected_phase_step=selected_step,
            baseline_development_score=baseline['macro_score'],best_development_score=best_score,
            final_evaluation=final,open_questions=general_samples,native_export_eligible=export_ready,
            source_unchanged=True,scope='Small curated curriculum; shared concepts and code operations across '
            'wording splits. Only math_new_inputs uses operands absent from this curriculum training. '
            'Prior pretraining exposure is unknown. Not a broad chatbot benchmark.')
        atomic_json(output/'report.json',report)
        print('RESULT:',json.dumps({k:v for k,v in report.items() if k not in ('final_evaluation','open_questions','scope')}),flush=True)
        print('REPORT:',output/'report.json',flush=True)
        return report
    except BaseException as error:
        atomic_json(output/'error.json',dict(error=str(error),last_completed_step=step))
        progress_path=output/'progress.json'
        if progress_path.exists():
            progress=json.loads(progress_path.read_text()); progress.update(status='failed',error=str(error))
            atomic_json(progress_path,progress)
        raise
    finally:
        for s,handler in handlers.items(): signal.signal(s,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','data-dir','sft-dir','run-dir'): p.add_argument('--'+name,required=True)
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--eval-every',type=int,default=200)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--export-native',action='store_true')
    a=p.parse_args()
    result=run(a)
    if a.export_native and result.get('native_export_eligible'):
        from chat_recovery_cpp import export
        export(a.run_dir,a.sft_dir)
    elif a.export_native:
        print('Native export withheld: no qualifying generalization result; original chat unchanged.',flush=True)
