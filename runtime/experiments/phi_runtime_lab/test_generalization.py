"""Offline data/scorer/runner verification, not a claim about actual 38M weights."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import torch

import chat_recovery_data as data
import generalization_curriculum as curriculum
import generalization_train as trainer
from test_chat_recovery import fixture, adapter


def main():
    torch.set_num_threads(2)
    splits=curriculum.build()
    assert {k:len(v) for k,v in splits.items()}=={'train':279,'dev':42,'final':42}
    for rows in splits.values():
        for row in rows:
            assert curriculum.score(row,row['answer'],True), row
            assert not curriculum.score(row,row['answer'],False)
            assert not curriculum.score(row,'irrelevant answer',True)
    assert not curriculum.code_passes('import os\ndef add(a,b):\n return a+b','add')
    assert not curriculum.code_passes('def add(a,b):\n return __import__("os").system("echo bad")','add')
    assert not curriculum.code_passes('def add(a,b):\n return a-b','add')
    assert curriculum.code_passes('def add(a,b):\n return b+a','add')
    assert not curriculum.code_passes('def is_even(x):\n return True','is_even')
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        tok,checkpoint=fixture(root)
        state=torch.load(checkpoint,weights_only=True)
        state.update(training_stage='chat_recovery_v1',tokenizer_sha256=data.sha(root/'tokenizer.model'),sft_steps=2000)
        torch.save(state,checkpoint)
        original=data.sha(checkpoint)
        stub=root/'upstream.md'; stub.write_text('offline fixture')
        def download(name,cache): return stub,data.sha(stub),stub
        sft=root/'sft'
        with patch.object(data,'download_source',download), \
             patch.object(data,'conversation_rows',adapter('conversation')), \
             patch.object(data,'python_rows',adapter('python')), \
             patch.object(data,'support_rows',adapter('support')), \
             patch.object(data,'math_rows',adapter('math')):
            data.build(root,sft,root/'cache',512)
        def args(folder):
            return SimpleNamespace(run_dir=str(root/folder),checkpoint=str(checkpoint),data_dir=str(root),
                                   sft_dir=str(sft),device='cpu',steps=2,eval_every=1,save_every=1)
        # Real forwards/backward/checkpoints; short mocked generations keep this a runner test.
        with patch.object(trainer,'decode',return_value=('',False)):
            full=trainer.run(args('full'))
            assert full['completed_steps']==2 and not full['native_export_eligible']
            atomic=trainer.atomic_json
            def stop_after_one(path,value):
                atomic(path,value)
                if path.name=='progress.json' and value.get('saved_step')==1:
                    (path.parent/'STOP').touch()
            with patch.object(trainer,'atomic_json',stop_after_one):
                stopped=trainer.run(args('resume'))
            assert stopped['status']=='stopped'
            (root/'resume/STOP').unlink()
            trainer.run(args('resume'))
            a=torch.load(root/'full/latest.pt',weights_only=True)
            b=torch.load(root/'resume/latest.pt',weights_only=True)
            assert a['phase_step']==b['phase_step']==2
            assert a['phase_tokens']==b['phase_tokens']
            assert any(not torch.equal(v,state['state_dict'][k]) for k,v in a['state_dict'].items())
            assert all(torch.equal(v,b['state_dict'][k]) for k,v in a['state_dict'].items())
            assert data.sha(checkpoint)==original
            assert trainer.run(args('full'))==full
            initial={'val_code_bpt':5.,'val_general_bpt':7.,'val_mean_bpt':6.}
            regression={**initial,'val_mean_bpt':7.}
            with patch.object(trainer.moe_train,'evaluate_both',side_effect=[initial,regression]):
                guarded=trainer.run(args('guarded'))
            assert guarded['status']=='retention_stopped' and guarded['completed_steps']==1
            assert guarded['selected_phase_step']==0 and not guarded['native_export_eligible']
    print('PASS: disjoint prompts, reference grading, arithmetic-only code interpreter, actual updates, exact resume, STOP, source preservation, retention stop and final-report reuse')


if __name__=='__main__': main()
