"""Offline checks for an isolated journal fork and data audit."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import torch
import chat_recovery_data as data
import chat_recovery_train as train
import journal_control as journal
from test_chat_recovery import fixture,adapter


def main():
    torch.set_num_threads(2)
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        corpus=root/'corpus_repaired_v1'; corpus.mkdir()
        tok,source=fixture(corpus)
        original=root/'run_60k_to_61k/latest.pt'; original.parent.mkdir()
        shutil.copyfile(source,original)
        stub=root/'source.md';stub.write_text('offline fixture')
        sft=root/'chat_recovery_v1/data'
        def download(name,cache): return stub,data.sha(stub),stub
        with patch.object(data,'download_source',download), \
             patch.object(data,'conversation_rows',adapter('conversation')), \
             patch.object(data,'python_rows',adapter('python')), \
             patch.object(data,'support_rows',adapter('support')), \
             patch.object(data,'math_rows',adapter('math')):
            data.build(corpus,sft,root/'cache',512)
        chat=root/'chat_recovery_v1/run'
        a=SimpleNamespace(run_dir=str(chat),checkpoint=str(original),sft_dir=str(sft),data_dir=str(corpus),
            steps=1,context=512,accum=1,device='cpu',eval_every=1,save_every=1,val_examples=1,patience=6,clear_stop=False)
        with patch.object(train,'PROBES',['What is Python?']): train.run(a)
        source_hash=data.sha(chat/'latest.pt')
        target=journal.fork_conversation(root)
        assert data.sha(target/'latest.pt')==source_hash
        assert data.sha(target/'best.pt')==data.sha(chat/'best.pt')
        assert journal.fork_conversation(root)==target
        audit=journal.audit_data(root)
        assert len(audit['sft'])==8
        assert all(r['input_tokens']>r['answer_tokens']>0 for r in audit['sft'])
        assert abs(sum(r['expected_sft_input_token_share'] for r in audit['sft'] if r['split']=='train')-1)<1e-9
        inv=journal.inventory(root)
        assert len(inv)==5 and inv[-1]['optimizer_saved']
        assert not journal.worker_active(target)
        journal.request_stop(root)
        assert (target/'STOP').exists() and not (chat/'STOP').exists()
        assert data.sha(chat/'latest.pt')==source_hash
        journal.monitor(root)
    print('PASS: exact independent fork, original preservation, idempotent fork, token audit, inventory and separate stop control')

if __name__=='__main__': main()
