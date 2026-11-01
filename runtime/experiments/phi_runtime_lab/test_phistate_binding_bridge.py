import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn

import phistate_binding_bridge as bridge


class TinyLM(nn.Module):
    def __init__(self):super().__init__();self.emb=nn.Embedding(12,12)
    def forward(self,x):return self.emb(x)


class BridgeTests(unittest.TestCase):
    def test_ineligible_copy_refused(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'report.json').write_text(json.dumps({'status':'complete','outcome':'single_copy_failed'}))
            with self.assertRaisesRegex(ValueError,'passing'):
                bridge.source_report(root)

    def test_rechecks_copy_gate_and_reuses_completed_result(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'copy.pt';out=root/'bridge';out.mkdir()
            m=TinyLM(); opt=torch.optim.AdamW(m.parameters(),lr=bridge.warm.LR,weight_decay=0.,foreach=False)
            reqbase={'scope':'fixture'}
            torch.save(dict(weights=m.state_dict(),optimizer=opt.state_dict(),
                            step=16,request=reqbase,rng=torch.get_rng_state()),src)
            base=dict(request=reqbase,steps_completed=16,seen_correct=32,dev_correct=32,new_wording_correct=4)
            seen={'exact_digit_then_eos':32};novel={'exact_digit_then_eos':4}
            copy_metrics={'train_familiar':seen,'dev_familiar':seen,'dev_new':novel}
            row=((1,3,5),5,6,2)
            toy_train=[[row]*4]*4
            toy_metrics={k:{'group_rate':.5} for k in ('train_familiar','dev_familiar','dev_new')}
            req={'steps':2,'seed':13}
            with (patch.object(bridge,'ckpt_load',side_effect=lambda _: (TinyLM(),{'sft_steps':4000,'chat_template':bridge.TEMPLATE})),
                  patch.object(bridge.spm,'SentencePieceProcessor',return_value=object()),
                  patch.object(bridge.warm,'banks',return_value={'train_familiar':toy_train}),
                  patch.object(bridge.copy,'datasets',return_value={}),
                  patch.object(bridge.warm,'measure',return_value=toy_metrics),
                  patch.object(bridge,'measure_copy',return_value=copy_metrics)):
                wrong={**base,'dev_correct':31}
                with self.assertRaisesRegex(ValueError,'reproduce'):
                    bridge._fit(None,None,src,wrong,out,req)
                r=bridge._fit(None,None,src,base,out,req)
                self.assertEqual((r['status'],r['steps_completed'],r['dev_new_wording_quartet_rate']),('complete',2,.5))
                self.assertEqual(bridge._fit(None,None,src,base,out,req),r)


if __name__=='__main__': unittest.main()
