import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

import phistate_binding_fit as fit


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(12,8)
        self.head=nn.Linear(8,12)
    def forward(self,ids):
        return self.head(self.embed(ids).cumsum(1))


class FitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.bank={}
        self.quartet=[((1,3,5),5,6,2),((1,4,6),6,5,2),
                      ((1,4,5),5,6,2),((1,3,6),6,5,2)]
        for split in ('train_familiar','dev_familiar','train_new','dev_new'):
            self.bank[split]=[self.quartet]
        self.bank['copy']=self.quartet

    def test_single_digit_targets_and_boundaries(self):
        class Tok:
            def eos_id(self):return 2
        tok=Tok()
        with patch.object(fit,'encode_example',side_effect=[([1,4,5],[-100,5,2]),([1,4,6],[-100,6,2])]):
            self.assertEqual(fit.encode(tok,'test','0','1'),((1,4,5),5,6,2))
        with patch.object(fit,'encode_example',side_effect=[([1,4,5],[-100,5,2]),([1,3,6],[-100,6,2])]):
            with self.assertRaisesRegex(ValueError,'identical prefix'):
                fit.encode(tok,'test','0','1')

    def test_stop_resume_and_completed_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); ref=root/'reference'; alt=root/'resume'
            ref.mkdir();alt.mkdir()
            torch.manual_seed(128)
            initial=TinyLM().state_dict()
            def model():
                m=TinyLM();m.load_state_dict(initial);return m
            req={'fixture':17}
            reference=fit.fit(model(),self.bank,ref,req,4,123)
            self.assertEqual(reference['steps_completed'],4)
            real_step=torch.optim.AdamW.step
            def stop_at_one(opt,*a,**kw):
                result=real_step(opt,*a,**kw);(alt/'STOP').touch();return result
            with patch.object(torch.optim.AdamW,'step',stop_at_one):
                stopped=fit.fit(model(),self.bank,alt,req,4,123)
            self.assertEqual((stopped['status'],stopped['steps_completed']),('stopped',1))
            (alt/'STOP').unlink()
            resumed=fit.fit(model(),self.bank,alt,req,4,123)
            self.assertEqual(reference['curve'],resumed['curve'])
            a=torch.load(ref/'latest.pt',weights_only=True)['weights']
            b=torch.load(alt/'latest.pt',weights_only=True)['weights']
            self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
            self.assertEqual(resumed,fit.fit(model(),self.bank,alt,req,4,123))

    def test_digit_plus_eos_is_required_for_quartet(self):
        class EmitsDigit(nn.Module):
            def forward(self,ids):
                logits=torch.zeros((*ids.shape,12))
                # Cheating oracle fixture exercises scoring, not a learned model.
                for i,row in enumerate(self_quartet):
                    logits[i,1,row[1]]=2
                    logits[i,2,2 if i!=3 else 1]=2
                return logits
        self_quartet=self.quartet
        result=fit.evaluate(EmitsDigit(),self.quartet)
        self.assertEqual((result['top_digit_hits'],result['exact_digit_then_eos'],result['all_correct_groups']),(4,3,0))


if __name__=='__main__':
    unittest.main()
