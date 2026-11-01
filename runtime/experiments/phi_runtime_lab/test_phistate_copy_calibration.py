from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn

import phistate_copy_calibration as c


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens=nn.Embedding(12,12)
    def forward(self,ids):
        return self.tokens(ids)


class CalibrationTests(unittest.TestCase):
    def test_equal_counts_and_holdout_prefix_isolation(self):
        def enc(tok,prompt,answer,foil):
            self.assertNotEqual(answer,foil)
            return ((prompt,answer),answer,foil,'eos')
        with patch.object(c.warm,'encode',side_effect=enc):
            b=c.datasets(None)
        self.assertTrue(all(len(rows)==32 for rows in b.values()))
        self.assertFalse({x[0][0] for x in b['train_familiar']} & {x[0][0] for x in b['dev_familiar']})

    def test_two_gates_and_cached_end(self):
        r=((1,2,5),5,6,2)
        pools={key:[r]*32 for key in ('train_familiar','dev_familiar','dev_new')}
        req={'max_steps':16,'seed':20}
        tests=[(32,24,'single_copy_passed',0),(32,23,'single_copy_failed',16),(28,32,'single_copy_failed',16)]
        torch.set_num_threads(1)
        for seen,dev,outcome,expected_step in tests:
            with TemporaryDirectory() as temp:
                root=Path(temp)
                def score(model,rows,group_size=1):
                    correct=seen if id(rows)==id(pools['train_familiar']) else dev
                    return {'items':32,'exact_digit_then_eos':correct,'group_rate':correct/32}
                with (patch.object(c.spm,'SentencePieceProcessor',return_value=object()),
                      patch.object(c,'datasets',return_value=pools),
                      patch.object(c,'ckpt_load',return_value=(TinyLM(),{'sft_steps':4000,'chat_template':c.TEMPLATE})),
                      patch.object(c.warm,'evaluate',side_effect=score)):
                    result=c._run(None,None,root,req)
                    cached=c._run(None,None,root,req)
                self.assertEqual(cached,result)
                self.assertEqual((result['status'],result['outcome'],result['steps_completed']),
                                 ('complete',outcome,expected_step))


if __name__=='__main__':
    unittest.main()
