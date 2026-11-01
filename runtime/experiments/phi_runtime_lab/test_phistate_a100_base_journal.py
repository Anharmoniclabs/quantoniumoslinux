import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import phistate_a100_base_journal as j


def metrics(other,context=.1,history=.1):
    return {'mean_bpt':other,'code':{'other_bpt':other,'context_gain_bpt':context,'wrong_history_gain_bpt':history},
            'general':{'other_bpt':other,'context_gain_bpt':context,'wrong_history_gain_bpt':history}}


class TestBaseJournal(unittest.TestCase):
    def test_prerequisite_before_hardware(self):
        with patch.object(j,'prerequisite',return_value={'qualified':False}), patch.object(j.training,'require_a100') as gate:
            with self.assertRaisesRegex(RuntimeError,'prerequisite'):j.run('.')
            gate.assert_not_called()

    def test_context_and_content_are_required(self):
        self.assertTrue(j.learns(metrics(9),metrics(8)))
        self.assertFalse(j.learns(metrics(9),metrics(8,context=.005)))
        self.assertFalse(j.learns(metrics(9),metrics(8,history=0.)))
        self.assertFalse(j.learns(metrics(8.01),metrics(8.)))
        bad=metrics(7); bad['code']['other_bpt']=8.08
        self.assertFalse(j.retains(metrics(8),bad))
        self.assertTrue(j.retains(metrics(8),metrics(7.5)))

    def test_partially_overlapping_windows_rejected(self):
        a={d:[{'offset':5,'context':8}] for d in ('code','general')}
        b={d:[{'offset':13,'context':8}] for d in ('code','general')}
        self.assertFalse(j.disjoint(a,b)); b['code'][0]['offset']=b['general'][0]['offset']=14
        self.assertTrue(j.disjoint(a,b))

    def test_two_seeds_before_any_warm_updates_and_cached_completion(self):
        for strong_phi in (True,False):
            with self.subTest(strong_phi=strong_phi),tempfile.TemporaryDirectory() as tmp,ExitStack() as st:
                root=Path(tmp);base={}
                for i,n in enumerate(('65K','66K')):
                    path=root/f'{n}.pt';torch.save({'step':i+1,'data_fingerprint':j.CORPUS},path)
                    base[n]=(path.name,j.sha(path),i+1)
                st.enter_context(patch.object(j,'BASES',base))
                st.enter_context(patch.object(j,'prerequisite',return_value={'qualified':True}))
                st.enter_context(patch.object(j.training,'require_a100'))
                st.enter_context(patch.object(j.windows,'open_corpus',return_value=SimpleNamespace(fingerprint=j.CORPUS)))
                st.enter_context(patch.object(j,'validation_plans',return_value=({'dev':0},{'ver':0},13)))
                st.enter_context(patch.object(j.windows,'schedule',side_effect=lambda *a,**k:{'fingerprint':str(k['seed'])}))
                st.enter_context(patch.object(j,'run_preflight',return_value={'passed':True}))
                st.enter_context(patch.object(j,'model_measure',side_effect=lambda path,*args,**kw:{'metrics':metrics(7. if path.name=='candidate.pt' else 9.),'samples':[]}))
                st.enter_context(patch.object(j,'endpoint_samples',return_value=[]))
                calls=[]
                def arm(parent,corpus,out,**kwargs):
                    name=kwargs['arm'];calls.append(name)
                    out.mkdir(exist_ok=True); path=out/'candidate.pt';path.write_bytes(name.encode())
                    end=metrics(7.) if strong_phi or name!='phistate' else metrics(8.99)
                    return {'status':'complete','trained_tokens':j.STEPS*j.BATCH*j.CONTEXT,'latest':str(path),
                            'device_receipt':{},'curve':[{'metrics':metrics(9.)},{'metrics':end}]}
                st.enter_context(patch.object(j.training,'run_arm',side_effect=arm))
                result=j.run(root)
                self.assertEqual(calls, ['dense','phistate','dense','phistate']+(['warm'] if strong_phi else []))
                self.assertEqual(result['outcome'],'base_loss_improved' if strong_phi else 'phistate_context_gate_failed')
                self.assertTrue((root/j.OUTPUT/'confirmation.json').exists()==strong_phi)
                self.assertEqual(result,j.run(root))
                self.assertEqual(len(calls),5 if strong_phi else 4)
