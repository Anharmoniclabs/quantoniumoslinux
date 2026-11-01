import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import phistate_binding_journal as j


class TestJournal(unittest.TestCase):
    def test_gate_boundaries(self):
        fields = tuple(j.GATES)
        for values,expected in [((.5,1,1),'training_examples_not_yet_fit'),((1,0,1),'seen_fit_only'),
                                ((1,1,.5),'transfer_with_familiar_wording'),((.875,.75,.75),'transfer_on_this_binding_task')]:
            self.assertEqual(j.decision(dict(zip(fields,values))),expected)

    def test_control_failure_stop_resume_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as s:
            root=Path(tmp)
            source=root/'coherence_execution_v1/source.pt'
            tokenizer=root/'chat_recovery_v1/data/tokenizer.model'
            for p in (source,tokenizer):
                p.parent.mkdir(parents=True);p.write_bytes(b'fixture')
            s.enter_context(patch.object(j,'PARENT',j.sha(source)))
            s.enter_context(patch.object(j,'TOKENIZER',j.sha(tokenizer)))
            s.enter_context(patch.object(j.torch.cuda,'is_available',return_value=False))
            ref = s.enter_context(patch.object(j.reference,'run',return_value={'qualified':False,'generic_dense_qualified':False}))
            warm = s.enter_context(patch.object(j.fit,'run'))
            stop=root/j.OUTPUT/'STOP'
            stop.parent.mkdir();stop.touch()
            self.assertEqual(j.run(root)['status'],'stopped')
            stop.unlink()
            result=j.run(root)
            self.assertEqual(result['outcome'],'positive_reference_failed')
            self.assertEqual(result,j.run(root))
            self.assertEqual(ref.call_count,1)
            warm.assert_not_called()


    def test_one_click_uses_copy_checkpoint_when_direct_fit_fails(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as mocks:
            root=Path(tmp)
            parent=root/'coherence_execution_v1/source.pt'
            tokenizer=root/'chat_recovery_v1/data/tokenizer.model'
            for path in (parent,tokenizer):
                path.parent.mkdir(parents=True); path.write_bytes(b'fixture')
            mocks.enter_context(patch.object(j,'PARENT',j.sha(parent)))
            mocks.enter_context(patch.object(j,'TOKENIZER',j.sha(tokenizer)))
            mocks.enter_context(patch.object(j.torch.cuda,'is_available',return_value=False))
            mocks.enter_context(patch.object(j.reference,'run',return_value={'qualified':True,'generic_dense_qualified':False}))
            direct=dict(status='complete',steps_completed=j.STEPS,**dict.fromkeys(j.GATES,0.))
            successful=dict(status='complete',steps_completed=j.STEPS,**dict.fromkeys(j.GATES,1.))
            cp=dict(status='complete',outcome='single_copy_passed',steps_completed=80,seen_correct=32,seen_total=32,
                    dev_correct=32,dev_total=32,new_wording_correct=4,new_wording_total=32)
            fit=mocks.enter_context(patch.object(j.fit,'run',return_value=direct))
            single=mocks.enter_context(patch.object(j.calibration,'run',return_value=cp))
            bridge=mocks.enter_context(patch.object(j.bridge,'run',return_value=successful))
            done=j.run(root)
            self.assertEqual(done['outcome'],'after_copy_transfer_on_this_binding_task')
            self.assertEqual(done['direct_outcome'],'training_examples_not_yet_fit')
            self.assertEqual(bridge.call_args.args[2],root/j.OUTPUT/'single_copy')
            self.assertEqual(j.run(root),done)
            self.assertEqual((fit.call_count,single.call_count,bridge.call_count),(1,1,1))

if __name__=='__main__':
    unittest.main()
