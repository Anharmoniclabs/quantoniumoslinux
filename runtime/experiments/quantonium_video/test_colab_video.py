"""Snapshot interruption/recovery and optional CUDA execution contracts."""
import json
from pathlib import Path
import tempfile
import unittest
import sys
import subprocess
import torch

from colab_video_runtime import save_snapshot, restore_latest, verify_snapshot
from train_v07 import PoseInterpolator
from train_open_v08 import external_eval


def fixture(run,step):
    run.mkdir(exist_ok=True)
    state = dict(step=step,config={'version':'test'},model={'x':torch.ones(1)})
    for name in ('latest.pt','best.pt'):
        torch.save(state,run/name)
    (run/'metrics_latest.json').write_text(json.dumps({'phase_step':step}))
    (run/'run_manifest.json').write_text(json.dumps({'config':state['config']}))


class ForegroundTests(unittest.TestCase):
    def test_output_is_logged_and_exit_failure_is_visible(self):
        from colab_video_runtime import run_foreground
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            command=[sys.executable,'-u','-c',"print('checkpoint saved'); raise SystemExit(3)"]
            with self.assertRaises(subprocess.CalledProcessError):
                run_foreground(command,root,root/'train.log',root/'STOP')
            self.assertIn('checkpoint saved',(root/'train.log').read_text())

    def test_interrupt_requests_stop_and_waits_for_save(self):
        from colab_video_runtime import run_foreground
        from unittest.mock import patch, Mock
        class Output:
            def __init__(self): self.calls=0
            def __iter__(self):
                self.calls+=1
                if self.calls==1: raise KeyboardInterrupt
                return iter(['DRIVE SNAPSHOT VERIFIED 101\n','STOPPED phase 101\n'])
            def close(self): pass
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);proc=Mock(stdout=Output());proc.wait.return_value=0
            with patch('subprocess.Popen',return_value=proc):
                self.assertEqual(run_foreground(['trainer'],root,root/'train.log',root/'STOP'),0)
            self.assertTrue((root/'STOP').exists())
            self.assertIn('STOPPED phase 101',(root/'train.log').read_text())
            proc.wait.assert_called_once()


class SnapshotTests(unittest.TestCase):
    def test_roundtrip_and_incomplete_write_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';fixture(run,100)
            saved=save_snapshot(run,root/'drive')
            (root/'drive/snapshots/incomplete').mkdir()
            restored=restore_latest(root/'drive',root/'restore',{'version':'test'})
            self.assertEqual(restored,saved)
            self.assertEqual(torch.load(root/'restore/latest.pt',weights_only=True)['step'],100)

    def test_corrupt_newer_snapshot_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';fixture(run,100)
            old=save_snapshot(run,root/'drive')
            fixture(run,125);new=save_snapshot(run,root/'drive')
            (new/'latest.pt').write_bytes(b'interrupted transfer')
            self.assertEqual(restore_latest(root/'drive',root/'restore',{'version':'test'}),old)

    def test_mismatched_lineage_cannot_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';fixture(run,100)
            save_snapshot(run,root/'drive')
            with self.assertRaisesRegex(ValueError,'silent reset'):
                restore_latest(root/'drive',root/'restore',{'version':'other'})

    def test_newer_local_checkpoint_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';fixture(run,100)
            save_snapshot(run,root/'drive');fixture(run,125)
            with self.assertRaisesRegex(ValueError,'newer'):
                restore_latest(root/'drive',run,{'version':'test'})
            self.assertEqual(torch.load(run/'latest.pt',weights_only=True)['step'],125)

    def test_nonfinite_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';fixture(run,100)
            s=torch.load(run/'latest.pt',weights_only=True);s['model']['x'].fill_(float('nan'))
            torch.save(s,run/'latest.pt')
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                save_snapshot(run,root/'drive')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware not present')
    def test_cuda_forward_backward_and_evaluation(self):
        model=PoseInterpolator(128,4).cuda()
        a=torch.rand(1,4,128,128,device='cuda');b=torch.rand_like(a)
        g=torch.zeros(1,13,128,128,device='cuda')
        pred,*_=model(a,b,.5,0.,g);pred.mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        class Data:
            items={'validation':[None]}
            def tensors(self,values):
                return a.cpu(),((a+b)/2).cpu(),b.cpu(),g.cpu()
        self.assertTrue(torch.isfinite(torch.tensor(external_eval(model,Data())['model_l1'])))


if __name__ == '__main__':
    unittest.main()
