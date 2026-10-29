"""Regression checks for image registration, rig guidance and contact contracts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from train_v07 import PoseData, PoseInterpolator, DEFAULT_ASSETS, contact_loss
from v07_image_rig import fit_image
from v07_preview import warp_highres
from v07_scene import (KEY_TIMES,FLOOR,conditioning,contact_point,frame_times,
                       pin_contact,rig_flow)


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data = PoseData(DEFAULT_ASSETS,64)

    def test_source_files_unchanged(self):
        for name,digest in self.data.hashes.items():
            self.assertEqual(hashlib.sha256((DEFAULT_ASSETS/name).read_bytes()).hexdigest(),digest)

    def test_every_preview_contains_all_exact_keys(self):
        for count in (24,48,72,120):
            times = frame_times(count)
            self.assertEqual(len(times),count)
            self.assertTrue(np.all(np.diff(times)>0))
            for key in KEY_TIMES:
                self.assertIn(key,times)

    def test_contact_uses_opaque_sole_not_canvas_edge(self):
        alpha = np.zeros((32,32))
        alpha[10:21,8:17] = 1
        alpha[31,31] = .1
        self.assertEqual(contact_point(alpha),(12.,20.))

    def test_contact_pin_corrects_drift(self):
        frame = torch.zeros(1,4,64,64)
        frame[:,:,14:40,18:27] = 1
        output,delta = pin_contact(frame,31,round(FLOOR*64))
        x,y = contact_point(output[0,3].numpy())
        self.assertLessEqual(abs(x-31),.5)
        self.assertEqual(y,round(FLOOR*64))
        self.assertNotEqual(delta,(0,0))

    def test_penetration_penalized_only_when_planted(self):
        target = self.data.frames[:1]
        bad = target.clone()
        bad[:,3,58:62,25:35] = 1
        self.assertGreater(contact_loss(bad,target,torch.ones(1)).item(),0)
        self.assertEqual(contact_loss(bad,target,torch.zeros(1)).item(),0)

    def test_rig_flow_identity_and_translation(self):
        points = np.array([[10,10],[20,30],[40,40],[32,55]])
        self.assertEqual(float(rig_flow(points,points,64).abs().max()),0)
        flow = rig_flow(points,points+[4,-7],64)
        self.assertTrue(torch.allclose(flow[0],torch.full((64,64),-4.)))
        self.assertTrue(torch.allclose(flow[1],torch.full((64,64),7.)))

    def test_highres_flow_uses_pixel_scale(self):
        image = torch.zeros(1,4,256,256)
        image[:,:,100:120,100:120] = 1
        flow = torch.zeros(1,2,64,64)
        flow[:,0] = 2
        out = warp_highres(image,flow,64)
        # Backward +2 training pixels shifts the image left by 8 output pixels.
        self.assertTrue(torch.allclose(out[:,:,100:120,92:112],torch.ones(1,4,20,20),atol=1e-4))

    def test_network_endpoints_are_exact_after_weight_changes(self):
        model = PoseInterpolator(64,8)
        torch.nn.init.normal_(model.head.weight,std=.1)
        a,b = self.data.frames[:1],self.data.frames[1:2]
        controls = torch.randn(1,13,64,64)
        for t,expected in ((0,a),(1,b)):
            result,*_ = model(a,b,t,1,controls)
            self.assertTrue(torch.equal(result,expected))

    def test_backward_and_alpha_contract(self):
        model = PoseInterpolator(64,8)
        a,b = self.data.frames[:1],self.data.frames[1:2]
        controls = torch.zeros(1,13,64,64)
        result,*_ = model(a,b,.5,1,controls)
        result.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(model.head.weight.grad.abs().sum().item(),0)
        self.assertTrue((result[:,:3]<=result[:,3:4]+1e-6).all())

    def test_missing_blender_shot_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                self.data.attach_shot(folder)

    def test_wrong_blender_source_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'shot_manifest.json').write_text(json.dumps({'data_fingerprint':'wrong'}))
            with self.assertRaisesRegex(ValueError,'does not match'):
                self.data.attach_shot(folder)

    def test_source_image_can_build_registered_joint_input(self):
        fitted = fit_image(DEFAULT_ASSETS/'pose_01.webp',DEFAULT_ASSETS)
        self.assertEqual(len(fitted['points']),14)
        self.assertEqual(fitted['dimensions'],[351,403])
        self.assertTrue(np.isfinite(fitted['points']).all())

    def test_empty_image_rejected_before_detector(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder,'empty.png')
            Image.new('RGBA',(100,200),(0,0,0,0)).save(path)
            with self.assertRaisesRegex(ValueError,'nonempty'):
                fit_image(path)


if __name__ == '__main__':
    unittest.main()
