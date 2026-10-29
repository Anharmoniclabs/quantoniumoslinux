"""Real tiny-Wan tests. No pretrained downloads and no claim of L4 validation."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from diffusers import WanTransformer3DModel, WanPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderKLWan
from peft import get_peft_model_state_dict

from wan_training import (add_adapter, flow_inputs, flow_loss, save_checkpoint,
    restore_checkpoint, find_checkpoint, verify_checkpoint, prune_checkpoints,
    read_cache, digest, SOURCE, VERSION, write_json)


def tiny_model(adapters=True):
    torch.manual_seed(44)
    model = WanTransformer3DModel(num_attention_heads=2, attention_head_dim=16,
        in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=64, num_layers=1)
    if adapters:
        add_adapter(model, 2)
        model.enable_gradient_checkpointing()
    return model


def sample():
    rng = torch.Generator().manual_seed(8)
    return dict(latent=torch.randn(1, 4, 2, 4, 4, generator=rng),
                prompt_embeds=torch.randn(1, 8, 16, generator=rng))


def update(model, opt, rng):
    model.train()
    opt.zero_grad(set_to_none=True)
    loss = flow_loss(model, sample(), rng, torch.device('cpu'), torch.float32)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    return float(loss.detach())


class WanTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_flow_target_and_endpoints(self):
        clean, noise = torch.zeros(1, 4, 2, 4, 4), torch.ones(1, 4, 2, 4, 4)
        for t in (0., .25, 1.):
            noisy, target = flow_inputs(clean, noise, torch.tensor([t]))
            self.assertTrue(torch.equal(noisy, torch.full_like(clean, t)))
            self.assertTrue(torch.equal(target, noise))

    def test_actual_wan_gradient_changes_only_adapters(self):
        model = tiny_model()
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
        loss = update(model, opt, torch.Generator().manual_seed(1))
        changed = [k for k, p in model.named_parameters() if not torch.equal(before[k], p)]
        self.assertGreater(loss, 0)
        self.assertTrue(changed)
        self.assertTrue(all('lora_' in k for k in changed))

    def test_bfloat16_base_with_float32_adapter_gradients(self):
        model = tiny_model(False).to(torch.bfloat16)
        params = add_adapter(model, 2)
        model.enable_gradient_checkpointing()
        before = [p.detach().clone() for p in params]
        opt = torch.optim.AdamW(params, lr=.001)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = flow_loss(model, sample(), torch.Generator(), torch.device('cpu'), torch.bfloat16)
        loss.backward()
        opt.step()
        self.assertTrue(all(p.dtype == torch.float32 for p in params))
        self.assertTrue(all(torch.isfinite(p).all() for p in params))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, params)))

    def test_actual_vae_still_and_video_temporal_shapes(self):
        vae = AutoencoderKLWan(base_dim=8, z_dim=4, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
                             latents_mean=[0]*4, latents_std=[1]*4).eval()
        with torch.no_grad():
            for frames in (1, 17):
                x = torch.zeros(1, 3, frames, 32, 32)
                z = vae.encode(x).latent_dist.mode()
                y = vae.decode(z).sample
                self.assertEqual(z.shape[2], (frames - 1)//4 + 1)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all())

    def test_resume_matches_uninterrupted_next_update_and_inference_loads(self):
        with tempfile.TemporaryDirectory() as folder:
            model = tiny_model()
            opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
            rng = torch.Generator().manual_seed(9)
            update(model, opt, rng)
            config = {'test': True, 'source': SOURCE}
            saved = save_checkpoint(folder, model, opt, rng, 1, config, .5, {}, True)
            expected_loss = update(model, opt, rng)
            expected = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
            clone = tiny_model()
            opt2 = torch.optim.AdamW((p for p in clone.parameters() if p.requires_grad), lr=.001)
            rng2 = torch.Generator()
            state = restore_checkpoint(saved, clone, opt2, rng2, config)
            self.assertEqual(state['step'], 1)
            self.assertAlmostEqual(update(clone, opt2, rng2), expected_loss, places=7)
            self.assertTrue(all(torch.equal(v, expected[k]) for k, v in get_peft_model_state_dict(clone).items()))
            saved2 = save_checkpoint(folder, clone, opt2, rng2, 2, config, .4, {}, True)
            pipeline = WanPipeline(tokenizer=None, text_encoder=None, vae=None,
                transformer=tiny_model(False), scheduler=FlowMatchEulerDiscreteScheduler())
            pipeline.load_lora_weights(str(saved2))
            clone.eval()
            pipeline.transformer.eval()
            with torch.no_grad():
                inp = sample()
                a = clone(inp['latent'], torch.tensor([500.]), inp['prompt_embeds'], return_dict=False)[0]
                b = pipeline.transformer(inp['latent'], torch.tensor([500.]), inp['prompt_embeds'], return_dict=False)[0]
                torch.testing.assert_close(a, b)
                latents = pipeline(prompt_embeds=inp['prompt_embeds'], guidance_scale=1.,
                    height=32, width=32, num_frames=5, num_inference_steps=2,
                    generator=torch.Generator().manual_seed(5), output_type='latent').frames
                self.assertEqual(tuple(latents.shape), (1, 4, 2, 4, 4))
                self.assertTrue(torch.isfinite(latents).all())

    def test_corrupt_latest_falls_back_and_no_silent_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            model = tiny_model()
            opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad))
            rng = torch.Generator()
            first = save_checkpoint(folder, model, opt, rng, 0, {}, 1., {}, True)
            second = save_checkpoint(folder, model, opt, rng, 1, {}, 1., {}, False)
            (second / 'state.pt').write_bytes(b'partial upload')
            self.assertEqual(find_checkpoint(folder, {}), first)
            with self.assertRaises(ValueError):
                find_checkpoint(folder, {'different': True})
            (first / 'pytorch_lora_weights.safetensors').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'silent'):
                find_checkpoint(folder, {})

    def test_rotation_preserves_zero_best_and_latest_three(self):
        with tempfile.TemporaryDirectory() as folder:
            model = tiny_model()
            opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad))
            for step in range(7):
                save_checkpoint(folder, model, opt, torch.Generator(), step, {}, 1., {}, step in (0, 1))
            prune_checkpoints(folder)
            steps = [verify_checkpoint(p)[0]['step'] for p in (Path(folder) / 'snapshots').iterdir()]
            self.assertEqual(sorted(steps), [0, 1, 4, 5, 6])

    def test_cache_detects_hash_corruption_and_split_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for group in ('appearance', 'motion', 'validation'):
                file = group + '.pt'
                torch.save(sample(), root / file)
                rows.append(dict(file=file, group=group, source=group, sha256=digest(root / file)))
            manifest = dict(source=SOURCE, version=VERSION, records=rows)
            write_json(root / 'manifest.json', manifest)
            self.assertEqual(len(read_cache(root)[0]['motion']), 1)
            rows[2]['source'] = 'motion'
            write_json(root / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'leaks'):
                read_cache(root)
            rows[2]['source'] = 'validation'
            write_json(root / 'manifest.json', manifest)
            (root / 'appearance.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                read_cache(root)


if __name__ == '__main__':
    unittest.main()
