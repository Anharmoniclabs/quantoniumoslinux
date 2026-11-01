"""Batched decoder regressions including a real saved mini PhiState model."""
import io
from pathlib import Path
from contextlib import redirect_stdout
import tempfile
import unittest
from unittest.mock import patch

import sentencepiece as spm
import torch

import a100_greedy_batch as batch
from chat_recovery_data import prompt_text
from coherence_quality import generate_answer, generation_prompt_text
from moe_model import PhiStateMoELM, config_for
from moe_train import ckpt_load


class ToyTokenizer:
    codes = {'short': [1], 'long': [2, 4, 6], 'stop': [2], 'bad': [4]}

    def encode(self, text):
        for prompt, ids in self.codes.items():
            if text == prompt_text([dict(role='user', content=prompt)]):
                return ids
        raise ValueError('Incorrect chat template')

    def eos_id(self): return 9

    def decode(self, ids): return ','.join(str(i) for i in ids)


class ToyModel(torch.nn.Module):
    def __init__(self, bad_token=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.bad_token = bad_token

    def step(self, token, cache):
        cache = [] if cache is None else cache
        outputs = torch.arange(10, device=token.device)
        # 1 -> 2 -> EOS. 4 -> 5. 6 -> 7 -> 8 -> EOS. Rows which ended
        # keep feeding EOS, on which nonfinite logits must be ignored.
        lookup = torch.tensor([1, 2, 9, 4, 5, 6, 7, 8, 9, 9], device=token.device)
        logits = (outputs == lookup[token].unsqueeze(-1)).float()
        logits[token.flatten() == 9] = float('nan')
        if self.bad_token is not None:
            logits[token.flatten() == self.bad_token] = float('nan')
        return logits, cache


class GreedyBatchTests(unittest.TestCase):
    def test_real_turns_form_one_training_template(self):
        turns = [dict(role='user', content='Remember code red.'), dict(role='assistant', content='I will.'),
                 dict(role='user', content='Which code?')]
        self.assertEqual(generation_prompt_text(turns), '<|general|>\nUser:\nRemember code red.\n'
                         'Assistant:\nI will.\nUser:\nWhich code?\nAssistant:\n')
        self.assertEqual(generation_prompt_text('short'), generation_prompt_text([dict(role='user', content='short')]))
        for bad in [[], turns[:-1], turns[1:], [dict(role='user', content=42)],
                    [dict(role='user', content='a'), dict(role='user', content='b')]]:
            with self.assertRaises(ValueError): generation_prompt_text(bad)
        with self.assertRaisesRegex(ValueError,'exactly one'):
            batch.generate_batch(ToyModel(), ToyTokenizer(), [dict(prompt='q',messages=turns,max_tokens=1)])

    def test_different_lengths_limits_eos_and_order(self):
        items = [dict(prompt=q, max_tokens=k) for q,k in [('long',2),('stop',7),('short',6),('long',7),('short',1)]]
        tokenizer = ToyTokenizer(); model = ToyModel()
        expected = [generate_answer(model, tokenizer, i['prompt'], max_tokens=i['max_tokens']) for i in items]
        self.assertEqual(expected[0], dict(text='7,8', tokens=2, eos=False))
        self.assertEqual(expected[1], dict(text='', tokens=0, eos=True))
        self.assertEqual(expected[2], dict(text='2', tokens=1, eos=True))
        self.assertEqual(expected[3], dict(text='7,8', tokens=2, eos=True))
        rng = torch.get_rng_state().clone()
        for size in (1,2,5):
            self.assertEqual(batch.generate_batch(model, tokenizer, items, size, verify=2), expected)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_actual_mini_checkpoint_matches_single_row(self):
        with tempfile.TemporaryDirectory() as temp:
            tokenizer_bytes = io.BytesIO()
            spm.SentencePieceTrainer.train(sentence_iterator=iter(['A short question and answer in Python.']*20),
                model_writer=tokenizer_bytes, model_type='bpe', vocab_size=350, byte_fallback=True,
                hard_vocab_limit=False, normalization_rule_name='identity', add_dummy_prefix=False,
                remove_extra_whitespaces=False, user_defined_symbols=['<|general|>','<|code|>'], minloglevel=2)
            tokenizer = spm.SentencePieceProcessor(model_proto=tokenizer_bytes.getvalue())
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(991)
                cfg = config_for('mini', tokenizer.vocab_size()); model = PhiStateMoELM(cfg)
            checkpoint = Path(temp)/'mini.pt'
            torch.save(dict(format='PhiStateMoE-v1', config=vars(cfg), state_dict=model.state_dict()), checkpoint)
            model, _ = ckpt_load(str(checkpoint))
            items = [dict(prompt=q, max_tokens=k) for q,k in [('Hi',7), ('Return only a number: 2 + 3.',4),
                ('Write Python code: add two numbers, then return their sum.',9), ('Hi',1)]]
            for value in ('red','blue'):
                items.append(dict(messages=[dict(role='user',content='My code is '+value+'.'),
                                            dict(role='assistant',content='Understood.'),
                                            dict(role='user',content='What was my code?')], max_tokens=6))
            expected = [generate_answer(model, tokenizer, i.get('prompt',i.get('messages')), max_tokens=i['max_tokens']) for i in items]
            actual = batch.generate_batch(model, tokenizer, items, batch_size=4, verify=len(items))
            self.assertEqual(expected, actual)
            too_long = [dict(role='user',content='a '*800), dict(role='assistant',content='Okay.'),
                        dict(role='user',content='Which letter?')]
            with self.assertRaisesRegex(ValueError,'512'):
                batch.generate_batch(model, tokenizer, [dict(messages=too_long,max_tokens=4)])
            # Force a mismatch during verification only. Every actual fallback
            # call must receive the original single string or list of turns.
            inputs = []
            def forced(m,t,prompt,max_tokens):
                inputs.append(prompt)
                if len(inputs)==1: return dict(text='forced mismatch',tokens=1,eos=True)
                return generate_answer(m,t,prompt,max_tokens=max_tokens)
            with patch.object(batch,'generate_answer',side_effect=forced), redirect_stdout(io.StringIO()):
                recovered = batch.generate_batch(model,tokenizer,items,batch_size=4,verify=1,serial_on_mismatch=True)
            self.assertEqual(recovered, expected)
            self.assertEqual(inputs[1:], [i.get('prompt',i.get('messages')) for i in items])

    def test_bad_logits_and_parity_failure_are_hard_errors(self):
        with self.assertRaisesRegex(FloatingPointError, 'Nonfinite'):
            batch.generate_batch(ToyModel(4), ToyTokenizer(), [dict(prompt='long', max_tokens=3)])
        model = ToyModel()
        with patch.object(batch, 'generate_answer', return_value=dict(text='changed',tokens=1,eos=False)):
            with self.assertRaisesRegex(RuntimeError,'single-row greedy generation mismatch'):
                batch.generate_batch(model, ToyTokenizer(), [dict(prompt='short',max_tokens=3)], verify=1)
        self.assertTrue(model.training)

    def test_input_validation_and_precision(self):
        model = ToyModel()
        with self.assertRaisesRegex(ValueError, 'FP32'):
            batch.generate_batch(model.double(), ToyTokenizer(), [dict(prompt='short',max_tokens=3)])
        with self.assertRaisesRegex(ValueError, 'limit'):
            batch.generate_batch(model, ToyTokenizer(), [dict(prompt='short',max_tokens=300)])
        self.assertEqual(batch.generate_batch(model, ToyTokenizer(), []), [])

    def test_parity_fallback_recomputes_every_row_in_original_order(self):
        model, tokenizer = ToyModel(), ToyTokenizer()
        items = [dict(prompt=q, max_tokens=k) for q,k in [('long',3),('short',1),('stop',2)]]
        answers = {item['prompt']: dict(text='serial '+item['prompt'], tokens=1, eos=True) for item in items}
        calls = []
        def serial_reference(model, tokenizer, prompt, max_tokens):
            calls.append((prompt,max_tokens))
            return answers[prompt]
        log = io.StringIO()
        with patch.object(batch,'generate_answer',side_effect=serial_reference), redirect_stdout(log):
            result = batch.generate_batch(model,tokenizer,items,verify=2,serial_on_mismatch=True)
        self.assertIn('BATCH_PARITY_MISMATCH',log.getvalue())
        self.assertEqual(result,[answers[item['prompt']] for item in items])
        self.assertEqual(result.decode_backend,'serial_after_batch_parity_mismatch')
        self.assertEqual(len(calls),1+len(items))
        self.assertEqual(calls[1:],[(item['prompt'],item['max_tokens']) for item in items])
        with self.assertRaisesRegex(FloatingPointError,'Nonfinite'):
            batch.generate_batch(ToyModel(4),tokenizer,[dict(prompt='long',max_tokens=3)],verify=1,serial_on_mismatch=True)
        with patch.object(batch,'generate_answer',side_effect=RuntimeError('CUDA out of memory')):
            with self.assertRaisesRegex(RuntimeError,'out of memory'):
                batch.generate_batch(model,tokenizer,items,verify=1,serial_on_mismatch=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
