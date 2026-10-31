"""Invariants for training and evaluating a change in the meaning of a prompt."""
from collections import Counter, defaultdict
import io
import json
import re
import unittest
import sentencepiece as spm
from chat_recovery_data import prompt_text
from coherence_quality import _bank, grade_answer, normalized
import paired_instruction_data as data


class PairedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = io.BytesIO()
        spm.SentencePieceTrainer.train(sentence_iterator=iter([
            'A small Python function. Please return a number. def add x.',
            'A note lists my book in the kitchen. What is the sum of two and three?',
        ] * 20), model_writer=model, model_type='bpe', vocab_size=360, byte_fallback=True,
            hard_vocab_limit=False, normalization_rule_name='identity', add_dummy_prefix=False,
            remove_extra_whitespaces=False, user_defined_symbols=['<|general|>', '<|code|>'], minloglevel=2)
        cls.tokenizer = spm.SentencePieceProcessor(model_proto=model.getvalue())
        cls.suite = data.build(cls.tokenizer, 512)

    def test_counts_disjoint_inputs_wordings_and_shared_eval(self):
        self.assertEqual(len(self.suite['train']), 1024)
        all_prompts, all_ids, previous_inputs, previous_wordings = set(), set(), set(), set()
        protected = {normalized(i['prompt']) for split in ('dev','final') for i in _bank(split)}
        for split in ('train','dev','final'):
            if split == 'train':
                by_pair = defaultdict(list)
                for r in self.suite['train']: by_pair[r['pair_id']].append(r)
                pairs = list(by_pair.values())
            else:
                pairs = [p['items'] for p in self.suite[split]]
            input_keys, wording_keys = set(), set()
            counts = Counter()
            for pair in pairs:
                self.assertEqual(len(pair), 2)
                first, second = pair
                self.assertEqual(first['pair_id'], second['pair_id'])
                self.assertEqual(first['input_key'], second['input_key'])
                self.assertNotIn(first['input_key'], input_keys)
                input_keys.add(first['input_key'])
                self.assertEqual(first['wording_id'], second['wording_id'])
                tmpl = int(first['wording_id'].rsplit('/',1)[1])
                wording_keys.add(data.WORDING[first['family']][split][tmpl])
                counts[first['family']] += 1
                for index, item in enumerate(pair):
                    self.assertEqual(item['split'], split)
                    question = normalized(item['prompt'])
                    self.assertNotIn(question, all_prompts | protected)
                    all_prompts.add(question)
                    self.assertNotIn(item['id'], all_ids)
                    all_ids.add(item['id'])
                    self.assertTrue(grade_answer(item, item['reference'])['passed'])
                    self.assertFalse(grade_answer(item, pair[1-index]['reference'])['passed'])
            self.assertEqual(counts, {name: 128 if split=='train' else 3 for name in data.FAMILIES})
            self.assertFalse(previous_inputs & input_keys)
            self.assertFalse(previous_wordings & wording_keys)
            previous_inputs.update(input_keys)
            previous_wordings.update(wording_keys)

    def test_reference_semantics_across_all_splits(self):
        rows = self.suite['train'] + [r for s in ('dev','final') for p in self.suite[s] for r in p['items']]
        for row in rows:
            args = json.loads(row['input_key'].split(':',1)[1]); prompt = row['prompt'].lower()
            if row['family'] == 'arithmetic':
                a,b = args
                if '+' in prompt or 'sum of' in prompt: expected = a+b
                elif '-' in prompt or 'minus' in prompt: expected = a-b
                elif '*' in prompt or 'product of' in prompt: expected = a*b
                else: self.fail('Arithmetic operation missing from prompt')
                self.assertEqual(row['reference'], str(expected))
            elif row['family'] == 'string':
                value, = args
                if 'uppercase' in prompt: expected = value.upper()
                elif 'lowercase' in prompt: expected = value.lower()
                elif 'reverse' in prompt: expected = value[::-1]
                else: self.fail('String operation missing from prompt')
                self.assertEqual(row['reference'], expected)
            elif row['family'] == 'grounded':
                a,b,p,q = args
                for word in (a,b,p,q): self.assertRegex(prompt,r'\b'+word+r'\b')
                self.assertEqual(row['reference'], p if row['side']==0 else q)
            else:
                name,k,var = args
                self.assertEqual(row['rubric']['function'], name)
                self.assertEqual(row['rubric']['args'], [var])
                self.assertIn(str(k), prompt)
                for case in row['rubric']['cases']:
                    x, = case['args']
                    if 'input plus' in prompt: expected = x+k
                    elif 'input minus' in prompt: expected = x-k
                    elif 'multiplied by' in prompt: expected = x*k
                    else: self.fail('Code edit missing from prompt')
                    self.assertEqual(case['expected'],expected)

    def test_training_target_ids_and_eos(self):
        for row in self.suite['train']:
            prefix = prompt_text(row['messages'][:-1]); n = len(self.tokenizer.encode(prefix))
            x, y = row['input_ids'], row['labels']
            self.assertEqual(self.tokenizer.decode(x), prefix + row['reference'])
            self.assertEqual(y[:n-1], [-100]*(n-1))
            self.assertEqual(y[n-1:], x[n:]+[self.tokenizer.eos_id()])
            self.assertGreaterEqual(row['max_tokens'],len(y)-n+1)
            self.assertLessEqual(n+row['max_tokens'],512)

    def test_exclusion_fails_closed_and_determinism(self):
        overlap = self.suite['train'][0]['prompt'].upper().replace(' ', '  ')
        with self.assertRaisesRegex(ValueError, 'Duplicate or excluded'):
            data.build(self.tokenizer,512,excluded_prompts=[overlap])
        self.assertEqual(self.suite, data.build(self.tokenizer,512))
        with self.assertRaises(ValueError): data.build(self.tokenizer,4)


if __name__ == '__main__': unittest.main()
