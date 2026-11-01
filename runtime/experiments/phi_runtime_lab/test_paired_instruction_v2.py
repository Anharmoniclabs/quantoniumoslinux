"""Design and target checks for the paired transfer benchmark; no model training."""

import json
import operator
import unittest
from collections import Counter, defaultdict

import paired_instruction_data as v1
import paired_instruction_v2 as v2
from chat_recovery_data import prompt_text
from coherence_quality import _bank, grade_answer, normalized


class CharacterTokenizer:
    def encode(self, text):
        return [ord(char) + 3 for char in text]

    def decode(self, ids):
        return ''.join(chr(token - 3) for token in ids)

    def eos_id(self):
        return 2


def fields(item):
    return json.loads(item['input_key'].split(':', 1)[1])


def input_identity(item):
    value = fields(item)
    family = item['family']
    if family == 'arithmetic':
        value = sorted(value)
    elif family == 'grounded':
        value = sorted(value[:2]) + sorted(value[2:])
    elif family == 'code':
        value = value[:2]
    return family, tuple(value)


def items(suite, split):
    if split == 'train':
        return suite[split]
    return [item for pair in suite[split] for item in pair['items']]


def atoms(family, item):
    value = fields(item)
    return set(value[0].lower().split('-') if family == 'string' else value)


class PairedTransferV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = CharacterTokenizer()
        cls.broad_prompts = {q['prompt'] for s in ('dev', 'final') for q in _bank(s)}
        cls.suite = v2.build(cls.tokenizer, 512, excluded_prompts=cls.broad_prompts)
        cls.previous = v1.build(cls.tokenizer, 512)

    def test_crossed_cells_and_deterministic_counts(self):
        self.assertEqual(len(self.suite['train']), 1024)
        expected = {(f, i, w): 12 for f in v2.FAMILIES for i in v2.REGIMES for w in v2.REGIMES}
        for split in ('dev', 'final'):
            pairs = self.suite[split]
            self.assertEqual(len(pairs), 192)
            self.assertEqual(len({p['pair_id'] for p in pairs}), len(pairs))
            self.assertEqual(
                Counter(
                    (p['family'], p['strata']['input_regime'], p['strata']['wording_regime']) for p in pairs
                ),
                expected,
            )
        self.assertEqual(self.suite, v2.build(self.tokenizer, 512, self.broad_prompts))

    def test_inputs_prompts_and_qualification_sets_are_disjoint(self):
        previous = [i for split in self.previous for i in items(self.previous, split)]
        forbidden = {input_identity(i) for i in previous}
        prompts = {normalized(i['prompt']) for i in previous} | {normalized(p) for p in self.broad_prompts}
        all_ids = set()
        for split in ('train', 'dev', 'final'):
            split_items = items(self.suite, split)
            identities = {input_identity(i) for i in split_items}
            self.assertEqual(len(identities) * 2, len(split_items))
            self.assertTrue(identities.isdisjoint(forbidden))
            forbidden.update(identities)
            for item in split_items:
                self.assertNotIn(normalized(item['prompt']), prompts)
                self.assertNotIn(item['id'], all_ids)
                prompts.add(normalized(item['prompt']))
                all_ids.add(item['id'])
        with self.assertRaisesRegex(ValueError, 'Duplicate or excluded'):
            v2.build(self.tokenizer, 512, [self.suite['dev'][-1]['items'][0]['prompt'].upper()])
        with self.assertRaises(ValueError):
            v2.build(self.tokenizer, 8)

    def test_familiarity_is_measured_against_actual_train_atoms_and_templates(self):
        known, templates = defaultdict(set), set()
        for item in self.suite['train']:
            known[item['family']].update(atoms(item['family'], item))
            templates.add(item['wording_id'])
        novel_templates = {}
        for split in ('dev', 'final'):
            novel_templates[split] = set()
            for pair in self.suite[split]:
                family = pair['family']
                for item in pair['items']:
                    self.assertEqual(item['strata'], pair['strata'])
                    is_familiar = pair['strata']['input_regime'] == 'familiar'
                    value = atoms(family, item)
                    if is_familiar:
                        self.assertTrue(value <= known[family], (split, item['input_key']))
                    elif family == 'code':
                        name, k, var = fields(item)
                        self.assertTrue({name, var} <= known[family])
                        self.assertNotIn(k, known[family])
                    else:
                        self.assertTrue(value.isdisjoint(known[family]))
                    familiar_wording = pair['strata']['wording_regime'] == 'familiar'
                    self.assertEqual(item['wording_id'] in templates, familiar_wording)
                    if not familiar_wording:
                        novel_templates[split].add(item['wording_id'])
        self.assertTrue(novel_templates['dev'].isdisjoint(novel_templates['final']))
        for family in v2.FAMILIES:
            trained = set(v1.WORDING[family]['train'])
            new_dev = set(v2.NOVEL_WORDING['dev'][family])
            new_final = set(v2.NOVEL_WORDING['final'][family])
            self.assertTrue(new_dev.isdisjoint(new_final | trained))
            self.assertTrue(new_final.isdisjoint(trained))

    def test_arithmetic_distribution_targets_and_code_behavior(self):
        math = {'+': operator.add, '-': operator.sub, '*': operator.mul}
        for split in ('dev', 'final'):
            all_sub = Counter()
            for input_regime in v2.REGIMES:
                for wording_regime in v2.REGIMES:
                    operation_counts = defaultdict(Counter)
                    sign, carry, borrow = Counter(), Counter(), Counter()
                    for p in self.suite[split]:
                        if p['strata'] != {'input_regime': input_regime, 'wording_regime': wording_regime}:
                            continue
                        if p['family'] not in ('arithmetic', 'code'):
                            continue
                        left, right = p['items']
                        self.assertNotEqual(left['operation'], right['operation'])
                        self.assertEqual(fields(left), fields(right))
                        for item in p['items']:
                            op = item['operation']
                            operation_counts[p['family']][op] += 1
                            if item['family'] == 'arithmetic':
                                a, b = fields(item)
                                answer = math[op](a, b)
                                self.assertEqual(str(answer), item['reference'])
                                verbs = {'+': 'plus', '-': 'minus', '*': 'multiplied by'}
                                self.assertTrue(
                                    f'{a} {op} {b}' in item['prompt']
                                    or f'{a} {verbs[op]} {b}' in item['prompt']
                                )
                                if op == '+':
                                    carries = a % 10 + b % 10 >= 10
                                    self.assertEqual(carries, p['features']['add_carry'])
                                    carry[carries] += 1
                                if op == '-':
                                    result_sign = 'positive' if a > b else 'negative' if a < b else 'zero'
                                    self.assertEqual(result_sign, p['features']['subtract_sign'])
                                    sign[result_sign] += 1
                                    all_sub[answer] += 1
                                    if a != b:
                                        needs_borrow = max(a, b) % 10 < min(a, b) % 10
                                        self.assertEqual(needs_borrow, p['features']['subtract_borrow'])
                                        borrow[needs_borrow] += 1
                            else:
                                name, k, var = fields(item)
                                rubric = item['rubric']
                                self.assertEqual((rubric['function'], rubric['args']), (name, [var]))
                                verbs = {'+': 'plus', '-': 'minus', '*': 'multiplied by'}
                                self.assertIn(f'return the input {verbs[op]} {k}', item['prompt'])
                                for case in rubric['cases']:
                                    self.assertEqual(case['expected'], math[op](case['args'][0], k))
                    for family in ('arithmetic', 'code'):
                        self.assertEqual(operation_counts[family], {'+': 8, '-': 8, '*': 8})
                    self.assertEqual(sign, {'positive': 3, 'negative': 3, 'zero': 2})
                    self.assertEqual(carry, {False: 4, True: 4})
                    self.assertEqual(borrow, {False: 3, True: 3})
            self.assertEqual(all_sub.pop(0), 8)
            self.assertEqual(len(all_sub), 24)
            self.assertTrue(all(n == 1 for n in all_sub.values()))

    def test_pair_grading_and_grounded_string_counterfactuals(self):
        for split in ('dev', 'final'):
            for pair in self.suite[split]:
                left, right = pair['items']
                self.assertEqual([left['side'], right['side']], [0, 1])
                self.assertNotEqual(left['reference'], right['reference'])
                for own, other in ((left, right), (right, left)):
                    self.assertTrue(grade_answer(own, own['reference'])['passed'])
                    self.assertFalse(grade_answer(own, other['reference'])['passed'])
                    if own['family'] == 'string':
                        s = fields(own)[0]
                        edits = {'lower': s.lower(), 'upper': s.upper(), 'reverse': s[::-1]}
                        self.assertEqual(own['reference'], edits[own['operation']])
                        self.assertIn(s, own['prompt'])
                        self.assertIn(
                            own['operation']
                            if own['operation'] == 'reverse'
                            else own['operation'] + 'case',
                            own['prompt'].lower(),
                        )
                    elif own['family'] == 'grounded':
                        a, b, p, q = fields(own)
                        for term in (a, b, p, q):
                            self.assertIn(term, own['prompt'])
                        self.assertSetEqual({left['reference'], right['reference']}, {p, q})
                        word_group, _, index = own['wording_id'].split('/')
                        bank = (
                            v1.WORDING['grounded']['train']
                            if word_group == 'train'
                            else v2.NOVEL_WORDING[split]['grounded']
                        )
                        pattern = bank[int(index)]
                        possibilities = [(a, p, q, p)]
                        if own['operation'] == 'swap_fact':
                            possibilities.append((a, q, p, q))
                        else:
                            possibilities.append((b, p, q, q))
                        expected = {
                            pattern.format(a=a, b=b, p=loc_a, q=loc_b, ask=ask): gold
                            for ask, loc_a, loc_b, gold in possibilities
                        }
                        self.assertEqual(expected[own['prompt']], own['reference'])
                self.assertFalse(all(grade_answer(x, left['reference'])['passed'] for x in pair['items']))

    def test_training_rows_mask_exact_answer_through_eos(self):
        for row in self.suite['train']:
            x, y = row['input_ids'], row['labels']
            prefix = self.tokenizer.encode(prompt_text(row['messages'][:-1]))
            n = len(prefix)
            self.assertEqual(x[:n], prefix)
            self.assertEqual(y[: n - 1], [-100] * (n - 1))
            self.assertEqual(y[n - 1 :], x[n:] + [self.tokenizer.eos_id()])
            self.assertEqual(self.tokenizer.decode(x[n:]), row['reference'])
            self.assertTrue(len(x) - n + 1 <= row['max_tokens'] <= 512 - n)


if __name__ == '__main__':
    unittest.main()
