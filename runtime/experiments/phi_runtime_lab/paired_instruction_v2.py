"""Counterfactual instruction pairs with a crossed evaluation design.

Every evaluation input is a fresh combination. In the familiar-input condition,
its components occurred individually in training. In the novel condition the
numbers (or string/grounded vocabulary) are new. Code keeps names and parameter
names familiar in both conditions and varies the range of constants. Familiar
wording reuses an exact training template. Novel wording is exclusive to its
evaluation split. Arithmetic inputs are nonnegative two-digit integers. Sign
balance concerns the subtraction answer. Borrow concerns its absolute magnitude.
Broad prompt exclusions are normalized exact matches. Score both answers, then
count a pair only if both pass.
"""

import random
from collections import Counter
from itertools import combinations, product

import paired_instruction_data as v1
from chat_recovery_data import canonical, digest, encode_example, prompt_text
from coherence_quality import grade_answer, normalized

VERSION = 'phistate-paired-instruction-v2'
FAMILIES = v1.FAMILIES
COUNTS = {'train': 128, 'dev': 48, 'final': 48}  # pairs per family
REGIMES = ('familiar', 'novel')
OPS = (('+', '-'), ('+', '*'), ('-', '*'))
STRING_OPS = (
    'Convert all letters to uppercase',
    'Convert all letters to lowercase',
    'Reverse all characters, preserving their case',
)

# Twelve pairs per evaluation cell. Each operator occurs eight times per cell.
# For the eight subtractions, the target is negative three times, positive
# three times and zero twice. Exactly half of the nonzero absolute differences
# need a ones-column borrow. Half of the eight additions need a units carry.
SUB = {
    0: ('negative', False),
    2: ('positive', True),
    3: ('positive', False),
    5: ('negative', True),
    6: ('zero', False),
    8: ('positive', False),
    9: ('negative', True),
    11: ('zero', False),
}
NUMBERS = {'train': range(10, 60), 'dev': range(60, 80), 'final': range(80, 100)}
WORDS = {
    'train': (
        'red blue green gold silver white black pink warm cool soft loud bright pale tall short thin dusty dry smooth plain deep shy loose',
        'pear lamp bird cup book coat door star brush leaf desk clock glass pen shell kite',
    ),
    'dev': ('bronze brisk blunt fine gray rich slim wide', 'fern bark cliff ford arch post rake vine'),
    'final': ('bold calm faint keen matte mild neat oval', 'twig seed pond hill pipe rope fork gate'),
}
THINGS = {
    'train': (
        'bag hat coin scarf rope note mug pen shoe key sock map pin card cap comb',
        'porch garage barn shed kitchen office bedroom attic pantry closet',
    ),
    'dev': ('bell watch ring bowl fork block badge tag', 'deck foyer vault gallery lobby loft den cellar'),
    'final': (
        'cable flask spoon marble stamp toy drill tile',
        'hall studio shop lab patio yard cabin tent',
    ),
}
NAMES = ['adjust', 'update', 'shift', 'change', 'offset', 'scale', 'move', 'revise']
PARAMS = ('x', 'n', 'value')

NOVEL_WORDING = {
    'dev': {
        'arithmetic': (
            'For my receipt I need the value of {expr}. Write just the integer.',
            'Give the integer for {words}. Omit any working.',
        ),
        'string': (
            'Edit this label: {value}. {action}. Supply only the edited label.',
            'The tag is {value}. Please {action_lower}. Output the resulting text alone.',
        ),
        'grounded': (
            'The {p} contains the {a} and the {q} contains the {b}. Give only the location of the {ask}.',
            'I logged the {b} at the {q} and the {a} at the {p}. Where is the {ask}? Only the location.',
        ),
        'code': (
            'Correct the return expression below so it will {action}. Preserve the signature.\n{start}\nOutput the Python definition.',
            'This Python helper needs to {action}. Repair the definition and return code alone.\n{start}',
        ),
    },
    'final': {
        'arithmetic': (
            'The entry on my form is {expr}. Fill in its integer value alone.',
            'Write the numerical answer to {words}, with no other text.',
        ),
        'string': (
            'Here is the label {value}. {action}. Give me just the new label.',
            'Apply one change to {value}: {action_lower}. Respond with only the changed text.',
        ),
        'grounded': (
            'On my card: {b} in the {q}, {a} in the {p}. State where the {ask} is, in one place name.',
            'The {a} was recorded in the {p}, the {b} in the {q}. I need the location of the {ask} alone.',
        ),
        'code': (
            'Make this definition {action}. Leave its name and parameters alone. Reply with the corrected Python.\n{start}',
            'My helper below must {action}. Rewrite its Python definition without commentary.\n{start}',
        ),
    },
}


def _key(family, fields):
    """Block reversed commutative operands, swapped facts and renamed code args."""
    if family == 'arithmetic':
        return tuple(sorted(fields))
    if family == 'grounded':
        a, b, p, q = fields
        return (*sorted((a, b)), *sorted((p, q)))
    if family == 'code':
        return tuple(fields[:2])  # (function name, constant), irrespective of arg name
    return tuple(fields)


def _prior_inputs():
    blocked = {family: set() for family in FAMILIES}
    for split, count in v1.COUNTS.items():
        for family, factory in zip(FAMILIES, (v1._arithmetic, v1._string, v1._grounded, v1._code)):
            for i in range(count):
                fields, _ = factory(split, i)
                blocked[family].add(_key(family, fields))
    return blocked


def _components(family, fields):
    if family == 'string':
        return tuple(fields[0].lower().split('-'))
    return tuple(fields)


def _pool(family, pool, rng):
    ns = NUMBERS[pool]
    if family == 'arithmetic':
        out = list(product(ns, repeat=2))
    elif family == 'code':
        out = [(name, n, rng.choice(PARAMS)) for name, n in product(NAMES, ns)]
    elif family == 'string':
        aa, bb = (s.split() for s in WORDS[pool])
        out = [(a.capitalize() + '-' + b.capitalize(),) for a, b in product(aa, bb)]
    else:
        objs, locs = (s.split() for s in THINGS[pool])
        out = []
        for (a, b), (p, q) in product(combinations(objs, 2), combinations(locs, 2)):
            if rng.randrange(2):
                a, b = b, a
            if rng.randrange(2):
                p, q = q, p
            out.append((a, b, p, q))
    rng.shuffle(out)
    return out


def _arithmetic_match(fields, slot):
    a, b = fields
    if '+' in OPS[slot % 3] and (a % 10 + b % 10 >= 10) != bool((slot // 3) % 2):
        return False
    if slot in SUB:
        sign = 'positive' if a > b else 'negative' if a < b else 'zero'
        borrow = max(a, b) % 10 < min(a, b) % 10
        return (sign, borrow) == SUB[slot]
    return True


def _choose(family, pool, blocked, familiar, slot, subtraction_use):
    choices = (
        f
        for f in pool
        if _key(family, f) not in blocked and (familiar is None or set(_components(family, f)) <= familiar)
    )
    if family == 'arithmetic':
        choices = (f for f in choices if _arithmetic_match(f, slot))
        if slot in SUB:
            # Spread the numerical answers instead of teaching a constant
            # difference. Ties keep their deterministic shuffled order.
            return min(choices, key=lambda f: subtraction_use[f[0] - f[1]])
    return next(choices)


def _variants(family, fields, slot):
    arithmetic, verbs = OPS[slot % 3], ('plus', 'minus', 'multiplied by')
    if family == 'arithmetic':
        a, b = fields
        rows = []
        for op in arithmetic:
            i = ('+', '-', '*').index(op)
            ans = (a + b, a - b, a * b)[i]
            rows.append(({'expr': f'{a} {op} {b}', 'words': f'{a} {verbs[i]} {b}'}, op, str(ans), None))
    elif family == 'string':
        (value,) = fields
        rows = []
        for idx in ((0, 1), (0, 2), (1, 2))[slot % 3]:
            action = STRING_OPS[idx]
            expected = (value.upper(), value.lower(), value[::-1])[idx]
            rows.append(
                (
                    {'value': value, 'action': action, 'action_lower': action[0].lower() + action[1:]},
                    ('upper', 'lower', 'reverse')[idx],
                    expected,
                    None,
                )
            )
    elif family == 'grounded':
        a, b, p, q = fields
        base = {'a': a, 'b': b, 'p': p, 'q': q, 'ask': a}
        swap_fact = slot % 2 == 1
        other = dict(base, p=q, q=p) if swap_fact else dict(base, ask=b)
        op = 'swap_fact' if swap_fact else 'change_query'
        rows = [(base, op, p, None), (other, op, q, None)]
    else:
        name, k, var = fields
        signature = f'{name}({var})'
        start = f'def {signature}:\n    return {var}'
        rows = []
        for op in arithmetic:
            i = ('+', '-', '*').index(op)
            action = f'return the input {verbs[i]} {k}'
            gold = f'```python\ndef {signature}:\n    return {var} {op} {k}\n```'
            cases = [{'args': [n], 'expected': (n + k, n - k, n * k)[i]} for n in (-7, 0, 3, 11)]
            rows.append(
                (
                    {'start': start, 'signature': signature, 'var': var, 'action': action},
                    op,
                    gold,
                    {'kind': 'code', 'function': name, 'args': [var], 'cases': cases},
                )
            )
    if (slot // 3) % 2:
        rows.reverse()
    return rows


def build(tokenizer, context, excluded_prompts=()):
    """Use V1 trainer/evaluator row shapes. Caller supplies all broad exclusions."""
    if type(context) is not int or context < 1:
        raise ValueError('context must be a positive integer')
    if isinstance(excluded_prompts, str):
        raise TypeError('excluded_prompts must be an iterable of strings')
    seen = {normalized(p) for p in excluded_prompts}
    used = _prior_inputs()
    result = {split: [] for split in COUNTS}
    for family in FAMILIES:
        rng = random.Random(VERSION + '/' + family)
        pools = {split: _pool(family, split, rng) for split in COUNTS}
        known = set()
        for split, count in COUNTS.items():
            subtraction_use = Counter()
            for i in range(count):
                cell, slot = divmod(i, 12)
                ev = split != 'train'
                in_new = ev and cell >= 2
                phrase_new = ev and cell % 2 == 1
                domain = split if in_new else 'train'
                wording = NOVEL_WORDING[split][family] if phrase_new else v1.WORDING[family]['train']
                # Randomized templates are balanced independently of the operation.
                if slot == 0:
                    template_deck = [j % len(wording) for j in range(12)]
                    rng.shuffle(template_deck)
                template = template_deck[slot]
                strata = {
                    'input_regime': 'novel' if in_new else 'familiar',
                    'wording_regime': 'novel' if phrase_new else 'familiar',
                }
                fields = _choose(
                    family,
                    pools[domain],
                    used[family],
                    known if ev and not in_new else None,
                    slot,
                    subtraction_use,
                )
                used[family].add(_key(family, fields))
                if not ev:
                    known.update(_components(family, fields))
                rows = _variants(family, fields, slot)
                features = {'operations': [r[1] for r in rows]}
                if family == 'arithmetic':
                    a, b = fields
                    features.update(
                        add_carry=(a % 10 + b % 10 >= 10) if '+' in features['operations'] else None,
                        subtract_sign=SUB[slot][0] if slot in SUB else None,
                        subtract_borrow=SUB[slot][1] if slot in SUB else None,
                    )
                    if slot in SUB:
                        subtraction_use[a - b] += 1
                pair_id = f'{VERSION}/{split}/{family}/{i:03}'
                pair = {
                    'pair_id': pair_id,
                    'family': family,
                    'strata': strata,
                    'features': features,
                    'items': [],
                }
                for side, (values, operation, answer, rubric) in enumerate(rows):
                    prompt = wording[template].format(**values)
                    if normalized(prompt) in seen:
                        raise ValueError('Duplicate or excluded prompt: ' + pair_id)
                    seen.add(normalized(prompt))
                    messages = [
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': answer},
                    ]
                    x, y = encode_example(messages, tokenizer, context)
                    prefix = len(tokenizer.encode(prompt_text(messages[:-1])))
                    budget = min(96 if family == 'code' else 32, min(context, 512) - prefix)
                    if budget < len(y) - prefix + 1:
                        raise ValueError('Generation budget too short: ' + pair_id)
                    item = {
                        'id': digest(canonical(messages)),
                        'pair_id': pair_id,
                        'side': side,
                        'split': split,
                        'family': family,
                        'bucket': family,
                        'input_key': family + ':' + canonical(fields),
                        'wording_id': f'{split if phrase_new else "train"}/{family}/{template}',
                        'strata': strata,
                        'operation': operation,
                        'prompt': prompt,
                        'reference': answer,
                        'max_tokens': budget,
                        'rubric': rubric if rubric is not None else {'kind': 'exact', 'expected': answer},
                    }
                    pair['items'].append(item)
                    if not ev:
                        result['train'].append(
                            dict(
                                **item, source='paired_' + family, messages=messages, input_ids=x, labels=y
                            )
                        )
                first, second = pair['items']
                for own, wrong in ((first, second), (second, first)):
                    if (
                        not grade_answer(own, own['reference'])['passed']
                        or grade_answer(own, wrong['reference'])['passed']
                    ):
                        raise ValueError('Ambiguous pair: ' + pair_id)
                if ev:
                    result[split].append(pair)
    return result
