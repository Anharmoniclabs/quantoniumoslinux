"""Deterministic paired examples for a bounded prompt-conditioning experiment.

Each pair requires two different answers after a small change to the request
or its supplied facts. Dev and final probe transfer within these four families.
"""

from chat_recovery_data import canonical, digest, encode_example, prompt_text
from coherence_quality import grade_answer, normalized

VERSION = 'phistate-paired-instruction-v1'
FAMILIES = ('arithmetic', 'string', 'grounded', 'code')
COUNTS = {'train': 128, 'dev': 3, 'final': 3}  # Pairs per family.

# Keep each wording exclusive to one split. The alternatives in each pair use
# the same wording apart from the operation, query, or fact that changes.
WORDING = {
    'arithmetic': {
        'train': ['Calculate {expr}. Give only the integer.', 'What is {words}? Reply with the number only.',
                  'Please evaluate {expr} and write just its integer value.', 'I need {words}. Answer with one integer.'],
        'dev': ['A worksheet asks for {expr}. Fill the blank with a number.', 'Find {words} and respond in digits only.',
                'For the expression {expr}, supply its signed integer result.'],
        'final': ['Enter the numerical result of {expr} by itself.', 'What integer do you get for {words}? No explanation.',
                  'My calculation is {expr}. Write only its result.']},
    'string': {
        'train': ['Input text: `{value}`. {action}. Output only the result.', 'For `{value}`, {action_lower}. Reply only with the transformed text.',
                  'Take the text `{value}`. {action}. Give the text alone.', 'Transform `{value}` as follows: {action_lower}. Return only the result.'],
        'dev': ['My label reads `{value}`. {action}; send back just the label.', 'Apply this edit to `{value}`: {action_lower}. Give only the edited text.',
                'Starting from `{value}`, {action_lower}. What is the final text? Return it alone.'],
        'final': ['I typed `{value}`. Please {action_lower}, then reply with the result alone.', 'The source string is `{value}`. {action}. Write only your output.',
                  'Use `{value}` as the input. Your operation is to {action_lower}. Respond only with the resulting string.']},
    'grounded': {
        'train': ['I left my {a} in the {p} and my {b} in the {q}. Where is my {ask}? Answer with just the place.',
                  'My {a} is in the {p}; my {b} is in the {q}. Which place has my {ask}? Give just its name.',
                  'I put the {a} in the {p}. I put the {b} in the {q}. Tell me where the {ask} is using only the place name.',
                  'Here is my packing note: {a} in the {p}, {b} in the {q}. Where should I look for my {ask}? Name only the place.'],
        'dev': ['I have forgotten where I put my things. My note says the {a} is in the {p} and the {b} in the {q}. Where is the {ask}? Reply with the place name alone.',
                'According to my note, the {p} holds my {a} and the {q} holds my {b}. Where did I store the {ask}? Say just the place.',
                'I remember leaving the {a} in the {p} and the {b} in the {q}. Where would I find the {ask}? Give one place name.'],
        'final': ['Can you read back a location from my list? {a}: {p}. {b}: {q}. I want my {ask}. Return just the location.',
                  'I moved the {a} to the {p} and the {b} to the {q}. Which spot should I visit for the {ask}? Answer with the spot alone.',
                  'My inventory lists the {b} in the {q} and the {a} in the {p}. Name only the place containing my {ask}.']},
    'code': {
        'train': ['Here is a Python function:\n{start}\nEdit it to {action}. Return the revised function as Python code.',
                  'Change this Python code so it will {action}:\n{start}\nReply with the edited function.',
                  'Please revise the body of this function to {action}. Keep the signature.\n{start}\nGive Python code only.',
                  'Starting code:\n{start}\nModify its return value to {action}. Output the complete Python function.'],
        'dev': ['I have `{signature}` with a body of `return {var}`. Rewrite it in Python to {action}. Keep the function and parameter names.',
                'Patch this Python definition:\n{start}\nIts intended behavior is to {action}. Show just the replacement definition.',
                'The identity function below needs an edit. Make it {action} and send the resulting Python code.\n{start}'],
        'final': ['Keep this Python signature, but replace the body so it will {action}:\n{start}\nProvide the updated definition.',
                  'Edit `{signature}` in Python: it currently returns {var}. It should {action}. Return only the function.',
                  'Please correct my Python helper. Instead of returning its input unchanged, it should {action}.\n{start}\nGive the complete corrected function.']},
}


def _exact(expected):
    return str(expected), {'kind': 'exact', 'expected': str(expected)}


def _arithmetic(split, i):
    a, b = ((2 + i//8, 2 + i % 8) if split == 'train' else
            (20+i, 11+i) if split == 'dev' else (25+i, 14+i))
    operations = [('+', f'the sum of {a} and {b}', a+b),
                  ('-', f'{a} minus {b}', a-b), ('*', f'the product of {a} and {b}', a*b)]
    first = i % 3
    second = (first + 1) % 3
    if operations[first][2] == operations[second][2]:
        second = (first + 2) % 3
    result = []
    for idx in (first, second):
        op, words, answer = operations[idx]
        result.append((dict(expr=f'{a} {op} {b}', words=words), *_exact(answer)))
    return [a,b], result


def _string(split, i):
    adjectives = ('red blue green gold silver white black pink warm cool soft loud bright pale tall short'.split())
    nouns = 'pear lamp bird cup book coat door star'.split()
    if split == 'train':
        a, b = adjectives[i//8], nouns[i%8]
    elif split == 'dev':
        a, b = [('fresh','lime'), ('round','stone'), ('quiet','drum')][i]
    else:
        a, b = [('sweet','plum'), ('clear','lake'), ('clean','glass')][i]
    value = a.capitalize()+'-'+b.capitalize()
    ops = [('Convert all letters to uppercase', value.upper()),
           ('Convert all letters to lowercase', value.lower()),
           ('Reverse the order of all characters, preserving their case', value[::-1])]
    result = []
    for idx in (i % 3, (i+1) % 3):
        action, expected = ops[idx]
        result.append((dict(value=value, action=action, action_lower=action[0].lower()+action[1:]), *_exact(expected)))
    return [value], result


def _grounded(split, i):
    objs = 'bag hat coin scarf rope note mug pen shoe key sock map pin card cap comb'.split()
    places = 'porch garage barn shed kitchen office bedroom attic'.split()
    if split == 'train':
        a, b = objs[i//8], objs[(i//8+3) % len(objs)]
        p, q = places[i % 8], places[(i+3) % len(places)]
    elif split == 'dev':
        a,b,p,q = [('ribbon','spool','hall','cellar'),('badge','ticket','studio','lobby'),('bowl','fork','pantry','yard')][i]
    else:
        a,b,p,q = [('sketchbook','chalk','loft','den'),('button','brush','balcony','closet'),('glove','bottle','cabin','tent')][i]
    base = dict(a=a, b=b, p=p, q=q, ask=a)
    # Alternate changing the question and changing the supplied fact. In the
    # latter pair the requested object stays fixed while its location changes.
    if i % 2:
        other = dict(base, p=q, q=p)
    else:
        other = dict(base, ask=b)
    return [a,b,p,q], [(base, *_exact(p)), (other, *_exact(q))]


def _code(split, i):
    if split == 'train':
        name = 'adjust update shift change offset scale move revise'.split()[i % 8]
        k, var = 2 + i//8, 'x' if i % 2 == 0 else 'n'
    elif split == 'dev':
        name, k, var = [('alter',18,'v'),('modify',19,'x'),('change_value',20,'n')][i]
    else:
        name, k, var = [('transform',21,'n'),('revise_value',22,'v'),('adjust_value',23,'x')][i]
    signature = f'{name}({var})'
    start = f'def {signature}:\n    return {var}'
    operations = [(f'return the input plus {k}', '+', lambda v: v+k),
                  (f'return the input minus {k}', '-', lambda v: v-k),
                  (f'return the input multiplied by {k}', '*', lambda v: v*k)]
    result = []
    for idx in (i % 3, (i+1) % 3):
        action, symbol, fn = operations[idx]
        gold = f'```python\ndef {signature}:\n    return {var} {symbol} {k}\n```'
        rubric = dict(kind='code', function=name, args=[var],
                      cases=[dict(args=[v], expected=fn(v)) for v in (-7,0,3,11)])
        result.append((dict(start=start, signature=signature, var=var, action=action), gold, rubric))
    return [name,k,var], result


def build(tokenizer, context, excluded_prompts=()):
    """Return 1,024 training rows, 12 dev pairs and 12 independent final pairs.

    Hosts can supply existing assessment and corpus prompts for strict exclusion.
    An exact normalized collision fails explicitly. No replacement rows are sampled.
    """
    if not isinstance(context, int) or isinstance(context, bool) or context < 1:
        raise ValueError('context must be a positive integer')
    if isinstance(excluded_prompts, str):
        raise TypeError('excluded_prompts must be an iterable of prompt strings')
    excluded = {normalized(p) for p in excluded_prompts}
    seen_prompts, seen_inputs = set(), set()
    result = dict(train=[], dev=[], final=[])
    factories = dict(arithmetic=_arithmetic, string=_string, grounded=_grounded, code=_code)
    for split in COUNTS:
        for family, factory in factories.items():
            patterns = WORDING[family][split]
            for i in range(COUNTS[split]):
                fields, variants = factory(split, i)
                input_key = family+':'+canonical(fields)
                if input_key in seen_inputs:
                    raise ValueError('Repeated underlying input: '+input_key)
                seen_inputs.add(input_key)
                template = i % len(patterns)
                pair_id = f'{VERSION}/{split}/{family}/{i:03d}'
                pair = []
                for side, (values, reference, rubric) in enumerate(variants):
                    prompt = patterns[template].format(**values)
                    key = normalized(prompt)
                    if key in excluded or key in seen_prompts:
                        raise ValueError('Duplicate or excluded prompt: '+pair_id)
                    seen_prompts.add(key)
                    messages = [dict(role='user', content=prompt), dict(role='assistant', content=reference)]
                    x, y = encode_example(messages, tokenizer, context)
                    prefix_length = len(tokenizer.encode(prompt_text(messages[:-1])))
                    available = min(context, 512)-prefix_length
                    max_tokens = min(96 if family == 'code' else 32, available)
                    if max_tokens < len(y)-prefix_length+1:
                        raise ValueError('Generation budget cannot cover the gold answer plus EOS: '+pair_id)
                    item = dict(id=digest(canonical(messages)), pair_id=pair_id, side=side, split=split,
                                family=family, bucket=family, input_key=input_key,
                                wording_id=f'{split}/{family}/{template}', prompt=prompt, reference=reference,
                                max_tokens=max_tokens, rubric=rubric)
                    pair.append(item)
                    if split == 'train':
                        result['train'].append(dict(**item, source='paired_'+family, messages=messages, input_ids=x, labels=y))
                for side, own in enumerate(pair):
                    if not grade_answer(own, own['reference'])['passed'] or grade_answer(own, pair[1-side]['reference'])['passed']:
                        raise ValueError('Pair has ambiguous or invalid reference answers: '+pair_id)
                if split != 'train':
                    result[split].append(dict(pair_id=pair_id, family=family, items=pair))
    return result
