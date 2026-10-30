"""Pinned public sources, complete-answer SFT examples and a shared chat template."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import urllib.request

import sentencepiece as spm

TEMPLATE = 'phistate-chat-newline-v1'
SOURCES = {
    'conversation': dict(repo='OpenAssistant/oasst1', revision='fdf72ae0827c1cda404aff25b6603abec9e3399b',
        file='data/train-00000-of-00001-b42a775f407cee45.parquet', license='apache-2.0',
        sha256='bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e', weight=.60),
    'python': dict(repo='xTayyub/High-Quality-Synthetic-Python-Dataset-with-Reasoning-Traces-Chain-of-Thought-for-LLM-Fine-Tuning',
        revision='29de362c24bc79b1cda01a7cfd33583a70abe9fc', file='dataset.json', license='apache-2.0',
        sha256='a572a9e949b88c6d6d3ca0e656d8e821d74c3198b55300f3500f0bf171b56af3', weight=.20),
    'support': dict(repo='bitext/Bitext-customer-support-llm-chatbot-training-dataset',
        revision='430d1a89bd93bd1fa23c16f29dd53e73f0087443',
        file='Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv', license='cdla-sharing-1.0',
        sha256='6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434', weight=.10),
    'math': dict(repo='openai/gsm8k', revision='740312add88f781978c0658806c59bc2815b9866',
        file='main/train-00000-of-00001.parquet', license='mit',
        sha256='ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d', weight=.10),
}
EXCLUDED = {
    'Rapidata/text-to-3d-with-llm-code': 'Render preference benchmark; inspected schema has no answer/code targets.',
    'shark803thusz/case_study_LLM_casual_reasoning': 'Small test workbook with ground truth AND incorrect competing model answers; not training data.',
    'APTO-001/llm-math-reasoning-dataset': 'Gated; not downloaded. GSM8K train supplies accessible math examples.',
}


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def read_jsonl(path):
    # str.splitlines() also splits literal Unicode separators INSIDE JSON strings.
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def clean_text(text):
    if not isinstance(text, str):
        raise ValueError('not text')
    text = text.replace('\r\n', '\n').replace('\r', '\n').strip()
    if not text or '\x00' in text or '<|code|>' in text or '<|general|>' in text:
        raise ValueError('empty/control/domain marker in content')
    if len(text) > 20000:
        raise ValueError('oversize')
    # Keep embedded code whitespace. Refuse strings that inject template turns.
    if re.search(r'(?m)^(?:User|Assistant):', text):
        raise ValueError('embedded role delimiter')
    return text


def prompt_text(messages):
    if not messages or messages[-1]['role'] != 'user':
        raise ValueError('Prompt must end in a user turn')
    result = '<|general|>\n'
    for i, row in enumerate(messages):
        expected = 'user' if i % 2 == 0 else 'assistant'
        if row['role'] != expected:
            raise ValueError('Turns must alternate starting with user')
        label = 'User' if expected == 'user' else 'Assistant'
        result += label + ':\n' + row['content'] + '\n'
    return result + 'Assistant:\n'


def encode_example(messages, tokenizer, context):
    if len(messages) < 2 or messages[-1]['role'] != 'assistant':
        raise ValueError('Missing complete assistant answer')
    prefix = prompt_text(messages[:-1])
    answer = messages[-1]['content']
    prefix_ids = tokenizer.encode(prefix)
    ids = tokenizer.encode(prefix + answer)
    # No separately encoded answer/dummy prefix and no fabricated EOS on truncation.
    if ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError('Tokenizer merged across the answer boundary')
    if tokenizer.decode(ids) != prefix + answer:
        raise ValueError('Tokenizer round trip failed')
    if len(ids) <= len(prefix_ids) or len(ids) > context:
        raise ValueError('Example does not fit complete in context')
    if tokenizer.eos_id() < 0:
        raise ValueError('EOS required')
    ids.append(tokenizer.eos_id())
    labels = ids[1:]
    labels[:len(prefix_ids) - 1] = [-100] * (len(prefix_ids) - 1)
    return ids[:-1], labels


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    elif isinstance(value, dict):
        yield value


def python_rows(path, counts):
    allowed = {'language', 'task_type', 'instruction', 'thought_process', 'code_solution', 'explanation'}
    for row in flatten(json.loads(path.read_text())):
        counts['raw'] += 1
        if set(row) - allowed or row.get('language', '').lower() != 'python':
            counts['malformed_schema'] += 1
            continue
        code = row.get('code_solution')
        if not isinstance(code, str):
            counts['missing_code'] += 1
            continue
        code = code.strip()
        if code.startswith('python\n'):
            code = code[len('python\n'):]
        if code.startswith('```python\n') and code.endswith('```'):
            code = code[len('```python\n'):-3].rstrip()
        try:
            tree = ast.parse(code)
            # A stub such as `python` or an empty/comment-only body is not a solution.
            if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in tree.body):
                raise ValueError('No defined implementation')
        except (SyntaxError, ValueError, TypeError):
            counts['invalid_python'] += 1
            continue
        # Syntax is a filter, not proof of functional correctness. Never execute dataset code.
        yield {'source': 'python', 'group': 'python:' + digest(ast.dump(tree, include_attributes=False)),
               'messages': [{'role': 'user', 'content': row.get('instruction')},
                            {'role': 'assistant', 'content': '```python\n' + code + '\n```'}]}


def support_rows(path, counts):
    with path.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            counts['raw'] += 1
            # Separate domain context prevents teaching the general assistant it has an order database.
            request = ('Customer-support writing task: draft a reply to the following customer message. '
                       'Retain any template placeholders.\n' + row['instruction'])
            yield {'source': 'support', 'group': 'support-intent:' + row['intent'],
                   'messages': [{'role': 'user', 'content': request},
                                {'role': 'assistant', 'content': row['response']}]}


def conversation_rows(path, counts):
    import pyarrow.parquet as pq
    records = {r['message_id']: r for r in pq.read_table(path).to_pylist()}
    for row in records.values():
        if row['role'] != 'assistant':
            continue
        counts['raw'] += 1
        if row['lang'] != 'en' or row['deleted'] or row['review_result'] is not True or row['rank'] != 0:
            counts['not_accepted_english_rank0'] += 1
            continue
        chain, seen, current = [], set(), row
        valid = True
        while current is not None:
            if current['message_id'] in seen or len(chain) >= 12:
                valid = False
                break
            seen.add(current['message_id'])
            if current['lang'] != 'en' or current['deleted'] or current['review_result'] is False:
                valid = False
                break
            chain.append({'role': 'user' if current['role'] == 'prompter' else 'assistant',
                          'content': current['text']})
            parent = current['parent_id']
            current = records.get(parent)
            if parent and current is None:
                valid = False
                break
        if not valid:
            counts['incomplete_or_unreviewed_chain'] += 1
            continue
        yield {'source': 'conversation', 'group': 'oasst-tree:' + row['message_tree_id'],
               'messages': list(reversed(chain))}


def math_rows(path, counts):
    import pyarrow.parquet as pq
    for row in pq.read_table(path).to_pylist():
        counts['raw'] += 1
        answer = row['answer']
        if '####' not in answer:
            counts['no_final_answer'] += 1
            continue
        explanation, final = answer.rsplit('####', 1)
        explanation = re.sub(r'<<[^<>]*>>', '', explanation).strip()
        answer = explanation + '\nAnswer: ' + final.strip()
        yield {'source': 'math', 'group': 'math:' + digest(row['question']),
               'messages': [{'role': 'user', 'content': row['question']},
                            {'role': 'assistant', 'content': answer}]}


def download_source(name, cache):
    spec = SOURCES[name]
    path = cache / name / Path(spec['file']).name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pending = path.with_suffix(path.suffix + '.partial')
        url = f'https://huggingface.co/datasets/{spec["repo"]}/resolve/{spec["revision"]}/{spec["file"]}'
        print('Downloading pinned source:', spec['repo'], flush=True)
        with urllib.request.urlopen(url, timeout=120) as src, pending.open('wb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        pending.replace(path)
    actual = sha(path)
    if spec.get('sha256') and actual != spec['sha256']:
        raise RuntimeError('Source bytes differ: ' + str(path))
    # Preserve upstream attribution/license documentation alongside prepared data.
    readme = path.parent / 'README.upstream.md'
    if not readme.exists():
        url = f'https://huggingface.co/datasets/{spec["repo"]}/resolve/{spec["revision"]}/README.md'
        with urllib.request.urlopen(url, timeout=120) as response:
            readme.write_bytes(response.read())
    return path, actual, readme


def group_split(rows):
    # Connected components keep duplicate prompts, equivalent Python ASTs and
    # full conversation trees on one side. Bitext intents stay disjoint too.
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        question = ' '.join(row['messages'][-2]['content'].lower().split())
        for key in (row['group'], 'prompt:' + digest(question)):
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    components = defaultdict(list)
    for i in range(len(rows)):
        components[find(i)].append(i)
    component_ids = {root: digest('|'.join(sorted({rows[i]['group'] for i in members})))
                     for root, members in components.items()}
    validation = set()
    # Reserve at least one whole component per source, deterministically.
    for source in SOURCES:
        roots = [root for root, members in components.items() if any(rows[i]['source'] == source for i in members)]
        roots.sort(key=lambda root: component_ids[root])
        validation.update(roots[:max(1, round(len(roots) * .08))])
    for root, members in components.items():
        for i in members:
            rows[i]['split_group'] = component_ids[root]
            rows[i]['split'] = 'val' if root in validation else 'train'
    return rows


def verify_prepared(output, tokenizer_hash, context):
    manifest = json.loads((output / 'manifest.json').read_text())
    fingerprint = manifest.pop('data_fingerprint')
    if digest(canonical(manifest)) != fingerprint:
        raise RuntimeError('Prepared manifest changed')
    manifest['data_fingerprint'] = fingerprint
    if manifest['tokenizer_sha256'] != tokenizer_hash or manifest['context'] != context or manifest['template'] != TEMPLATE:
        raise RuntimeError('Prepared data settings mismatch')
    for name, expected in manifest['files'].items():
        if sha(output / name) != expected:
            raise RuntimeError('Prepared data changed: ' + name)
    return manifest


def build(data_dir, output, cache, context=512):
    data_dir, output, cache = map(Path, (data_dir, output, cache))
    token_path = data_dir / 'tokenizer.model'
    token_hash = sha(token_path)
    corpus = json.loads((data_dir / 'manifest.json').read_text())
    if corpus.get('whitespace_repair', {}).get('tokenizer_sha256') != token_hash:
        raise RuntimeError('Use the verified whitespace-repaired corpus')
    if output.exists():
        return verify_prepared(output, token_hash, context)
    tokenizer = spm.SentencePieceProcessor(model_file=str(token_path))
    rows, stats, provenance, readmes = [], {}, {}, {}
    adapters = dict(conversation=conversation_rows, python=python_rows, support=support_rows, math=math_rows)
    seen = set()
    for source, adapter in adapters.items():
        path, file_hash, readme = download_source(source, cache)
        counts = Counter()
        for row in adapter(path, counts):
            try:
                row['messages'] = [{'role': m['role'], 'content': clean_text(m['content'])} for m in row['messages']]
                x, y = encode_example(row['messages'], tokenizer, context)
            except (ValueError, KeyError, TypeError):
                counts['rejected_content_or_length'] += 1
                continue
            key = digest(canonical(row['messages']))
            if key in seen:
                counts['exact_duplicate'] += 1
                continue
            seen.add(key)
            row.update(id=key, input_ids=x, labels=y)
            rows.append(row)
            counts['accepted_before_split'] += 1
        stats[source] = dict(counts)
        provenance[source] = {**SOURCES[source], 'sha256': file_hash}
        readmes[source] = readme
        print('SOURCE', source, json.dumps(stats[source]), flush=True)
    rows = group_split(rows)
    counts = {source: {split: sum(r['source'] == source and r['split'] == split for r in rows)
                       for split in ('train', 'val')} for source in SOURCES}
    for source, n in counts.items():
        if n['train'] < 10 or n['val'] < 1:
            raise RuntimeError(f'Too few complete {source} examples: {n}; no training started')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + '.pending_', dir=output.parent))
    for split in ('train', 'val'):
        with (stage / (split + '.jsonl')).open('w', encoding='utf-8') as f:
            for row in sorted((r for r in rows if r['split'] == split), key=lambda r: r['id']):
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    shutil.copyfile(token_path, stage / 'tokenizer.model')
    for source, path in readmes.items():
        shutil.copyfile(path, stage / (source + '.upstream.md'))
    manifest = dict(format='PhiStateChatRecovery-v1', template=TEMPLATE, context=context,
                    tokenizer_sha256=token_hash, base_data_fingerprint=corpus['data_fingerprint'],
                    sources=provenance, excluded=EXCLUDED, filtering=stats, counts=counts,
                    split='Connected groups: conversation trees, Python ASTs, support intents, duplicate prompts; about 8% groups reserved',
                    answer_targets='Complete final answers plus EOS only; no truncation, no thought_process targets',
                    files={p.name: sha(p) for p in stage.iterdir()})
    manifest['data_fingerprint'] = digest(canonical(manifest))
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify_prepared(stage, token_hash, context)
    stage.rename(output)
    print('PREPARED', output, json.dumps(counts), flush=True)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--context', type=int, default=512)
    a = p.parse_args()
    build(a.data_dir, a.output, a.cache, a.context)
