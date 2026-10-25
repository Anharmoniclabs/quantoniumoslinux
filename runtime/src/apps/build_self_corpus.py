# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Build Training Corpus from Your Own Materials
==============================================

Extracts training data from:
1. Markdown documentation (docs/*.md)
2. Research papers (LaTeX, converted)
3. Python docstrings and comments
4. README files

Output: JSONL file with prompt/target pairs for spectral fine-tuning.

This is 100% legal - it's YOUR content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


def extract_markdown_chunks(path: Path, chunk_size: int = 500) -> List[Dict]:
    """Extract Q&A style chunks from markdown files."""
    chunks = []
    try:
        content = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return chunks

    # Split by headers
    sections = re.split(r'\n(#{1,3}\s+[^\n]+)\n', content)

    current_header = path.stem
    current_content = []

    for i, section in enumerate(sections):
        if re.match(r'^#{1,3}\s+', section):
            # This is a header
            if current_content:
                text = '\n'.join(current_content).strip()
                if len(text) > 50:
                    chunks.append({
                        'source': str(path.relative_to(REPO_ROOT)),
                        'header': current_header,
                        'content': text[:chunk_size * 2],
                    })
            current_header = section.strip('#').strip()
            current_content = []
        else:
            current_content.append(section)

    # Last section
    if current_content:
        text = '\n'.join(current_content).strip()
        if len(text) > 50:
            chunks.append({
                'source': str(path.relative_to(REPO_ROOT)),
                'header': current_header,
                'content': text[:chunk_size * 2],
            })

    return chunks


def extract_python_docstrings(path: Path) -> List[Dict]:
    """Extract docstrings and function signatures from Python files."""
    chunks = []
    try:
        content = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return chunks

    # Module docstring
    module_doc = re.search(r'^"""(.*?)"""', content, re.DOTALL)
    if module_doc:
        doc = module_doc.group(1).strip()
        if len(doc) > 50:
            chunks.append({
                'source': str(path.relative_to(REPO_ROOT)),
                'type': 'module_doc',
                'content': doc[:1000],
            })

    # Function/class docstrings
    pattern = r'(def|class)\s+(\w+)[^:]*:\s*"""(.*?)"""'
    for match in re.finditer(pattern, content, re.DOTALL):
        kind, name, doc = match.groups()
        doc = doc.strip()
        if len(doc) > 30:
            chunks.append({
                'source': str(path.relative_to(REPO_ROOT)),
                'type': f'{kind}_doc',
                'name': name,
                'content': doc[:800],
            })

    return chunks


def chunk_to_qa(chunk: Dict) -> Dict[str, str]:
    """Convert a content chunk to Q&A format."""
    content = chunk.get('content', '')
    source = chunk.get('source', 'unknown')

    # Generate question based on content type
    if 'header' in chunk:
        header = chunk['header']
        prompt = f"Explain {header} in the context of quantoniumos/RFTMW."
    elif chunk.get('type') == 'module_doc':
        prompt = f"What does the module {source} do?"
    elif chunk.get('type') == 'def_doc':
        name = chunk.get('name', 'this function')
        prompt = f"What does the function {name} do?"
    elif chunk.get('type') == 'class_doc':
        name = chunk.get('name', 'this class')
        prompt = f"Describe the {name} class."
    else:
        prompt = f"Explain this concept from {source}."

    # Target is the content (first ~500 chars for training)
    target = content[:500].strip()

    return {
        'prompt': prompt,
        'target': target,
        'source': source,
    }


def build_corpus(
    output_path: str,
    max_samples: int = 5000,
    include_code: bool = True,
) -> int:
    """Build training corpus from repo materials."""
    all_chunks = []

    # 1. Markdown docs
    print("Extracting markdown documentation...")
    for md_path in REPO_ROOT.rglob('*.md'):
        if '.venv' in str(md_path) or 'node_modules' in str(md_path):
            continue
        chunks = extract_markdown_chunks(md_path)
        all_chunks.extend(chunks)
    print(f"  Found {len(all_chunks)} markdown chunks")

    # 2. Python docstrings
    if include_code:
        print("Extracting Python docstrings...")
        code_chunks = []
        for py_path in REPO_ROOT.rglob('*.py'):
            if '.venv' in str(py_path) or 'node_modules' in str(py_path):
                continue
            chunks = extract_python_docstrings(py_path)
            code_chunks.extend(chunks)
        print(f"  Found {len(code_chunks)} code chunks")
        all_chunks.extend(code_chunks)

    # 3. Convert to Q&A format
    print("Converting to Q&A format...")
    qa_pairs = [chunk_to_qa(c) for c in all_chunks]

    # 4. Deduplicate by content hash
    seen = set()
    unique_pairs = []
    for pair in qa_pairs:
        h = hashlib.sha256(pair['target'].encode()).hexdigest()[:16]
        if h not in seen:
            seen.add(h)
            unique_pairs.append(pair)

    print(f"  Unique pairs: {len(unique_pairs)}")

    # 5. Limit and shuffle
    import random
    random.shuffle(unique_pairs)
    final_pairs = unique_pairs[:max_samples]

    # 6. Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in final_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(final_pairs)} training pairs to {output_path}")
    return len(final_pairs)


def main():
    parser = argparse.ArgumentParser(description="Build self-training corpus")
    parser.add_argument('--out', default='ai/training/datasets/self_corpus.jsonl')
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--no_code', action='store_true',
                       help='Exclude Python docstrings')
    args = parser.parse_args()

    print("=" * 60)
    print("  BUILD SELF-TRAINING CORPUS")
    print("  100% your own content - legal & unbiased")
    print("=" * 60)
    print()

    n = build_corpus(
        output_path=args.out,
        max_samples=args.max_samples,
        include_code=not args.no_code,
    )

    print()
    print("=" * 60)
    print(f"  CORPUS READY: {n} samples")
    print(f"  Next: python src/apps/train_rftmw_spectral.py --data {args.out}")
    print("=" * 60)


if __name__ == '__main__':
    main()
