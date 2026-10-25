"""Tiny local RAG over project markdown, pure NumPy TF-IDF.

No sklearn, no torch, no network. Indexes *.md files under a root directory,
chunks them into overlapping windows, builds a TF-IDF vector space, and
returns the top-k most relevant chunks for a query.

Good enough to make the model feel like it "knows the project" without any
fine-tuning. Not a vector database; don't use for multi-million-doc corpora.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{1,}")

_STOPWORDS = frozenset("""
a an and are as at be but by for from has have he her him his i if in is it
its of on or our she that the their them then there these they this to was
we were will with you your yours do does did not no so such than too very
can cannot could would should may might must shall also just much many more
most other some any each every all one two three same different new old own
""".split())


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]


@dataclass
class Chunk:
    doc_id: int
    rel_path: str
    start: int
    text: str


@dataclass
class RAGIndex:
    root: Path
    chunks: List[Chunk] = field(default_factory=list)
    vocab: dict = field(default_factory=dict)  # term -> col index
    idf: Optional[np.ndarray] = None            # (vocab_size,)
    matrix: Optional[np.ndarray] = None         # (n_chunks, vocab_size), L2-normalized

    @property
    def ready(self) -> bool:
        return self.matrix is not None and len(self.chunks) > 0


def _iter_markdown(root: Path, max_depth: int = 4, max_bytes: int = 200_000) -> Iterable[Path]:
    skip_dirs = {"node_modules", ".git", "dist", "build", "__pycache__", "venv", ".venv"}
    root = root.resolve()
    for p in root.rglob("*.md"):
        try:
            rel_parts = p.relative_to(root).parts
        except ValueError:
            continue
        if any(part in skip_dirs for part in rel_parts):
            continue
        if len(rel_parts) > max_depth:
            continue
        try:
            if p.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        yield p


def _chunk_text(text: str, size: int = 800, overlap: int = 150) -> List[Tuple[int, str]]:
    """Chunk on paragraph boundaries where possible, with a character cap."""
    out: List[Tuple[int, str]] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return out
    buf: List[str] = []
    buf_len = 0
    start = 0
    cursor = 0
    for para in paragraphs:
        plen = len(para) + 2
        if buf and buf_len + plen > size:
            chunk = "\n\n".join(buf)
            out.append((start, chunk))
            # Overlap: keep the tail of the previous chunk as the head of the next
            tail = chunk[-overlap:] if overlap < len(chunk) else chunk
            buf = [tail, para] if tail.strip() else [para]
            buf_len = sum(len(x) + 2 for x in buf)
            start = cursor
        else:
            buf.append(para)
            buf_len += plen
        cursor += plen
    if buf:
        out.append((start, "\n\n".join(buf)))
    return out


def build_index(root: str | Path) -> RAGIndex:
    """Scan root for markdown files and build a TF-IDF index in-memory."""
    root = Path(root)
    idx = RAGIndex(root=root)

    doc_tokens: List[List[str]] = []

    for doc_id, path in enumerate(_iter_markdown(root)):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        for start, piece in _chunk_text(text):
            toks = _tokenize(piece)
            if len(toks) < 10:
                continue
            idx.chunks.append(Chunk(doc_id=doc_id, rel_path=rel, start=start, text=piece))
            doc_tokens.append(toks)

    if not idx.chunks:
        return idx

    # Build vocabulary: terms that appear in >= 2 chunks, < 80% of chunks
    df: dict = {}
    for toks in doc_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    n_chunks = len(idx.chunks)
    min_df = 2
    max_df = int(0.8 * n_chunks)
    vocab_terms = [t for t, c in df.items() if min_df <= c <= max_df]
    vocab_terms.sort()
    idx.vocab = {t: i for i, t in enumerate(vocab_terms)}
    V = len(vocab_terms)
    if V == 0:
        return idx

    # IDF: log((N + 1) / (df + 1)) + 1
    idf = np.empty(V, dtype=np.float32)
    for t, i in idx.vocab.items():
        idf[i] = math.log((n_chunks + 1) / (df[t] + 1)) + 1.0
    idx.idf = idf

    # TF-IDF matrix, L2-normalized rows
    M = np.zeros((n_chunks, V), dtype=np.float32)
    for row, toks in enumerate(doc_tokens):
        counts: dict = {}
        for t in toks:
            j = idx.vocab.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        if not counts:
            continue
        total = sum(counts.values())
        for j, c in counts.items():
            M[row, j] = (c / total) * idf[j]
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    idx.matrix = (M / norms).astype(np.float32)
    return idx


def retrieve(idx: RAGIndex, query: str, k: int = 4,
             min_score: float = 0.05) -> List[Tuple[float, Chunk]]:
    """Return up to k top-scoring chunks for the query."""
    if not idx.ready or not idx.vocab:
        return []
    toks = _tokenize(query)
    if not toks:
        return []
    V = len(idx.vocab)
    q = np.zeros(V, dtype=np.float32)
    for t in toks:
        j = idx.vocab.get(t)
        if j is not None:
            q[j] += 1.0
    if not q.any():
        return []
    q *= idx.idf  # type: ignore[operator]
    nrm = np.linalg.norm(q)
    if nrm == 0:
        return []
    q /= nrm
    scores = idx.matrix @ q  # type: ignore[operator]
    top = np.argsort(-scores)[:k]
    out: List[Tuple[float, Chunk]] = []
    for i in top:
        s = float(scores[i])
        if s < min_score:
            break
        out.append((s, idx.chunks[int(i)]))
    return out


def format_retrieved(hits: List[Tuple[float, Chunk]], max_chars: int = 3000) -> str:
    """Format hits as a context block to inject into the system prompt."""
    if not hits:
        return ""
    parts: List[str] = ["RELEVANT CONTEXT (from project docs):"]
    used = 0
    for score, ch in hits:
        header = f"\n[{ch.rel_path}] (score={score:.2f})\n"
        remaining = max_chars - used - len(header)
        if remaining < 200:
            break
        body = ch.text[:remaining]
        parts.append(header + body)
        used += len(header) + len(body)
    return "\n".join(parts)


# ── Convenience singleton for the server ───────────────────────────
_GLOBAL: Optional[RAGIndex] = None


def init_global(root: str | Path) -> RAGIndex:
    global _GLOBAL
    t0 = time.perf_counter()
    _GLOBAL = build_index(root)
    elapsed = time.perf_counter() - t0
    n = len(_GLOBAL.chunks)
    v = len(_GLOBAL.vocab)
    print(f"[rag] Indexed {n} chunks, vocab={v}, in {elapsed:.1f}s")
    return _GLOBAL


def get_global() -> Optional[RAGIndex]:
    return _GLOBAL
