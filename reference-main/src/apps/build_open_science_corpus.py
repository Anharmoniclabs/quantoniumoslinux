# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Build a trainable open-science corpus from arXiv metadata/abstracts.

This script collects open scientific abstracts from arXiv and writes JSONL rows:
  {"text": "...", "source": "arxiv", ...}

The output is compatible with `src/apps/train_local_chat_lora.py --corpus`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
OPEN_NS = {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}


@dataclass
class ArxivRecord:
    entry_id: str
    title: str
    summary: str
    categories: List[str]
    authors: List[str]
    published: str
    updated: str
    link: str
    license_url: str


def normalize_ws(text: str) -> str:
    return " ".join((text or "").strip().split())


def parse_iso8601(value: str) -> Optional[dt.datetime]:
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(raw)
    except ValueError:
        return None


def build_query(categories: List[str], terms: str) -> str:
    cat_expr = " OR ".join(f"cat:{c}" for c in categories if c)
    term_expr = normalize_ws(terms)
    if cat_expr and term_expr:
        return f"({cat_expr}) AND ({term_expr})"
    if cat_expr:
        return cat_expr
    return term_expr or "all:science"


def fetch_arxiv_page(
    *,
    search_query: str,
    start: int,
    max_results: int,
    sort_by: str = "lastUpdatedDate",
    sort_order: str = "descending",
    timeout_sec: int = 45,
) -> str:
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    url = f"{ARXIV_API_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "quantoniumos-open-science-corpus/1.0"})
    with urlopen(req, timeout=timeout_sec) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_total_results(xml_text: str) -> Optional[int]:
    root = ET.fromstring(xml_text)
    total = root.find("opensearch:totalResults", OPEN_NS)
    if total is None or not (total.text or "").strip():
        return None
    try:
        return int(total.text.strip())
    except ValueError:
        return None


def parse_entries(xml_text: str) -> List[ArxivRecord]:
    root = ET.fromstring(xml_text)
    out: List[ArxivRecord] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        entry_id = normalize_ws(entry.findtext("atom:id", default="", namespaces=ATOM_NS))
        title = normalize_ws(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
        summary = normalize_ws(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
        published = normalize_ws(entry.findtext("atom:published", default="", namespaces=ATOM_NS))
        updated = normalize_ws(entry.findtext("atom:updated", default="", namespaces=ATOM_NS))
        license_url = normalize_ws(entry.findtext("arxiv:license", default="", namespaces=ATOM_NS))

        categories = []
        for c in entry.findall("atom:category", ATOM_NS):
            term = normalize_ws(c.attrib.get("term", ""))
            if term:
                categories.append(term)

        authors = []
        for a in entry.findall("atom:author", ATOM_NS):
            name = normalize_ws(a.findtext("atom:name", default="", namespaces=ATOM_NS))
            if name:
                authors.append(name)

        link = ""
        for ln in entry.findall("atom:link", ATOM_NS):
            rel = normalize_ws(ln.attrib.get("rel", ""))
            href = normalize_ws(ln.attrib.get("href", ""))
            if rel == "alternate" and href:
                link = href
                break
        if not link:
            link = entry_id

        if not entry_id or not title or not summary:
            continue

        out.append(
            ArxivRecord(
                entry_id=entry_id,
                title=title,
                summary=summary,
                categories=categories,
                authors=authors,
                published=published,
                updated=updated,
                link=link,
                license_url=license_url,
            )
        )
    return out


def to_training_row(rec: ArxivRecord, max_abstract_chars: int) -> Dict[str, str]:
    abstract = rec.summary[:max_abstract_chars].strip()
    cats = ", ".join(rec.categories) if rec.categories else "unknown"
    authors = ", ".join(rec.authors[:12]) if rec.authors else "unknown"

    text = "\n".join(
        [
            "Source: arXiv",
            f"Title: {rec.title}",
            f"Authors: {authors}",
            f"Categories: {cats}",
            f"Published: {rec.published}",
            f"Updated: {rec.updated}",
            f"URL: {rec.link}",
            f"License: {rec.license_url or 'unspecified'}",
            f"Abstract: {abstract}",
        ]
    )

    return {
        "text": text,
        "source": "arxiv",
        "entry_id": rec.entry_id,
        "url": rec.link,
        "published": rec.published,
        "updated": rec.updated,
        "categories": rec.categories,
        "license": rec.license_url,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ai/training/datasets/open_science_arxiv.jsonl")
    ap.add_argument(
        "--meta_out",
        default="ai/training/datasets/open_science_arxiv_meta.jsonl",
        help="Optional metadata-only JSONL output.",
    )
    ap.add_argument(
        "--categories",
        default="cs.AI,cs.LG,cs.CL,stat.ML,cs.CR,quant-ph,eess.SP,math.NA",
        help="Comma-separated arXiv categories.",
    )
    ap.add_argument(
        "--terms",
        default="all:quantum OR all:machine learning OR all:signal processing OR all:cryptography",
        help="Extra arXiv search expression.",
    )
    ap.add_argument("--max_results", type=int, default=1200)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--since_days", type=int, default=3650)
    ap.add_argument("--max_abstract_chars", type=int, default=4000)
    ap.add_argument("--require_license", action="store_true")
    ap.add_argument("--timeout_sec", type=int, default=45)
    # arXiv requests responsible batching; default to a conservative delay between requests.
    ap.add_argument("--sleep_sec", type=float, default=3.0)
    args = ap.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    query = build_query(categories, args.terms)
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max(args.since_days, 0))

    print(f"arXiv query: {query}")
    print(f"Collecting up to {args.max_results} records...")

    seen: set[str] = set()
    selected: List[ArxivRecord] = []
    total_from_api: Optional[int] = None
    start = 0

    while len(selected) < args.max_results:
        page_size = min(args.batch_size, args.max_results - len(selected))
        try:
            xml_text = fetch_arxiv_page(
                search_query=query,
                start=start,
                max_results=page_size,
                timeout_sec=args.timeout_sec,
            )
        except Exception as e:
            raise SystemExit(
                "arXiv fetch failed. Check network/firewall access. "
                f"Details: {type(e).__name__}: {e}"
            ) from e
        if total_from_api is None:
            total_from_api = parse_total_results(xml_text)

        page_entries = parse_entries(xml_text)
        if not page_entries:
            break

        added_this_page = 0
        for rec in page_entries:
            if rec.entry_id in seen:
                continue
            seen.add(rec.entry_id)

            if args.require_license and not rec.license_url:
                continue

            updated_dt = parse_iso8601(rec.updated) or parse_iso8601(rec.published)
            if updated_dt is not None and updated_dt.tzinfo is None:
                updated_dt = updated_dt.replace(tzinfo=dt.timezone.utc)
            if updated_dt is not None and updated_dt < cutoff:
                continue

            selected.append(rec)
            added_this_page += 1
            if len(selected) >= args.max_results:
                break

        print(f"Fetched page start={start}: parsed={len(page_entries)} kept={added_this_page} total_kept={len(selected)}")
        if len(page_entries) < page_size:
            break
        start += len(page_entries)
        time.sleep(max(0.0, args.sleep_sec))

    rows = [to_training_row(rec, args.max_abstract_chars) for rec in selected]
    if not rows:
        raise SystemExit(
            "No rows collected from arXiv. "
            "Broaden --terms/--categories, increase --since_days, or remove --require_license."
        )
    out_path = Path(args.out)
    meta_path = Path(args.meta_out)

    n_rows = write_jsonl(out_path, rows)
    meta_rows = [
        {
            "source": "arxiv",
            "entry_id": rec.entry_id,
            "title": rec.title,
            "published": rec.published,
            "updated": rec.updated,
            "categories": rec.categories,
            "authors": rec.authors,
            "url": rec.link,
            "license": rec.license_url,
        }
        for rec in selected
    ]
    n_meta = write_jsonl(meta_path, meta_rows)

    print(
        "Done: "
        f"rows={n_rows} meta={n_meta} "
        f"api_total={total_from_api if total_from_api is not None else 'unknown'} "
        f"output={out_path}"
    )


if __name__ == "__main__":
    main()
