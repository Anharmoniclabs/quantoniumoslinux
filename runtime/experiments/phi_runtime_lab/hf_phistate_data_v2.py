#!/usr/bin/env python3
"""Stream bounded Hugging Face code + educational text to PhiState JSONL.
Uses Parquet directly to bypass legacy CodeParrot dataset scripts.
Run: python hf_phistate_data_v2.py --out /tmp/hf_smoke --code-mb 2 --general-mb 2
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

CODE = "codeparrot/github-code"
GENERAL = "HuggingFaceFW/fineweb-edu"
LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "0bsd", "cc0-1.0", "unlicense"}
SECRET = re.compile(r"(?im)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?:api[_-]?key|secret[_-]?key|password)\\s*[:=]\\s*['\"][^'\"]{8,}")

def run(args):
    from datasets import load_dataset
    dest = Path(args.out).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if any((dest / f"{k}.jsonl").exists() for k in ("code", "general")):
        raise SystemExit("Output already exists: choose a fresh --out path")
    # Direct parquet, not load_dataset(CODE), because that invokes an unsupported script.
    streams = {
        "code": lambda: load_dataset("parquet", data_files={"train": "hf://datasets/codeparrot/github-code/data/train-00000-of-01126.parquet"}, split="train", streaming=True),
        "general": lambda: load_dataset(GENERAL, name="sample-10BT", split="train", streaming=True),
    }
    summary = {}
    seen = set()
    for kind in ("code", "general"):
        cap = int(getattr(args, kind + "_mb") * 1048576)
        used = accepted = scanned = 0
        with (dest / (kind + ".jsonl")).open("w", encoding="utf-8") as output:
            for row in streams[kind]():
                scanned += 1
                if scanned > args.max_scan:
                    break
                if kind == "code":
                    license_id = str(row.get("license") or "").lower().strip()
                    if license_id not in LICENSES:
                        continue
                    # Raw Parquet exposes content/path/license; the legacy loader
                    # derived language from the filename extension.
                    path = str(row.get("path") or "").lower()
                    exts = {
                        ".py": "python", ".js": "javascript",
                        ".ts": "typescript", ".tsx": "typescript",
                        ".rs": "rust", ".c": "c", ".h": "c",
                        ".cpp": "c++", ".cc": "c++", ".cxx": "c++",
                        ".hpp": "c++", ".hh": "c++", ".hxx": "c++",
                    }
                    language = next((lang for ext, lang in exts.items() if path.endswith(ext)), "")
                    if language not in args.languages:
                        continue
                    body = row.get("content") or row.get("code")
                    source = f"{row.get('repo_name', '')}/{row.get('path', '')}"
                else:
                    license_id = "odc-by"
                    language = "english"
                    body = row.get("text")
                    source = str(row.get("url") or row.get("id") or "")
                if not isinstance(body, str):
                    continue
                body = body.replace("\x00", "").strip()
                raw = body.encode("utf-8")
                if len(raw) < 160 or len(raw) > args.max_document_kb * 1024 or SECRET.search(body):
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                if digest in seen:
                    continue
                if used + len(raw) > cap:
                    break
                seen.add(digest)
                output.write(json.dumps({"text": body, "source": source, "license": license_id, "language": language, "sha256": digest, "dataset": CODE if kind == "code" else GENERAL}, ensure_ascii=False) + "\n")
                used += len(raw)
                accepted += 1
                if accepted % 100 == 0:
                    print(f"{kind}: {accepted} docs, {used / 1048576:.2f} MiB", flush=True)
        summary[kind] = {"documents": accepted, "bytes": used, "scanned": scanned}
        print(kind, summary[kind], flush=True)
        if accepted == 0:
            raise SystemExit(f"No {kind} records passed filters. Check the HF source schema and license fields.")
    (dest / "sources_manifest.json").write_text(json.dumps({"sources": summary, "code_dataset": CODE, "general_dataset": GENERAL, "note": "Review source terms and attribution. This is not a PII scrub. New tokenizer/corpus requires new training run."}, indent=2) + "\n")
    print("SAVED", dest, flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--code-mb", type=float, default=2)
    p.add_argument("--general-mb", type=float, default=2)
    p.add_argument("--max-scan", type=int, default=250000)
    p.add_argument("--max-document-kb", type=int, default=192)
    p.add_argument("--languages", nargs="+", default=["python", "javascript", "typescript", "rust", "c", "c++"])
    args = p.parse_args()
    if args.code_mb <= 0 or args.general_mb <= 0:
        p.error("limits must be positive")
    args.languages = {x.lower() for x in args.languages}
    run(args)
