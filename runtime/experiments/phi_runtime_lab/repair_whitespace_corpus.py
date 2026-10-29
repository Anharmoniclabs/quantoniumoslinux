#!/usr/bin/env python3
"""Recover original document membership from token streams; rebuild whitespace.

No split-hash guessing, downloads, training, or checkpoint writes. Run in Colab:
python -u repair_whitespace_corpus.py --base-data ORIGINAL --repair-dir REPAIR
The recovered raw JSONL files and corrected tokenizer must already be in REPAIR.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

SPLITS = ("train", "val", "test")
DOMAINS = ("code", "general")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def log(message):
    print(message, flush=True)


def pack(ids, dtype):
    return np.asarray(ids, dtype=dtype).tobytes()


def recover_domain(records, streams, stats, modes, itemsize):
    """Account for every byte and document, refusing ambiguous text matches.

records: [{'text': normalized raw text, 'tokens': original encoded bytes}].
Boundary modes affect proof only; they never alter the original token files.
"""
    index = defaultdict(list)
    key_lengths = set()
    for number, record in enumerate(records):
        encoded = record["tokens"]
        require(encoded, f"Document {number} encoded to no tokens")
        n = min(8 * itemsize, len(encoded))
        key_lengths.add(n)
        index[encoded[:n]].append(number)
    successes, failures = [], {}
    for name, prefix, suffix in modes:
        used, assignments = set(), {}
        try:
            for split in SPLITS:
                data, position, order, byte_count = streams[split], 0, [], 0
                while position < len(data):
                    require(data.startswith(prefix, position),
                            f"{split}: boundary mismatch at token {position // itemsize}")
                    start = position + len(prefix)
                    matches = []
                    for n in key_lengths:
                        for number in index.get(data[start:start+n], ()):
                            if number in used:
                                continue
                            encoded = records[number]["tokens"]
                            end = start + len(encoded)
                            if data.startswith(encoded, start) and data.startswith(suffix, end):
                                matches.append(number)
                    require(matches, f"{split}: no document at token {start // itemsize}")
                    # Identical raw texts are interchangeable; differently formatted
                    # texts collapsed by the old tokenizer must not be guessed.
                    require(len({records[i]['text'] for i in matches}) == 1,
                            f"{split}: ambiguous raw documents at token {start // itemsize}")
                    number = min(matches)
                    record = records[number]
                    position = start + len(record["tokens"]) + len(suffix)
                    used.add(number)
                    order.append(number)
                    byte_count += len(record["text"].encode("utf-8"))
                require(len(order) == stats[f"{split}_documents"],
                        f"{split}: original document count differs")
                require(byte_count == stats[f"{split}_bytes"],
                        f"{split}: original text byte count differs")
                assignments[split] = order
            require(len(used) == len(records), "Some recovered documents were not accounted for")
            successes.append((name, prefix, suffix, assignments))
        except RuntimeError as error:
            failures[name] = str(error)
    require(successes, "Cannot prove original membership. No files changed. Details: "
            + json.dumps(failures, sort_keys=True))
    first = successes[0]
    require(all(result[3] == first[3] for result in successes),
            "Multiple boundary policies give different assignments; refusing to guess")
    return first


def boundary_modes(tokenizer, dtype):
    modes, seen = [], set()
    bos, eos, pad = tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.pad_id()
    options = [("none", [], [])]
    if eos >= 0:
        options.append(("EOS", [], [eos]))
    if bos >= 0:
        options.append(("BOS", [bos], []))
    if bos >= 0 and eos >= 0:
        options.append(("BOS+EOS", [bos], [eos]))
    if pad >= 0:
        options.append(("PAD separator", [], [pad]))
    for text in ("\n", "\n\n"):
        options.append((f"encoded separator {text!r}", [], tokenizer.encode(text)))
    for name, before, after in options:
        prefix, suffix = pack(before, dtype), pack(after, dtype)
        if (prefix, suffix) not in seen:
            seen.add((prefix, suffix))
            modes.append((name, prefix, suffix))
    return modes


def verify_output(path, source_fingerprint, tokenizer_hash, raw_hashes):
    m = json.loads((path / "manifest.json").read_text())
    lineage = m["whitespace_repair"]
    require(lineage["original_data_fingerprint"] == source_fingerprint,
            "Existing output belongs to a different original corpus")
    require(lineage["raw_sha256"] == raw_hashes, "Existing output uses different raw files")
    require(sha(path / "tokenizer.model") == tokenizer_hash,
            "Existing repaired tokenizer differs")
    for key, spec in m["bins"].items():
        fp = path / spec["file"]
        require(fp.stat().st_size == spec["tokens"] * np.dtype(m["dtype"]).itemsize,
                f"Bad output length: {key}")
        require(sha(fp) == spec["sha256"], f"Bad output hash: {key}")
    require(sha(path / "membership.json") == lineage["membership_sha256"],
            "Membership proof changed")
    fingerprint = m.pop("data_fingerprint")
    actual = hashlib.sha256(json.dumps(m, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(fingerprint == actual, "Repaired manifest fingerprint differs")
    return fingerprint


def run(base, repair, output):
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb

    original = json.loads((base / "manifest.json").read_text())
    require(original["format"] == "PhiMoECorpus-v1" and original["tokenizer"] == "bpe",
            "Expected the original BPE corpus")
    dtype = np.dtype(original["dtype"])
    require(dtype in (np.dtype("uint16"), np.dtype("uint32")), "Unsupported token dtype")
    recovery = json.loads((repair / "raw_recovery_complete.json").read_text())
    require(recovery["original_data_fingerprint"] == original["data_fingerprint"],
            "Recovery marker does not match original corpus")
    raw = Path(recovery["raw_directory"])
    require(output.resolve() != base.resolve() and output.resolve() != repair.resolve(),
            "Output must be a separate directory")
    raw_hashes = {}
    for domain in DOMAINS:
        log(f"Checking saved raw file: {domain}")
        raw_hashes[domain] = sha(raw / f"{domain}.jsonl")
        require(raw_hashes[domain] == recovery["file_sha256"][domain], "Raw recovery file changed")

    old = spm.SentencePieceProcessor(model_file=str(base / "tokenizer.model"))
    fixed = spm.SentencePieceProcessor(model_file=str(repair / "tokenizer.model"))
    a, b = pb.ModelProto(), pb.ModelProto()
    a.ParseFromString(old.serialized_model_proto())
    b.ParseFromString(fixed.serialized_model_proto())
    require([p.SerializeToString() for p in a.pieces] == [p.SerializeToString() for p in b.pieces],
            "Tokenizer vocabulary or IDs changed")
    require(old.vocab_size() == fixed.vocab_size() == original["vocab"], "Vocabulary mismatch")
    require((old.bos_id(), old.eos_id(), old.pad_id(), old.unk_id()) ==
            (fixed.bos_id(), fixed.eos_id(), fixed.pad_id(), fixed.unk_id()), "Special IDs changed")
    examples = ["def add(a, b):\n    return a + b\n", "def f():\n\treturn 42\n",
                "first\n\nthird\n", "    indent\n", 'x = "two  spaces"\n']
    for text in examples:
        require(fixed.decode(fixed.encode(text)) == text, "Corrected tokenizer loses whitespace")
    fixed_hash = sha(repair / "tokenizer.model")
    if output.exists():
        fingerprint = verify_output(output, original["data_fingerprint"], fixed_hash, raw_hashes)
        log(f"ALREADY COMPLETE: {output}\nFingerprint: {fingerprint}")
        return

    modes = boundary_modes(old, dtype)
    membership = {"proof": "Exact original token streams, document counts and normalized text bytes",
                  "original_data_fingerprint": original["data_fingerprint"], "domains": {}}
    new_bins = {}
    # Local staging avoids thousands of tiny Drive writes. Nothing is published
    # until both domains have reproduced all their original token streams.
    with tempfile.TemporaryDirectory(prefix="phistate_whitespace_") as temp:
        stage = Path(temp)
        for domain in DOMAINS:
            records = []
            log(f"Encoding recovered {domain} with ORIGINAL tokenizer (CPU only)...")
            with (raw / f"{domain}.jsonl").open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    row = json.loads(line)
                    body = row["text"]
                    require(hashlib.sha256(body.encode()).hexdigest() == row["sha256"],
                            f"Raw document hash mismatch: {domain}/{number}")
                    text = body.replace("\r\n", "\n").replace("\r", "\n")
                    records.append({"text": text, "tokens": pack(old.encode(text), dtype)})
                    if number % 250 == 0:
                        log(f"  {domain}: {number:,} documents encoded")
            streams = {}
            for split in SPLITS:
                key = f"{split}_{domain}"
                spec = original["bins"][key]
                data = (base / spec["file"]).read_bytes()
                require(len(data) == spec["tokens"] * dtype.itemsize, f"Original length mismatch: {key}")
                require(hashlib.sha256(data).hexdigest() == spec["sha256"], f"Original hash mismatch: {key}")
                streams[split] = data
            name, prefix, suffix, assignment = recover_domain(
                records, streams, original["documents"][domain], modes, dtype.itemsize)
            del streams
            log(f"PASS {domain}: all three original streams accounted for; boundaries={name}")
            domain_proof = {"boundaries": name, "splits": {}}
            for split in SPLITS:
                key = f"{split}_{domain}"
                token_count, document_hashes = 0, []
                with (stage / f"{key}.bin").open("wb") as stream:
                    for n, number in enumerate(assignment[split], 1):
                        text = records[number]["text"]
                        ids = fixed.encode(text)
                        require(fixed.decode(ids) == text,
                                f"Corrected tokenizer does not round-trip {key} document {n}; not publishing")
                        encoded = prefix + pack(ids, dtype) + suffix
                        stream.write(encoded)
                        token_count += len(encoded) // dtype.itemsize
                        document_hashes.append(hashlib.sha256(text.encode()).hexdigest())
                        if n % 500 == 0:
                            log(f"  rebuilding {key}: {n:,}/{len(assignment[split]):,}")
                new_bins[key] = {"file": f"{key}.bin", "tokens": token_count,
                                 "sha256": sha(stage / f"{key}.bin")}
                domain_proof["splits"][split] = document_hashes
                log(f"SAVED staged {key}: {token_count:,} tokens")
            membership["domains"][domain] = domain_proof
            del records

        (stage / "membership.json").write_text(json.dumps(membership, indent=2) + "\n")
        shutil.copyfile(repair / "tokenizer.model", stage / "tokenizer.model")
        manifest = {
            "format": "PhiMoECorpus-v1", "tokenizer": "bpe", "vocab": original["vocab"],
            "dtype": original["dtype"], "bins": new_bins, "documents": original["documents"],
            "split_policy": "Original membership recovered by exact token-stream matching",
            "notes": "Repaired ORIGINAL corpus only; expanded instruction data is not included. Test split remains held out.",
            "whitespace_repair": {
                "original_data_fingerprint": original["data_fingerprint"],
                "original_tokenizer_sha256": sha(base / "tokenizer.model"),
                "tokenizer_sha256": fixed_hash, "raw_sha256": raw_hashes,
                "membership_sha256": sha(stage / "membership.json"),
                "token_ids_unchanged": True, "all_documents_roundtrip_exact": True,
                "all_original_streams_accounted_for": True,
            },
        }
        manifest["data_fingerprint"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_output(stage, original["data_fingerprint"], fixed_hash, raw_hashes)
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = Path(tempfile.mkdtemp(prefix=output.name + ".pending_", dir=output.parent))
        log(f"Copying verified corpus to {pending}")
        for file in stage.iterdir():
            shutil.copyfile(file, pending / file.name)
        verify_output(pending, original["data_fingerprint"], fixed_hash, raw_hashes)
        require(not output.exists(), "Output appeared during repair; pending copy preserved")
        pending.rename(output)
        verify_output(output, original["data_fingerprint"], fixed_hash, raw_hashes)
    log(f"REPAIR COMPLETE: {output}")
    log("All six original token streams matched; corrected documents round-trip exactly.")
    log("Checkpoints untouched. No training started. Use an explicit corpus-migration continuation, not ordinary --resume.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--repair-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run(args.base_data, args.repair_dir, args.out or args.repair_dir / "corpus_repaired_v1")


if __name__ == "__main__":
    main()
