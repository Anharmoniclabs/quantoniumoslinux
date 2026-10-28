#!/usr/bin/env python3
"""Stream bounded public Q&A/chat SFT examples for PhiState-MoE.

Sources: HuggingFaceH4/ultrachat_200k (MIT) and OpenAssistant/oasst1
(Apache-2.0). Data remains local. Does NOT train a new tokenizer or modify
the pretraining dataset/checkpoints. Conversations are grouped when split.
Review source model/data terms and filter sensitive content before reuse.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SOURCE_LICENSES = {
    "HuggingFaceH4/ultrachat_200k": "mit",
    "OpenAssistant/oasst1": "apache-2.0",
}
SENSITIVE = re.compile(
    r"(?i)(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    r"|(?:api[_ -]?key|password|secret[_ -]?key)\s*[:=]\s*['\"]?.{10,}"
    r"|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b)")
CODE_WORD = re.compile(r"\b(python|javascript|typescript|rust|function|"
                       r"debug|program|code|script|algorithm|compile|error)\b", re.I)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sanitize_turns(messages, args):
    if not isinstance(messages, list):
        return None
    result = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        role = "user" if role == "prompter" else role
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", msg.get("text"))
        if not isinstance(content, str):
            return None
        content = content.replace("\x00", "").strip()
        if not content or len(content) > args.max_turn_chars or SENSITIVE.search(content):
            return None
        if result and result[-1]["role"] == role:
            return None
        result.append({"role": role, "content": content})
    if len(result) < 2 or result[0]["role"] != "user":
        return None
    if result[-1]["role"] != "assistant":
        return None
    answer = result[-1]["content"]
    if not args.min_answer_chars <= len(answer) <= args.max_answer_chars:
        return None
    # Keep the recent context and always start at a user message.
    result = result[-args.max_turns:]
    while result and result[0]["role"] != "user":
        result.pop(0)
    if len(result) < 2 or result[-1]["role"] != "assistant":
        return None
    return result


def ultra_stream(args):
    from datasets import load_dataset
    # train_sft only. Do not use test_sft as training examples.
    stream = load_dataset("HuggingFaceH4/ultrachat_200k",
                          split="train_sft", streaming=True)
    # Shuffle BEFORE take; shuffle is bounded by a small buffer.
    stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    scanned = 0
    for row in stream:
        scanned += 1
        if scanned > args.max_scan: break
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        group = str(row.get("prompt_id") or digest(json.dumps(messages,sort_keys=True)))
        # Select one assistant answer per conversation to reduce near-duplicates.
        latest = None
        for idx in range(1,len(messages)):
            if messages[idx].get("role") == "assistant":
                candidate = sanitize_turns(messages[:idx+1],args)
                if candidate: latest = candidate
        if latest:
            yield group, latest


def oasst_stream(args):
    from datasets import load_dataset
    # OASST1 is a message tree. Reconstruct parent chains; do not mix
    # unrelated turns or leak whole trees between train and validation.
    records = {}
    for row in load_dataset("OpenAssistant/oasst1",split="train",streaming=True):
        mid = row.get("message_id")
        if mid: records[mid] = row
    accepted = 0
    for mid,row in sorted(records.items(),key=lambda it:digest(it[0])):
        if row.get("role") != "assistant" or row.get("deleted") is True:
            continue
        if row.get("lang") != "en" or row.get("review_result") is False:
            continue
        if row.get("rank") not in (None,0):
            continue
        path = []
        seen = set()
        current = row
        for _ in range(args.max_turns+2):
            ident = current.get("message_id")
            if not ident or ident in seen: break
            seen.add(ident)
            path.append({"role": current.get("role"), "content": current.get("text")})
            parent_id = current.get("parent_id")
            if not parent_id: break
            current = records.get(parent_id)
            if current is None: break
        path.reverse()
        result = sanitize_turns(path,args)
        if not result: continue
        group = str(row.get("message_tree_id") or row.get("parent_id") or mid)
        yield group,result
        accepted += 1
        if accepted >= args.oasst*4: break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--ultrachat",type=int,default=8000)
    p.add_argument("--oasst",type=int,default=2000)
    p.add_argument("--seed",type=int,default=20261028)
    p.add_argument("--val-percent",type=int,default=5)
    p.add_argument("--max-scan",type=int,default=250000)
    p.add_argument("--shuffle-buffer",type=int,default=2000)
    p.add_argument("--max-turns",type=int,default=6)
    p.add_argument("--max-turn-chars",type=int,default=1600)
    p.add_argument("--min-answer-chars",type=int,default=30)
    p.add_argument("--max-answer-chars",type=int,default=1100)
    a=p.parse_args()
    if a.ultrachat<0 or a.oasst<0 or not 1<=a.val_percent<=30:
        p.error("source counts nonnegative; val-percent between 1 and 30")
    if a.ultrachat+a.oasst<10: p.error("need at least 10 records")
    out=Path(a.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    targets=[out/"train.jsonl",out/"val.jsonl",out/"manifest.json"]
    if any(x.exists() for x in targets):
        p.error("Output exists. Use a fresh --out folder; never overwrite an evaluation split.")
    handles={}
    counts=Counter()
    hashes=set()
    try:
        handles={split:(out/(split+".jsonl")).open("w",encoding="utf8")
                 for split in ("train","val")}
        for source,cap,stream_fn in (
            ("HuggingFaceH4/ultrachat_200k",a.ultrachat,ultra_stream),
            ("OpenAssistant/oasst1",a.oasst,oasst_stream)):
            if cap==0: continue
            for group,messages in stream_fn(a):
                canonical=json.dumps(messages,ensure_ascii=False,sort_keys=True)
                key=digest(canonical)
                if key in hashes: continue
                hashes.add(key)
                h=int(digest(source+"|"+group)[:8],16)%100
                split="val" if h<a.val_percent else "train"
                request=" ".join(m["content"] for m in messages if m["role"]=="user")
                domain="coding" if CODE_WORD.search(request) else "general"
                record={"id":key,"source":source,"license":SOURCE_LICENSES[source],
                        "conversation_group":digest(group),
                        "domain":domain,"messages":messages}
                handles[split].write(json.dumps(record,ensure_ascii=False)+"\n")
                counts[(source,split)]+=1
                counts[(domain,split)]+=1
                counts[(source,"total")]+=1
                if counts[(source,"total")]%500==0:
                    print(source,counts[(source,"total")],"saved",flush=True)
                if counts[(source,"total")]>=cap:break
        for v in handles.values():v.flush()
    finally:
        for v in handles.values():v.close()
    if counts[("general","train")]+counts[("coding","train")]<100:
        raise SystemExit("Too few training examples, check sources and filtering.")
    if counts[("general","val")]+counts[("coding","val")]<10:
        raise SystemExit("Too few validation examples; use more data or increase --val-percent.")
    source_totals={src:{sp:counts[(src,sp)] for sp in ("train","val","total")}
                   for src in SOURCE_LICENSES}
    report={"format":"PhiStateChatPairs-v1","seed":a.seed,
            "split":"Stable hash of full conversation tree/prompt id, not message",
            "sources":source_totals,"train_total":sum(counts[(src,"train")] for src in SOURCE_LICENSES),
            "val_total":sum(counts[(src,"val")] for src in SOURCE_LICENSES),
            "domain_count":{d:{sp:counts[(d,sp)] for sp in ("train","val")}
                            for d in ("coding","general")},
            "source_licenses":SOURCE_LICENSES,
            "tokenizer":"USE EXISTING hf_coding_knowledge_v2/tokenizer.model; do not retrain.",
            "safety":"Basic secret/email filter only. Review PII, content, licenses and synthetic output limitations manually.",
            "sealed_test":"Reserved UltraChat test_sft and OASST1 test; never fetched."}
    (out/"manifest.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print("SAVED",out,json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
