#!/usr/bin/env python3
"""Answer-only chat supervised fine tuning of an existing PhiState-MoE checkpoint.

Compatible with moe_train.py PhiStateMoE-v1 and the SAME original SentencePiece
tokenizer/corpus fingerprint. Does not train on the new SFT data as raw language
model text; prompt token losses are masked. Keeps pretraining weights untouched.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import sentencepiece as spm
import torch
import torch.nn.functional as F

from moe_train import ckpt_load, load_data, atomic_checkpoint, setup

LN2 = math.log(2)


def file_digest(path):
    h=hashlib.sha256()
    with path.open("rb") as fd:
        for buf in iter(lambda: fd.read(1024*1024),b""):
            h.update(buf)
    return h.hexdigest()


def encode_pair(row, sp, context, max_prompt_tokens):
    messages=row.get("messages")
    if not isinstance(messages,list) or len(messages)<2:
        return None
    if messages[-1].get("role")!="assistant":return None
    prompt_parts=[]
    for item in messages[:-1]:
        role=item.get("role")
        if role not in ("user","assistant"):return None
        text=item.get("content")
        if not isinstance(text,str):return None
        label="User" if role=="user" else "Assistant"
        prompt_parts.append(f"{label}: {text}\n")
    prompt="".join(prompt_parts)+"Assistant: "
    answer=messages[-1].get("content","")
    if not isinstance(answer,str) or not answer.strip():
        return None
    prompt_ids=sp.encode(prompt,out_type=int)
    answer_ids=sp.encode(answer,out_type=int)
    if not prompt_ids or not answer_ids:return None
    # Retain the most recent conversation context; reserve space for answer.
    budget=min(max_prompt_tokens,context-12)
    prompt_ids=prompt_ids[-budget:]
    answer_ids=answer_ids[:context-len(prompt_ids)-1]
    if len(answer_ids)<8:return None
    ids=prompt_ids+answer_ids+[sp.eos_id()]
    x=torch.tensor(ids[:-1],dtype=torch.long)
    y=torch.tensor(ids[1:],dtype=torch.long)
    # y[j] predicts ids[j+1]; only train when the next token is answer/EOS.
    y[:max(0,len(prompt_ids)-1)]=-100
    if (y!=-100).sum().item()<8:return None
    return x,y


def load_examples(path,sp,context,max_prompt_tokens):
    rows=[]
    with path.open("r",encoding="utf8") as fd:
        for line in fd:
            item=json.loads(line)
            pair=encode_pair(item,sp,context,max_prompt_tokens)
            if pair is not None:rows.append(pair)
    if not rows: raise RuntimeError("No usable examples from "+str(path))
    return rows


def pack(batch,context,eos):
    x=torch.full((len(batch),context),eos,dtype=torch.long)
    y=torch.full((len(batch),context),-100,dtype=torch.long)
    for i,(src,labels) in enumerate(batch):
        x[i,:len(src)]=src
        y[i,:len(labels)]=labels
    return x,y


@torch.inference_mode()
def evaluate(model,examples,context,eos,batch_size,limit):
    model.eval()
    count=total_nll=0
    for start in range(0,min(limit,len(examples)),batch_size):
        entries=examples[start:min(start+batch_size,len(examples),limit)]
        x,y=pack(entries,context,eos)
        logits=model(x)
        loss=F.cross_entropy(logits.reshape(-1,model.cfg.vocab),
                             y.reshape(-1),ignore_index=-100,reduction="sum")
        total_nll+=float(loss.item())
        count+=int((y!=-100).sum().item())
    return {"val_answer_bpt":round(total_nll/max(1,count)/LN2,5),
            "val_answer_tokens":count,
            "val_examples":min(limit,len(examples))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True,help="Existing 10k pretraining checkpoint")
    p.add_argument("--data-dir",required=True,help="Original HF V2 corpus (tokenizer and fingerprint)")
    p.add_argument("--sft-dir",required=True,help="New hf_instruction_data.py train/val files")
    p.add_argument("--out",required=True,help="NEW fine-tuned checkpoint, never original")
    p.add_argument("--steps",type=int,default=500,help="Total SFT optimizer steps")
    p.add_argument("--context",type=int,default=256)
    p.add_argument("--max-prompt-tokens",type=int,default=128)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--accum",type=int,default=2)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--seed",type=int,default=20261028)
    p.add_argument("--lr",type=float,default=0.0001)
    p.add_argument("--optimizer",choices=["adafactor","adamw"],default="adafactor")
    p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--val-examples",type=int,default=80)
    p.add_argument("--clip",type=float,default=1.0)
    p.add_argument("--resume",default="",help="Optional SFT .last.pt; do not confuse with pretrained checkpoint")
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    if min(a.context,a.steps,a.batch,a.accum,a.eval_every,a.threads,a.val_examples)<1:
        p.error("positive context/steps/batch/accum/eval-every/threads/val-examples needed")
    if a.context<64 or not 8<=a.max_prompt_tokens<a.context-8:
        p.error("context>=64 and 8<=max-prompt-tokens<context-8 required")
    setup(a.threads,a.seed)
    original, pretrained=ckpt_load(a.checkpoint)
    pretrain_manifest,_,base=load_data(a.data_dir)
    if pretrained.get("data_fingerprint")!=pretrain_manifest.get("data_fingerprint"):
        p.error("Pretraining corpus fingerprint mismatch")
    if pretrain_manifest.get("tokenizer")!="bpe":
        p.error("BPE/SentencePiece tokenizer required")
    sp=spm.SentencePieceProcessor(model_file=str(base/"tokenizer.model"))
    if original.cfg.vocab!=sp.vocab_size():
        p.error("Vocabulary mismatch: keep the old tokenizer")
    sftdir=Path(a.sft_dir).expanduser().resolve()
    manifest=sftdir/"manifest.json"
    if not manifest.is_file(): p.error("Missing SFT manifest")
    metadata=json.loads(manifest.read_text())
    if metadata.get("format")!="PhiStateChatPairs-v1":
        p.error("Wrong SFT dataset format")
    trainfile=sftdir/"train.jsonl"
    valfile=sftdir/"val.jsonl"
    if not trainfile.is_file() or not valfile.is_file():
        p.error("Missing chat train.jsonl or val.jsonl")
    chat_fingerprint=file_digest(trainfile)+":"+file_digest(valfile)
    training=load_examples(trainfile,sp,a.context,a.max_prompt_tokens)
    validation=load_examples(valfile,sp,a.context,a.max_prompt_tokens)
    out=Path(a.out).expanduser().resolve()
    if out==Path(a.checkpoint).expanduser().resolve():
        p.error("Output would overwrite pretrained checkpoint")
    last=out.with_name(out.stem+".last"+out.suffix)
    if (out.exists() or last.exists()) and not a.resume:
        p.error("Output path exists: specify a new --out or --resume existing SFT latest.")
    model=original
    step=0
    best=float("inf")
    rng=random.Random(a.seed+917)
    if a.resume:
        model,saved=ckpt_load(a.resume)
        if saved.get("training_stage")!="answer_only_sft":
            p.error("Resume file is not SFT checkpoint")
        if saved.get("sft_data_fingerprint")!=chat_fingerprint:
            p.error("SFT dataset fingerprint mismatch")
        if saved["config"]!=pretrained["config"]:
            p.error("Architecture mismatch")
        step=int(saved["step"])
        best=float(saved.get("best_validation_bpt",best))
        if "sft_rng_state" in saved:rng.setstate(saved["sft_rng_state"])
        if "sft_torch_rng" in saved:torch.set_rng_state(saved["sft_torch_rng"])
        print("SFT RESUMED",step,flush=True)
    print("SFT PLAN",json.dumps({"pretraining_checkpoint":str(Path(a.checkpoint).resolve()),
        "pretrained_steps":pretrained.get("step"),
        "sft_start":step,"sft_end":a.steps,
        "parameters":sum(q.numel() for q in model.parameters()),
        "context":a.context,"training_pairs":len(training),"val_pairs":len(validation),
        "trainable_answer_tokens_only":True,"sft_corpus":str(sftdir)},indent=2),flush=True)
    if a.dry_run: return
    if a.optimizer=="adafactor":
        if not hasattr(torch.optim,"Adafactor"):
            raise SystemExit("Adafactor not available in PyTorch; use --optimizer adamw")
        optimizer=torch.optim.Adafactor(model.parameters(),lr=a.lr,weight_decay=0.,foreach=False)
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.,foreach=False)
    if a.resume and "optimizer_state" in saved:
        optimizer.load_state_dict(saved["optimizer_state"])
    model.train()
    begin=time.perf_counter()
    out.parent.mkdir(parents=True,exist_ok=True)
    for index in range(step+1,a.steps+1):
        optimizer.zero_grad(set_to_none=True)
        nll=0.0
        for _ in range(a.accum):
            chosen=[training[rng.randrange(len(training))] for _ in range(a.batch)]
            x,y=pack(chosen,a.context,sp.eos_id())
            logits=model(x)
            lm=F.cross_entropy(logits.reshape(-1,model.cfg.vocab),
                               y.reshape(-1),ignore_index=-100)
            router,_=model.router_penalty()
            loss=(lm+router)/a.accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite SFT loss at {index}")
            loss.backward()
            nll+=float(lm.detach().item())/a.accum
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),a.clip))
        optimizer.step()
        if index==1 or index%a.eval_every==0 or index==a.steps:
            info=evaluate(model,validation,a.context,sp.eos_id(),a.batch,a.val_examples)
            current=info["val_answer_bpt"]
            improved=current<best
            best=min(best,current)
            report={"sft_step":index,"train_answer_bpt":round(nll/LN2,5),
                **info,"best_val_answer_bpt":round(best,5),
                "grad_norm":round(grad,4),
                "elapsed_seconds":round(time.perf_counter()-begin,2)}
            print("SFT RESULT",json.dumps(report),flush=True)
            record={
                "format":"PhiStateMoE-v1","training_stage":"answer_only_sft",
                "config":pretrained["config"],"state_dict":model.state_dict(),
                "step":index,"sft_steps":index,
                "pretrain_step":pretrained.get("step"),
                "best_validation_bpt":best,
                "data_fingerprint":pretrained["data_fingerprint"],
                "tokenizer":pretrained["tokenizer"],
                "sft_data_fingerprint":chat_fingerprint,
                "sft_rng_state":rng.getstate(),"sft_torch_rng":torch.get_rng_state(),
                "train_args":vars(a)}
            if improved:
                atomic_checkpoint(out,record)
                print("SFT BEST",out,flush=True)
            record["optimizer_state"]=optimizer.state_dict()
            atomic_checkpoint(last,record)
            print("SFT LAST",last,flush=True)
            model.train()
    print("SFT FINISHED",a.steps,"BEST",best,flush=True)


if __name__=="__main__":
    main()
