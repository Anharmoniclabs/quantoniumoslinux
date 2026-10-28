#!/usr/bin/env python3
"""CUDA readiness / throughput probe for a private PhiState-MoE checkpoint.

No user dataset needed; no checkpoint written or altered. Tests real forward,
loss, backward, and optimizer work on synthetic token IDs. Used to decide if
the existing locally developed model can be moved to a Colab CUDA runtime
before spending paid credits on extended training.
"""
import argparse
import gc
import json
import time

import torch
import torch.nn.functional as F


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--context",type=int,default=128)
    p.add_argument("--warmup",type=int,default=2)
    p.add_argument("--steps",type=int,default=8)
    a=p.parse_args()
    if min(a.batch,a.context,a.steps)<1 or a.warmup<0:
        p.error("batch/context/steps must be positive and warmup>=0")
    if a.device=="cuda" and not torch.cuda.is_available():
        p.error("No CUDA GPU: choose Runtime > Change runtime type > GPU")
    device=torch.device(a.device)
    print("DEVICE",str(device),torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU",flush=True)
    from moe_train import ckpt_load
    model, ckpt=ckpt_load(a.checkpoint)
    print("CHECKPOINT",{"step":ckpt.get("step"),"config":ckpt.get("config",{})},flush=True)
    ckpt.clear()
    del ckpt
    gc.collect()
    try:
        model=model.to(device).train()
        vocab=int(model.cfg.vocab)
        x=torch.randint(0,vocab,(a.batch,a.context),device=device)
        y=torch.randint(0,vocab,(a.batch,a.context),device=device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.0,foreach=False)
        if device.type=="cuda":torch.cuda.reset_peak_memory_stats()
        times=[]
        for i in range(a.warmup+a.steps):
            if device.type=="cuda":torch.cuda.synchronize()
            start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            if logits.shape!=(a.batch,a.context,vocab):
                raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")
            loss=F.cross_entropy(logits.reshape(-1,vocab),y.reshape(-1))
            if not torch.isfinite(loss):raise RuntimeError("Non-finite synthetic probe loss")
            loss.backward()
            optimizer.step() # lr=0: no weight updates
            if device.type=="cuda":torch.cuda.synchronize()
            duration=time.perf_counter()-start
            if i>=a.warmup:times.append(duration)
            print(json.dumps({"step":i+1,"warmup":i<a.warmup,
                              "seconds":round(duration,5),"loss":round(float(loss.item()),5)}),flush=True)
        result={"device":a.device,"batch":a.batch,"context":a.context,
                "benchmark_steps":a.steps,
                "tokens_per_second":round(a.batch*a.context*len(times)/sum(times),2),
                "seconds_per_step":round(sum(times)/len(times),5),
                "peak_gpu_mib":round(torch.cuda.max_memory_allocated()/1048576,1)
                  if device.type=="cuda" else None,
                "note":"Synthetic tokens; optimizer lr=0; not real training quality or elapsed end-to-end throughput."}
        print("GPU PROBE PASS",json.dumps(result,indent=2),flush=True)
    except Exception:
        print("GPU PROBE FAILED: model source may contain CPU-only tensor allocations or unsupported kernels. Do NOT start paid GPU training until the error is fixed.",flush=True)
        raise


if __name__=="__main__":
    main()
