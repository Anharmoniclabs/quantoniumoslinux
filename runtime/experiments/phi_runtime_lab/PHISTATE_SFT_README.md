# PhiState-MoE: continue 5k to 10k, then public Q&A/chat SFT

NO new tokenizer. NO random weight restart. NO modification to the original
pretraining corpus or its sealed test split. CPU/PyTorch training only;
native RFTMW remains a separate inference export/parity test after SFT.

The 38.3M model still generates incoherent text at 5k steps. Pretraining
supplies language modeling; chat SFT teaches response formatting.
SFT cannot create broad knowledge or reasoning by itself.

## Public sources and licensing

- HuggingFaceH4/ultrachat_200k (MIT; model-generated dialogues)
- OpenAssistant/oasst1 (Apache-2.0; human-authored/annotated message trees)

The downloader only uses UltraChat train_sft and OASST1 train, splits
whole conversations by stable hash, deduplicates exact messages, and filters
some suspicious/PII strings. It does not touch reserved test splits.
Basic filtering is NOT guaranteed privacy/safety purification; review
licenses, attributions, provenance, and sample quality. No personal
ChatGPT conversations are accessed automatically.

## Installing on your existing Codespaces feature branch

These scripts are in separate commits on the remote main branch because
feat/phistate-rftmw-native exists only in your Codespace. Run git fetch
origin main and cherry-pick the new commits listed in the ChatGPT reply.
This does not replace your existing moe_train.py or moe_model.py.

## Stage A — inspect 5k validation results

    cd /workspaces/quantoniumoslinux
    source .phistate-env/bin/activate
    LAB=runtime/experiments/phi_runtime_lab
    DATA="$LAB/datasets/hf_coding_knowledge_v2"
    LAST="$LAB/checkpoints/moe_38m_hf_v2.last.pt"
    tail -n 6 "$LAB/checkpoints/moe_38m_hf_v2.metrics.jsonl"

Proceed if val_code_bpt, val_general_bpt and/or val_mean_bpt still
improve. The runner uses last.pt (not necessarily the best checkpoint).
If the old optimizer state was not stored, the weights and RNG can resume,
but the optimizer starts fresh. This is not a bit-exact continuation.

## Stage B — additional pretraining to 10k TOTAL steps

    mkdir -p ~/quantonium-checkpoints
    python -u "$LAB/pretrain_continue_10k.py" \
      --checkpoint "$LAST" --data-dir "$DATA" \
      --out ~/quantonium-checkpoints/moe_38m_hf_10k.pt \
      --target-steps 10000 --eval-every 250 --eval-batches 16 \
      --threads 4 --dry-run

After inspecting the dry-run's steps/fingerprint, rerun without --dry-run.
The output location is different, protecting the original 5k checkpoint.
The new best and latest files consume additional disk space (budget
several hundred MiB). Latest includes optimizer state by default.
Optionally run a smaller first continuation to 6k with a unique output
prefix and evaluate before proceeding to 10k.

## Stage C — stream and prepare public chat/Q&A data

    source .phistate-env/bin/activate
    python -m pip install datasets huggingface_hub sentencepiece
    python -u "$LAB/hf_instruction_data.py" \
      --out ~/quantonium-data/hf_chat_v1 \
      --ultrachat 8000 --oasst 2000

Target is about 10k examples subject to filtering; resulting actual
counts and licenses are recorded in the local manifest. The downloader
creates train.jsonl and val.jsonl with conversation-disjoint splits.
Data is stored under home, not in the repository. The old tokenizer is
preserved; this stage does NOT call moe_data.py.

## Stage D — supervised chat tuning with masked prompt loss

    PRETRAIN="$HOME/quantonium-checkpoints/moe_38m_hf_10k.pt"
    CHAT="$HOME/quantonium-data/hf_chat_v1"

    python -u "$LAB/moe_chat_sft.py" \
      --checkpoint "$PRETRAIN" --data-dir "$DATA" \
      --sft-dir "$CHAT" \
      --out ~/quantonium-checkpoints/moe_38m_chat_smoke.pt \
      --steps 10 --eval-every 10 --val-examples 20 --threads 4

If the smoke test succeeds, run the full SFT experiment using a DIFFERENT
checkpoint path, still initialized from the same 10k pretrained weights:

    python -u "$LAB/moe_chat_sft.py" \
      --checkpoint "$PRETRAIN" --data-dir "$DATA" \
      --sft-dir "$CHAT" \
      --out ~/quantonium-checkpoints/moe_38m_chat_sft.pt \
      --steps 500 --eval-every 100 --val-examples 80 \
      --context 256 --batch 1 --accum 2 \
      --optimizer adafactor --threads 4

User/context prompt positions are masked (labels -100). Only assistant
target tokens and EOS receive language-model loss, plus the MoE router
penalty. Uses original 8,192-token SentencePiece tokenizer. The model
still may be incoherent after SFT; measure unseen generation and QA.

## Evaluate generated chat

    python -u "$LAB/moe_train.py" generate \
      --checkpoint ~/quantonium-checkpoints/moe_38m_chat_sft.pt \
      --data-dir "$DATA" \
      --prompt 'User: Explain what an operating system does.
    Assistant: ' \
      --new-tokens 100 --threads 4

The SFT checkpoint deliberately uses PhiStateMoE-v1 format for
compatibility with the original generator and C++ model export tools.
Compare against 5k/10k checkpoints on identical prompts, held-out QA and
simple code-execution tests. NEVER use sealed test data for model selection.

## Resources

No paid GPU or cloud training. Data downloads stream only capped samples.
Checkpoints and datasets remain in $HOME, never added to Git by these scripts.
The model remains a research prototype, not a dependable assistant.
