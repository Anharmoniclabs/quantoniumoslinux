# quanton

QuantoniumOS runtime — the Tauri desktop shell, Rust engines (Resonart image gen, QuantDiff U-Net, RFTMW kernels), Qwen 2.5 Coder 3B base (RFTMW-packed), and the Eldrae SmartWeights overlays.

## Layout

| Path | What |
|---|---|
| `desktop-shell/` | Tauri v2 app: React + Vite frontend, Rust backend with engine dispatch |
| `desktop-shell/src-tauri/src/engines/` | Rust engines — `resonart` (spectral renderer), `quantdiff` (DDIM + U-Net), `rftmw/`, `crypto/` |
| `smartweights/` | Eldrae SmartWeights packs (core + imagegen + coding-pro + guild-ops + safety-guard), schema, Ollama Modelfile |
| `src/resonart/` | Python Resonart (training + reference DDIM/U-Net) |
| `src/aethercanvas/` | Original procedural renderer (Python ancestor of Resonart) |
| `src/apps/` | Python sidecar: `qshll_chatbox.py` + trainer + Ollama client |
| `qwen2.5-coder_3b.rftmwpk` | RFTMW-packed Qwen 2.5 Coder 3B base (2 GB, Git LFS) |
| `tokenizer.json` | Qwen 3B tokenizer |
| `qshellchatbox.ps1` | Python chatbox launcher |
| `quantoniumos-bootstrap.sh` | One-shot installer |

## Launch

**Desktop shell (main OS):**
```
cd desktop-shell
npm install
npm run tauri dev
```

**Python chatbox sidecar:**
```
./qshellchatbox.ps1
```

## Requirements

- Rust toolchain (stable)
- Node 18+
- Python 3.11+
- Ollama with `eldrae-smartweights` model built from `smartweights/ollama/Modelfile.eldrae-smartweights`
- Git LFS (for the 2 GB RFTMW pack)

## SmartWeights

Five overlay packs under `smartweights/packs/`. Core is a LoRA manifest (planned 14.96M trainable params over 5 disjoint-module phases); the other four are prompt-overlay knowledge snapshots.

See `smartweights/schema/smartweight.schema.json` for the manifest contract.

## Constraint

Everything runs locally. No external model APIs, no HuggingFace downloads at inference, no cloud calls. See `desktop-shell/docs/RESONART.md` (in source repo) for the image-pipeline provenance.
