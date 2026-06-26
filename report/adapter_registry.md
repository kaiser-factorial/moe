# LoRA Adapter Registry (authoritative)

**Purpose:** stop the name collisions. Several adapters share checkpoint names
(`checkpoint-1188`, `lora3`, etc.) but differ in *what they actually trained*.
This file is the single source of truth, verified **2026-06-23** by reading the
real `adapter_model.safetensors` weight keys (not just `adapter_config.json`,
which can mislead because PEFT matches `target_modules` by name suffix).

All adapters: **r = 32, α = 64, dropout = 0.05, bias = none**,
base = `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
(52 layers: 23 Mamba-2, 23 MoE = 128 routed experts + 1 shared each, 6 attention).

## The three structural families

| Family | Attn (q/k/v/o) | Mamba (in/out_proj) | Shared expert (up/down) | Routed experts (128×) | Router/gate | Tensors |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | ✅ | ✅ | ✅ | ❌ | ❌ | 232 |
| **B** | ❌ | ✅ | ✅ | ✅ | ❌ | 11,960 |
| **C** | ✅ | ✅ | ✅ | ✅ | ❌ | 12,008 |

- Family A `target_modules` = explicit list: `q_proj, k_proj, v_proj, o_proj,
  in_proj, out_proj, shared_experts.up_proj, shared_experts.down_proj`.
  (6 attn layers → 48 attn tensors.)
- Family B `target_modules` = regex `.*\.(in_proj|out_proj|up_proj|down_proj)$`.
  The bare `up_proj`/`down_proj` ALSO match the 128 routed experts (23 MoE
  layers × 128 = 5,888 down + 5,888 up). No attention captured.
- Family C = explicit list including both attention **and** bare
  `up_proj`/`down_proj` → attention + Mamba + shared + all routed experts.
- **No adapter trains the router gate.** There is **no shared-experts-ONLY**
  adapter (a half-remembered one that does not exist).

## Which artifact is which

| Where | Slug / repo | Family | Proposed rename | Used in |
|---|---|:---:|---|---|
| HF | `brick-factorial/nemotron-lora-symbolic-reasoning` | A | `…-lora-attn-mamba-shared` | ⭐ **Phase 2/3** (routing capture, divergence, steering — all of SUMMARY.md) |
| Kaggle | `corinakaiser/lora-attn4/pyTorch/default` | A | `nemotronh30b-lora-attn-mamba-shared` | (≡ the HF adapter; Corina's "attn") |
| Kaggle | `corinakaiser/lora3-checkpt800-1188/pyTorch/v3` | B | `nemotronh30b-lora-mamba-shared-routed-ckpt1188` | **CoT analysis** (`expert_anal_lora.md` ← `expert_inference_log_lora.txt`) |
| Kaggle | `corinakaiser/nemotron-lora-adapter2/pyTorch/v2` | B | *duplicate of lora3 — consolidate/delete* | — |
| Kaggle | `corinakaiser/nemo-lora-4-all/pyTorch/v0` | C | `nemotronh30b-lora-all-but-router` | (Corina's "all"; not yet analyzed) |

Note: `lora3-checkpt800-1188` also bundles `checkpoint-850` (same Family B
recipe) plus optimizer/tokenizer state — that's why the download is ~3.3 GB
despite the adapter itself being ~65 MB.

## ⚠️ Confirmed two-adapter split (do not conflate)

Two analyses were run on **different adapters**:

- **Phase 2/3 routing/divergence/steering** → **Family A** (attn + Mamba +
  shared; routed experts and router **frozen**). This is what makes the
  "re-weighting, not re-routing" / indirect-steering story valid:
  `scripts/phase2_pod.sh` L48 loads `brick-factorial/...`.
- **`expert_anal_lora.md` CoT pass-rate (6/14)** → **Family B** (routed experts
  *were* trained): `expert_inference_log_lora.txt` L24 loads
  `lora3-checkpt800-1188/.../checkpoint-1188`.

How it happened: `RUNLOG.md` L222 found the HF repo empty, took a local
`checkpoint-1188`, "confirmed" it against the CoT log, and re-uploaded to HF —
but the local→HF adapter is Family A while the CoT log's adapter is Family B.
Two different `checkpoint-1188`s got crossed. The SUMMARY.md routing
conclusions stand (Family A); only the CoT pass-rate figures belong to a
different (routed-experts) model. Do not cite them as one model.

## Renaming caveat

Changing a Kaggle slug changes its `kagglehub.model_download()` URL and breaks
any script pointing at the old path. Current live pipeline points at the **HF**
path (safe); the only Kaggle paths live in `expert_inference_log_lora.txt` (a
historical log, won't be re-run). So renaming is low-risk now — do it before
more scripts depend on the old slugs, and put `r=32, α=64, base=…` in each
Kaggle model description.

## How to re-verify any adapter (ground truth)

```python
import struct, json, re, collections
p = "adapter_model.safetensors"
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]   # header length
    hdr = json.loads(f.read(n))             # header lists ALL tensors,
keys = [k for k in hdr if k != "__metadata__"]  # complete even on a partial DL
routed = [k for k in keys if re.search(r'experts\.\d+', k) and 'shared' not in k]
attn   = [k for k in keys if re.search(r'mixer\.[qkvo]_proj', k)]
print(len(keys), "tensors | routed-expert:", len(routed), "| attn:", len(attn))
```
The safetensors **header** sits at the start of the file, so this works on a
partially-downloaded file too — no need to pull multi-GB weights to confirm
which modules were trained.
