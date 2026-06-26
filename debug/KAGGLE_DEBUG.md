# Kaggle Debug Log

Issues and fixes for running NemotronH-30B-A3B work on Kaggle (notebooks,
2×T4). Newest first.

---

## 2026-06-23 — RTX Pro 6000 (Blackwell) offline-deps recipe

For the better Kaggle GPU (RTX Pro 6000, Blackwell **sm_120**, ~96 GB, native
bf16 — fits 30B in BF16, no quantization). It's **internet-off**, so deps use the
build-once-online → save-output → attach-offline pattern.

**Build step** = `ryanholbrook/nvidia-utility-script` (public). It builds, with
`TORCH_CUDA_ARCH_LIST=12.0`, `FLASH_ATTENTION_FORCE_BUILD=TRUE`, `MAX_JOBS=2`,
into `/kaggle/working`:
- torch/vision/audio **nightly cu128** (`download.pytorch.org/whl/nightly/cu128`)
- `nvidia-cutlass`
- `causal-conv1d>=1.4.0` + `flash-attn` (`--no-build-isolation`)
- mamba from source (`git+https://github.com/state-spaces/mamba.git`)

**Consumption (corrected from Corina's `nemotron-train-4-0-all` cell 1):** you do
NOT rebuild offline or save/attach a separate output. Add the utility script
**as a Utility-Script input** → Kaggle mounts its prebuilt packages at
`/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script` (incl.
`triton/backends/nvidia/bin/ptxas-blackwell`); just `sys.path.insert` it. Works
offline.

**transformers v5 gotcha:** the util ships a v5 nightly. `apply_chat_template(
tokenize=True)` now returns `tokenizers.Encoding` objects (not a flat int list)
→ `torch.tensor()` fails "Could not infer dtype of tokenizers.Encoding" in the
collator. Fix: render `tokenize=False` then re-tokenize the text with
`add_special_tokens=False` (as her `format_and_mask` did). Also `warmup_ratio`
warns (removed in v5.2) but still works — ignore for now.

**OOM gotcha — `PYTORCH_CUDA_ALLOC_CONF` ordering:** it must be set *before* the
first `torch.cuda.*` call (which inits CUDA and locks the allocator). v1 of the
notebook set it in cell 2 but cell 1 already called `torch.cuda.get_device_name`
→ `expandable_segments:True` never applied → fragmentation ("N GiB reserved but
unallocated") → OOM in backward at MAX_LEN=4096 + eager attention. Fix: set the
env at the very top of cell 1 before `import torch`, **restart the kernel** (CUDA
must re-init), and MAX_LEN=2048 for the probe. Her original set these env vars at
the top of cell 1 for exactly this reason.

**Required Blackwell patches (all three, from her cell 1):**
1. **Shadow `mamba_ssm.modules.mamba3`** — its eager `Mamba3` import pulls cutlass
   built for sm_90, crashes on sm_120. Stub mamba3; keep the real Triton ops.
2. **Patch triton `get_ptxas`** for arch ≥ 100 → the mounted `ptxas-blackwell`.
3. **Force slow path:** set `is_fast_path_available = False` on the
   `modeling_nemotron_h` module *after* model load — the fast path's
   `causal_conv1d` is sm_90 and crashes on Blackwell. (Essential; easy to miss.)

**Recipe note (verified):** the original trainer built assistant-only `labels`
in `format_and_mask` (prompt → -100), BUT then used
`DataCollatorForLanguageModeling(mlm=False)`, whose `torch_call` does
`labels = input_ids.clone(); labels[labels==pad_id] = -100` — it **overwrites**
the provided `labels`. Confirmed by reading the collator source. So the
prompt-masking was discarded and A/B/C **effectively trained on the full sequence**
(only pad/eos masked; pad==eos so in-sequence eos also masked). => Family D matches
via `MASK_PROMPT=False` (the default). Record this in report §4 ("what the
training actually did", alongside the LoRA-target correction). Notebook:
`train_router_kaggle_rtxpro.ipynb` (v2, BF16, single-card, `probe`/`full`,
+ padding_side=right, device_map=auto, pretraining_tp=1, Triton warm-up); T4
4-bit notebook stays as fallback.

---

## 2026-06-23 — Router-training probe notebook (`scripts/train_router_kaggle_probe.ipynb`)

Three-pass review of the first draft before any GPU run. Five real bugs; the
first would have silently defeated the probe's purpose.

### Pass 1 — training-logic correctness

**BUG (critical): load-balance aux loss carried no gradient.**
The aux term `α·N·Σ f_e·P_e` was computed *inside* a forward hook. But
`prepare_model_for_kbit_training` turns on **gradient checkpointing**, so the
gate forward runs inside a checkpointed block whose first pass is under
`torch.no_grad()`. The captured `scores` were therefore detached → `α·aux`
contributed **zero gradient to `gate.weight`**. The balance penalty would have
been logged but inert, making every LR look more collapse-prone than it really
is and pushing the probe toward an unnecessarily low LR.
**Fix:** the hook now stores only the *detached gate input* (hidden states);
the aux loss is recomputed **after** the forward, where `gate.weight` is a live
leaf. Gradient reaches the weight (hidden is treated as a constant input, which
is exactly what we want). Works regardless of checkpointing.

**BUG: all-masked-labels → NaN loss.**
Assistant-only masking sets prompt tokens to `-100`. If a problem's prompt
exceeds `MAX_LEN`, every label is `-100` and the LM loss is NaN for that sample.
**Fix:** guard — if a row is fully masked, unmask its last token.

### Pass 2 — Kaggle / 2×T4 environment

**BUG: missing `torch_dtype=float16` → bf16 on Turing.**
With only `bnb_4bit_compute_dtype=fp16`, the *non-quantized* parts (embeddings,
norms, residual stream) load in the checkpoint's **bf16**, which T4 (Turing,
sm_75) does not support → dtype clashes against the fp16 compute path.
**Fix:** pass `torch_dtype=torch.float16` to `from_pretrained` so the whole
non-quantized graph is fp16.

**BUG: `device_map="auto"` + `Trainer` → DataParallel fight.**
A model sharded across both T4s (model-parallel) can get wrapped in
DataParallel by `Trainer`, which conflicts with the sharding.
**Fix:** `model.is_parallelizable = True; model.model_parallel = True`.

**BUG: gradient checkpointing enabled twice.**
Once by `prepare_model_for_kbit_training(use_gradient_checkpointing=True)` and
again by `TrainingArguments(gradient_checkpointing=True)`.
**Fix:** enable once in the load cell with
`gradient_checkpointing_kwargs={"use_reentrant": False}`; set the
`TrainingArguments` flag to `False`.

### Pass 3 — edge cases / runtime

**BUG: multi-GPU aux reduction.**
The 23 gates live on two different T4s, so `torch.stack` of their aux terms
throws a device-mismatch.
**Fix:** move each term to the loss device before stacking.

**Confirmed (not a bug):** the probe needs **none** of `capture_routing.py`'s
cache/KV-cache/ptxas patches — those are generation-only. Training runs
`use_cache=False`, so the stock forward is correct.

**Hardening:** added startup asserts (exactly 23 trainable tensors, all
`requires_grad`) so a freeze/unfreeze mistake fails loudly, not after an hour.

### Residual risks (can't be caught by static review — only a real T4 run will tell)
- **mamba-ssm / causal-conv1d source build** is the most fragile step. Pin the
  versions that worked when the LoRAs were trained on Kaggle; the build is slow
  (~15-20 min) and must match the installed torch. Prefer a prebuilt-wheel
  dataset if available.
- **Memory fit**: 4-bit 30B (~16-18 GB) + fp16 activations + grad checkpointing
  across 2×T4 (32 GB) at `MAX_LEN=1536` should fit but is tight. On OOM: drop
  `MAX_LEN`→1024, then verify `model.hf_device_map` actually splits layers
  across both cards.

### Context / rationale
- 30B BF16 ≈ 60 GB > 32 GB (2×T4) → 4-bit base is mandatory; the router
  (`gate.weight`, a raw `nn.Parameter`) is *not* an `nn.Linear`, so bitsandbytes
  never quantizes it → it stays fp32 + trainable. "QLoRA, but for the router."
- The 4-bit/fp16 base is a **precision confound** vs the BF16 Families A–C, so
  this notebook is a **probe only** (pick the LR); the real Family-D run is on
  the RunPod H100 in BF16 via `scripts/train_router.py`.
