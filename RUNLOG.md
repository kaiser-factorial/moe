# Run Log — MoE Expert Routing Research

Chronological record of environment details, errors encountered, fixes applied,
and results. Maintained by Claude during autonomous research sessions.

---

## Session 1 — 2026-06-10

### Setup & local work (sandbox)

**Dataset build** (`scripts/build_dataset.py`, seed 42)
- 111 problems, 6 categories. MMLU/GSM8K/MATH-500 sampled live via the HF
  datasets-server API. Symbolic problems generated fresh in the Wonderland
  style (numeral system, unit conversion, cipher, bit manipulation, symbol
  transform, base conversion) with programmatically computed answers — same
  distribution as the LoRA training domain but unseen items.
- No errors.

**Analysis pipeline smoke test** (`scripts/analyze_routing.py`)
- Tested on synthetic routing logs with planted structure (each category
  biased to a 12–20 expert block; difficulty controls spread; accuracy higher
  for easy problems).
- ERROR 1: `ModuleNotFoundError: scipy` → fixed: `pip install scipy
  --break-system-packages`.
- ERROR 2: matplotlib deprecation, `boxplot(labels=)` renamed →
  `tick_labels=`. Fixed in script.
- Pipeline correctly recovered planted structure: block-diagonal heatmap,
  difficulty → higher entropy, and negative entropy↔accuracy point-biserial
  r ≈ −0.30 (as planted). Pipeline verified end-to-end.

**Git**
- ERROR 3: stale `.git/index.lock` could not be removed ("Operation not
  permitted") — Cowork mounts block deletion by default. Fixed via the
  file-delete permission tool, then committed normally.
- Remote: git@github.com:kaiser-factorial/moe.git (waiting on deploy key
  with write access to push).

### RunPod setup

- Pod: **NVIDIA H100 80GB HBM3** (NOTE: user ordered H200; pod reports H100
  80GB. Proceeding — prior runs peaked ~70.2GB on this workload, fits.)
- Disk: `/workspace` = 163GB network volume (model cache placed here so it
  survives pod swaps), overlay root 100GB.
- Python 3.12.3, torch 2.8.0+cu128, triton 3.4.0 preinstalled.
- Installed: transformers 5.11.0, peft 0.19.1, accelerate, hf_transfer.
- ERROR 4: `pip install` rejected by PEP 668 (externally-managed env) →
  fixed with `--break-system-packages`.
- ERROR 5: `hf` CLI not on PATH in the same shell that installed it (ordering
  bug in my one-liner — install ran after the env check failed). Re-ran after
  install; fine.
- HF cache: `HF_HOME=/workspace/hf`, `HF_HUB_ENABLE_HF_TRANSFER=1`.
- Base model (~60GB BF16, 35 files) + LoRA adapter downloading in background.
  LoRA done in <1s; base reached 44GB within ~1 min.

### Model environment errors (RunPod)

- Base model download: 35 files, ~59GB, ~2 min with hf_transfer. No errors.
- ERROR 6: probe crashed — `modeling_nemotron_h.py` hard-raises
  `ImportError: mamba-ssm is required` (unlike some hybrid models there is NO
  slow-path fallback). → install `causal-conv1d` + `mamba-ssm`.
- ERROR 7 (the big one): `pip install mamba-ssm` silently **upgraded torch
  2.8.0+cu128 → 2.12.0+cu130**. Result: `torch.cuda.is_available() == False`
  (cu130 stack incompatible with pod driver) and
  `causal_conv1d_cuda...undefined symbol: _ZN3c104cuda29...` (C++ ABI
  mismatch — extension built against the old torch).
  → Fix: reinstall `torch==2.8.0+cu128` from the PyTorch cu128 index, then
  force-rebuild `causal-conv1d` and `mamba-ssm` from source with
  `--no-deps --no-build-isolation` so they link against the correct torch.
  Lesson for reproducibility: always install mamba-ssm with `--no-deps`.

### Probe results (router located)

- MoE mixer class: `NemotronHMOE` with children `.experts` (ModuleList of
  128), `.gate` (**NemotronHTopkRouter**), `.shared_experts` (NemotronHMLP).
- Router mechanics (from `modeling_nemotron_h.py` L874–918): DeepSeek-V3
  style — `sigmoid` scores (not softmax) + `e_score_correction_bias`,
  group-limited top-k (`n_group=1`, `topk_group=1` → group logic is a no-op
  here), `num_experts_per_tok=6`, `norm_topk_prob=true` then
  `routed_scaling_factor=2.5`. Forward returns `(topk_indices, topk_weights)`,
  each `(n_tokens, 6)`.
- CHANGE: `capture_routing.py` hook rewritten to consume the
  `(indices, weights)` tuple directly instead of assuming raw 128-dim logits
  (cleaner and cheaper than the planned softmax-topk).
- ERROR 8 (comedy): `ssh ... 'pkill -f probe_model; ...'` exits 255 — the
  remote shell's own command line contains the pattern, so pkill killed its
  parent shell. Twice. Fix: self-escaping pattern `pkill -f "probe_mode[l]"`.

### Base model run — false starts

- ERROR 9: transformers 5.11.0 incompatible with the model's custom code:
  `TypeError: 'NoneType' object is not subscriptable` at `cache_position[-1]`
  in `prepare_inputs_for_generation` (transformers 5.x changed cache-position
  plumbing). Model card says "tested on 4.57.3" → pinned transformers==4.57.3.
- ERROR 10: with 4.57.3, `AttributeError: module 'mamba_ssm' has no attribute
  '__version__'` — caused by OUR `shadow_mamba()` Blackwell workaround (stub
  module lacks `__version__`, and 4.57.3's `is_mamba_2_ssm_available()` reads
  it). Fix: shadow only on SM>=10 GPUs + stub now sets `__version__`.
- ERROR 11 (performance, the big rewrite): sequential generation measured at
  **~2 tok/s** (768 tok in 386s, GPU util ~46%) → est. 12-20h for 111
  problems. Bottleneck: eager per-token decode through 23 MoE layers with a
  Python expert loop, plus our per-token-per-layer `.cpu()` hook syncs.
  Also the first factual answer hit the 768-token cap mid-CoT (thinking model
  needs bigger budgets) and FAILed.
  → Rewrote `capture_routing.py` for **batched generation** (batch 8, grouped
  by category), GPU-buffered hooks (single CPU sync per batch at finalize),
  per-row trimming of post-EOS decode steps, budgets raised (factual 1536,
  comp 2048, reasoning/symbolic 3072, creative 1024, social 1536).
  Router flattens batch to (B*T,6) at prefill and (B,6) per decode step;
  per-problem arrays reconstructed from the call stream. Prefill rows include
  left-pads (analysis excludes prefill by default, so harmless).
- ERROR 12 (recurring comedy): `ssh 'pkill -f capture_routin[g]; ... python
  scripts/capture_routing.py ...'` — bracket trick defeated because the SAME
  command line legitimately contains the target string, so pkill again killed
  its own shell (exit 255). Fix: separate ssh invocations for kill and launch.

### Base model run — batched
- Launched: 111 problems, batch 8 by category, temp 0.6 / top_p 0.95 /
  seed 42 per batch. Log: /workspace/run_base.log.
