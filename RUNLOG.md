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

### Base model run
- Launched: 111 problems, top-6 routing per token at 23 MoE layers,
  temp 0.6 / top_p 0.95 / seed 42 per problem, category-based token budgets.
- Log: /workspace/run_base.log; outputs: /workspace/moe/outputs/logs/base/.
