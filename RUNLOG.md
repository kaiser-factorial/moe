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

### Base model run — batched (false starts continue)

- ERROR 13: OOM at ~73.9GB during first batch-8 factual batch (~10GB growth
  over decode). Initially blamed cache growth; actually a symptom of ERROR 15.
- ERROR 14: `stack expects each tensor to be equal size... [428,6] vs [432,6]`
  in finalize() — decode gate calls had GROWING row counts (4×107, 4×108...).
  Diagnosis: generation was re-processing the full sequence every step.
- ERROR 15 (root cause of 11/13/14 — NVIDIA modeling code bug chain):
  1. `prepare_inputs_for_generation` returns the cache as "past_key_values"
     but `forward()` only accepts `cache_params` → cache dropped into
     **kwargs → silently CACHELESS generation (the L1454 warning).
  2. Bridging that exposed: `HybridMambaAttentionDynamicCache` lacks
     `conv_kernel_size` (mixers read it; class never sets it).
  3. `update_conv_state`/`update_ssm_state` call `.device` on the python
     LIST `self.conv_states` → AttributeError when cache actually used.
  4. `torch_forward` L563 also reads `.device` on the raw list.
  5. generate() round-trips the cache as model_kwargs["cache_params"]
     (ALL_CACHE_NAMES match) but prepare_inputs only reads its
     past_key_values arg → fresh empty cache EVERY decode step →
     `expanded size (4096) vs (6144)` (their cache also allocates conv
     states with the wrong channel count — intermediate_size instead of
     conv_dim — masked only because init-replace happens on first use).
  6. Fused decode kernel rejects their layout: `causal_conv1d_update:
     weight must have shape (dim, width)` → forced pure-torch mamba path.
  All patched at runtime in `apply_nemotron_patches()` (capture_routing.py);
  no model files modified.
- ERROR 16: generate() validates kwargs by INSPECTING the signature of
  prepare_inputs_for_generation — a `(*a, **k)` wrapper broke validation
  ("model_kwargs not used: input_ids") → fixed with functools.wraps.
- VERIFIED after patches: batch-8 factual ran at **15.9 tok/s aggregate**
  (vs 2 tok/s), no crash, npz written. 8× speedup.
- ERROR 17 (open): first batch outputs were DEGENERATE (repetition loops,
  lost-the-plot rambling; all 8 factual FAIL). Controlled tests:
  single/uniform-batch/padded-batch short greedy generations are all clean
  ("Paris" × 12) → cache + padding machinery fine at short lengths.
  Long-generation test (1024 tok, thinking, sampled, single vs batch 8) was
  RUNNING when the pod went unreachable (SSH timeout). Hypotheses: bf16 SSM
  state drift over long decode in the torch path vs. legitimate base-model
  behavior under these sampling settings. TO RESUME: check
  /workspace/debug_pad2.log on the pod (tests D & E).
- INFRA: pod connection lost mid-test (~03:45 pod time). /workspace is a
  network volume — model cache, scripts, and logs survive restarts; only
  the SSH port should change.

## Session 2 — 2026-06-11 (new pod 31.24.80.40, same volume)

- Fresh container: pip env rebuilt (transformers 4.57.3, peft, accelerate;
  einops was an undeclared mamba-ssm runtime dep when installed --no-deps).
- ERROR 17 RESOLVED — full diagnosis chain:
  - debug_pad2 (recovered from dead pod): degenerate even SINGLE-row →
    not a padding/batching issue.
  - Re-enabled the fused mamba path with a layout shim for
    `causal_conv1d_update` (modeling passes x as (B,1,dim); kernel wants
    (B,dim,1)) — ERROR 18: still degenerate.
  - state_probe.py (Beatrix/Marco prompt: fact mid-prompt, question at
    end): model knew the question but guessed "Polly" → can only see the
    prompt TAIL. Cache dump: **all 6 attention layers had EMPTY KV
    (1, 0)** after prefill.
  - ERROR 19 (the actual root cause of all degeneracy):
    `NemotronHBlock.forward` calls attention mixers WITHOUT the cache:
    `self.mixer(hidden_states, cache_position=...)` — no `past_key_value`.
    Attention layers stored nothing at prefill and attended to a single
    token during decode. Patched at runtime (block forward override for
    attention blocks). Post-patch: KV (1,2,120,128) × 6 layers, model
    quotes mid-prompt facts verbatim, answers "Marco" correctly.
- NOTE for the report: with cacheless generation (the accidental original
  mode) every step re-ran full prefill, so attention layers DID see the
  whole sequence — that's why the slow run was coherent. The bug only
  manifests with caching on. NVIDIA's shipped code appears never tested
  with cached generation in transformers.
- Full Phase 1 run relaunched (batch 8, all patches).

### Phase 1 run — COMPLETE ✅

- 111/111 problems captured (~3.5h wall, 5-16 tok/s depending on category).
- ERROR 20 (final comedy entry): the auto-analysis watcher hung forever —
  its own `bash -c` command line contained "capture_routing.py --model",
  matching its own pgrep poll. The self-match curse claimed its author.
  Killed; analysis run manually.
- Accuracy (scored, non-truncated, n=95): 82.3%. computational 24/24,
  symbolic 15/15, factual 88.9%, reasoning 56.3%. social_ethical scoring
  void (no boxed-answer suffix for free_text_with_position outputs —
  extraction mismatch, treated as unscored). 16 truncations.
- VERIFICATION pass (task 7): selectivity baseline corrected for unequal
  category mass (symbolic 28% vs creative 5% of tokens) — naive 1/6
  baseline replaced by per-category normalized masses. Median corrected
  selectivity 0.32 (~2x uniform), inverted-U over depth (0.23 L1 → 0.37
  L17-20 → 0.27 L51), 26 expert-layer pairs >0.8 selectivity (e.g. L10-E8
  100% factual, L17-E63 99% creative). Difficulty effects on entropy: flat
  (unique/token decline = length artifact, as pre-registered).
  Concentration-correctness r=0.31 (p=0.005), positive in all difficulty
  strata (hard: r=0.46, p=0.03) → not a difficulty artifact.
- Results, discussion, conclusion written into report/phase1_report.md.

## Session 3 — 2026-06-11/12 (Phase 2: LoRA capture + divergence)

Run executed via direct RunPod REST API + autonomous on-pod script
(`scripts/phase2_pod.sh`), after an opbdh detour. Pods: 5suhqtpq7rkhn7,
6nexhwafi2nhss, 135dve03dzwm5o (attempts), ynkqgx8vm6qhfh (success, H100
HBM3 $3.29/hr EU-FR-1, ~4h). Results pulled to `outputs/logs/lora/`
(commit ea64082).

- ERROR 21 — **SPEC's adapter repo was empty**: HF
  `brick-factorial/nemotron-lora-symbolic-reasoning` contained only
  .gitattributes; the volume cache snapshot likewise. Real adapter located
  at `~/Downloads/nemotron-sub/lora-adapter3/checkpoint-1188` (confirmed
  against expert_inference_log_lora.txt) and uploaded to the SPEC path
  (README/base_model metadata fixed to pass HF YAML validation).
- ERROR 22 — **runpod/pytorch:2.8.0 image now ships a DEV torch**
  (2.8.0.dev20250319+cu128). Prebuilt mamba-ssm/causal-conv1d wheels
  (fetched by their setup.py even under --no-binary!) target release-torch
  ABI → `undefined symbol ...SetDeviceEab`. Session 2's pin recipe is
  insufficient on this image. Fix: `pip install torch==2.8.0 --index-url
  https://download.pytorch.org/whl/cu128` (WITH deps — see 23) then the
  usual --no-deps --no-build-isolation kernel installs.
- ERROR 23 — torch==2.8.0 with `--no-deps` → `ncclCommWindowRegister`
  missing (dev image's NCCL ≠ release torch's pin). Install torch with its
  own deps.
- ERROR 24 — release torch + image's dev-built torchvision →
  `operator torchvision::nms does not exist` at transformers import.
  `pip uninstall torchvision torchaudio` (text model; not needed).
- ERROR 25 — `capture_routing.py:patch_ptxas()` crashes on release-torch
  triton (`get_ptxas()` signature changed; patched fn demanded `arch`).
  H100/Hopper never needs the Blackwell redirect → run a pod-local copy
  with `patch_ptxas` neutered (first-line `return`). Keep the patch for
  Blackwell pods.
- ERROR 26 — **THE POD ASSASSIN**: pods carrying the account
  RUNPOD_API_KEY in their create-env were terminated by RunPod's secret
  scanner 3-11 min after start, audit-logged as the account owner (looked
  exactly like someone deleting pods in the console; Corina was innocent).
  Evidence: only key-carrying pods died; key lastUsed predates deletions.
  Fix: account key goes in /root/.podenv (container-disk file) written
  over ssh AFTER boot. Note the auto-injected pod-scoped key cannot
  stop/delete its own pod (REST 403, GraphQL Unauthorized).
- ERROR 27 — opbdh (RunPod launcher, github.com/lumpenspace/opbdh):
  hard-coded `minVCPUPerGPU:8, minRAMPerGPU:64` made every create fail
  "no pods with required specifications" (patched to 4/24 in
  ~/.opbdh-venv on the Mac); global ~/.config/opbdh/config.json
  (auto_network_volume + US-MD-1) silently pinned runs to the wrong DC;
  its monitor can't run in the Claude sandbox (bwrap --die-with-parent
  kills background processes per call) — run it on the Mac, or skip
  opbdh and drive the REST API directly (what Session 3 did). opbdh's
  HF cache root is /workspace/opbdh-cache, NOT our /workspace/hf.
- Gotchas reconfirmed: pgrep/pkill self-match (use [b]rackets — claimed
  two more victims this session); stop→start WIPES container disk (pip
  env, /root/*) and changes the SSH port; EU-FR-1 stocks H100 HBM3/H200
  but rarely A100; sandbox clock can lag wall time by hours (trust
  RunPod audit timestamps); volume-attached pods are DC-pinned.
- Recovered `routing_symb-medi-17.npz` (0 bytes locally after Session 2's
  sync) from the volume original — base set is 111/111 intact again.
- Phase 2 results: see report §4.4, DYFA Q2a-c, and
  `outputs/analysis/divergence/`. Watchdog scheduled task verified
  completion; pod self-deleted; balance after run $5.35.
