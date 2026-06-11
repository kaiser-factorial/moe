# Expert Routing Specialization in a Hybrid Mamba-MoE Model

**Phase 1: Base-Model Routing Analysis of Nemotron-3-Nano-30B-A3B**

*Status: DRAFT — Sections 4-6 pending Phase 1 run completion.*

---

## 1. Introduction

Mixture-of-Experts (MoE) layers replace a transformer's single feed-forward
network with a pool of smaller "expert" networks and a learned router that
sends each token to only a few of them. The model gets the capacity of the
full pool while paying the compute cost of the chosen few — Nemotron-3-Nano
carries 31.6B parameters but activates roughly 3B per token (hence "30B-A3B").

The router is where the interesting questions live. If routing were random,
an MoE would just be an awkward ensemble. The premise of the architecture is
that the router learns *useful structure*: experts come to serve particular
token distributions, and the router recognizes which tokens belong where.
What that structure looks like in a trained model is an empirical question:

- **Q1a** — Do experts specialize by problem *type* (factual recall vs.
  arithmetic vs. creative writing vs. moral reasoning)?
- **Q1b** — Does problem *difficulty* change routing (more experts? different
  experts? flatter or sharper routing distributions)?
- **Q1c** — Does routing behavior correlate with whether the model gets the
  answer *right* (is routing diversity a usable signal of confidence)?

Phase 2 (separate report section) compares this base-model picture against a
LoRA adapter fine-tuned exclusively on symbolic reasoning, asking whether
fine-tuning reshapes routing on its native domain and whether the reshaping
leaks into unrelated domains.

## 2. Background

### 2.1 The model

[Nemotron-3-Nano-30B-A3B](https://hf.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
is a *hybrid* architecture: of its 52 layers, only 6 are attention layers.
The rest alternate between Mamba-2 state-space layers (23) and MoE
feed-forward layers (23), in roughly 7-layer blocks
(Mamba → MoE → Mamba → MoE → Mamba → Attention → MoE):

| Type | Layer indices | Count |
|---|---|---|
| Mamba-2 SSM | 0,2,4,7,9,11,14,16,18,21,23,25,28,30,32,35,37,39,41,44,46,48,50 | 23 |
| MoE FFN | 1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51 | 23 |
| Attention | 5,12,19,26,33,42 | 6 |

Each MoE layer has **128 routed experts** plus one **shared expert** that
processes every token unconditionally. The shared expert gives the layer a
guaranteed common pathway; the routed experts are the specialization budget.

### 2.2 The router

Each MoE layer's router (`NemotronHTopkRouter`) is a DeepSeek-V3-style gate.
For each token it computes 128 scores via a linear map followed by a
**sigmoid** (not a softmax — scores are independent per expert), adds a
learned per-expert bias used only for selection (`e_score_correction_bias`,
a load-balancing device), picks the **top 6** experts, renormalizes their
scores to sum to 1, and scales by 2.5. The selected experts' outputs are
combined with these weights and added to the shared expert's output.

Two consequences matter for analysis. First, "how strongly a token wanted
expert e" is only observable for the 6 winners — the routing weights we log
are the renormalized winner scores. Second, because selection bias and score
are decoupled, an expert can be frequently *chosen* yet carry modest weight.

### 2.3 What we log

We hook every MoE layer's router and record, for every token at every layer,
the 6 winning expert indices and their weights — during both prompt
processing (prefill) and generation. Analysis defaults to generation-phase
tokens only: those reflect the model's own reasoning process rather than
prompt encoding, and in batched runs prefill rows include padding tokens.

### 2.4 Metrics

For a problem p and layer L, let m(e) be the total routing weight expert e
received over p's generated tokens, and P = m / Σm its normalization.

- **Unique experts**: |{e : m(e) > 0}| — coverage of the expert pool.
- **Entropy**: H(P) = −Σ P log P, in nats; range [0, log 128 ≈ 4.85].
  High = routing mass spread evenly; low = few experts dominate.
- **Concentration**: max(P) — share of the single most-used expert.
- **Selectivity** (per expert, per layer): across problem categories, the
  largest share of the expert's mass coming from a single category. 1.0 =
  exclusively serves one category; 1/6 ≈ 0.17 = perfectly uniform across our
  six categories.
- **Jaccard similarity** (Phase 2): |A∩B|/|A∪B| between top-expert sets.
- **Accuracy correlation**: point-biserial r between correctness and each
  routing metric, aggregated per problem across layers.

## 3. Methodology

### 3.1 Dataset

111 problems across six categories, each with graded difficulty
(`data/problems.json`, seed 42, every benchmark item fetched at build time):

| Category | n | Source | Difficulty axis |
|---|---|---|---|
| Factual | 24 | MMLU (global facts, geography, college bio/chem, professional medicine/law) | grade-school → professional |
| Computational | 24 | GSM8K (binned by solution step count) + generated arithmetic | 2-step → 5+-step |
| Reasoning | 18 | MATH-500 (binned by level) | level 1-2 → 4-5 |
| Creative | 12 | Generated templates | free-form → constrained forms |
| Social/ethical | 15 | Curated dilemmas | clear-cut → multi-stakeholder |
| Symbolic | 18 | Generated fresh, Wonderland-style | param ranges / base sizes |

The symbolic category mirrors the LoRA training domain (numeral systems,
unit conversion, ciphers, bit manipulation, operator transforms, base
conversion) but contains only newly generated, unseen items — it gives
Phase 2 its native-domain comparison set inside the same dataset.

Scoreable categories carry exact answers (computed or benchmark-provided);
creative and ambiguous social problems are unscored by design.

### 3.2 Inference protocol

Batched generation (batch 8, grouped by category so token budgets match),
temperature 0.6, top-p 0.95, seed fixed per batch, thinking mode enabled,
budgets 1024-3072 tokens by category. Answers extracted from `\boxed{}`,
scored by exact/numeric match (multiple choice: last standalone letter).
Generations that hit the token budget without emitting an answer are flagged
*truncated* and excluded from accuracy analysis (they are not model errors).
Hardware: 1× H100 80GB; model in BF16; transformers 4.57.3 pinned.

### 3.3 A necessary detour: repairing cached generation

This study's inference stack required fixing the model's shipped code. The
findings matter for anyone running Nemotron-3-Nano through Hugging Face
transformers, so we document them as a result in their own right.

**The model ships with cached generation broken end-to-end.** Five distinct
defects in `modeling_nemotron_h.py` interact so that `generate()` silently
runs without any cache:

1. `prepare_inputs_for_generation` returns the cache under the key
   `past_key_values`, but `forward()` only accepts it as `cache_params` —
   the cache is swallowed by `**kwargs` and dropped, every step.
2. Even when bridged, generate() round-trips the returned cache as
   `model_kwargs["cache_params"]`, which `prepare_inputs_for_generation`
   ignores — so a fresh, empty cache is constructed at every decode step.
3. The cache class (`HybridMambaAttentionDynamicCache`) lacks the
   `conv_kernel_size` attribute its own mixers read, and its
   `update_conv_state`/`update_ssm_state` methods call `.device` on a
   Python list — they crash on first genuine use.
4. The fused decode kernel is called with a transposed layout
   (`causal_conv1d_update` receives x as (B, 1, dim) instead of (B, dim, 1)).
5. Most consequentially: `NemotronHBlock.forward` never passes the cache to
   attention mixers at all. All six attention layers store no KV during
   prefill and attend to a *single token* during decode.

The failure mode of (5) is subtle and instructive: the model does not crash
or produce gibberish from step one. The Mamba conv states carry the last few
prompt tokens, so the model knows roughly *that* it was asked something and
can see the instruction suffix — it then confabulates the rest, producing
plausible-looking reasoning about a question it cannot see, decaying into
repetition loops. In a probe with a mid-prompt fact ("a parrot called
Marco") and an end-of-prompt question, the broken model guessed "Polly";
the patched model quotes the prompt verbatim and answers correctly.

Why was this never caught? Cacheless generation — the accidental default —
re-processes the full sequence every step, so attention layers *do* see
everything and outputs are coherent, just quadratically slow (~2 tok/s
observed, with O(t²) memory growth). Any test that tolerated the slowness
would pass. All five defects are patched at runtime in
`apply_nemotron_patches()` (scripts/capture_routing.py); no model files are
modified. Post-patch throughput: ~8-16 tok/s batched, coherent long-form
generation. The full debugging chronology, including dead ends, is in
RUNLOG.md (19 numbered errors).

### 3.3 Analysis pipeline

`scripts/analyze_routing.py` aggregates per-token logs into per-problem,
per-layer routing-mass vectors, then computes the metrics of §2.4. The
pipeline was validated end-to-end on synthetic logs with planted structure
(category-blocked experts, difficulty-controlled spread) and recovered all
planted effects before touching real data.

## 4. Results

*Pending Phase 1 run completion.*

### 4.1 Expert specialization by problem type (Q1a)
TBD

### 4.2 Difficulty effects on routing (Q1b)
TBD

### 4.3 Routing and accuracy (Q1c)
TBD

## 5. Discussion

*To be completed from results. Interpretation guide, written before
analysis (pre-registered so the data can't seduce us into a story):*

**On Q1a (specialization).** Selectivity well above the 1/6 ≈ 0.17 uniform
baseline for many experts would indicate problem-type specialization. Two
distinct positive patterns are possible: a few highly selective experts per
category (sparse specialists) vs. broadly shifted distributions (soft
division of labor). A negative result — near-uniform selectivity — would
itself be informative: DeepSeek-V3-style sigmoid routing with bias-based
load balancing actively fights expert collapse, and may homogenize experts.
Note also: token-level routing may track surface statistics (numbers,
poetry line breaks) rather than "topics"; expert-category alignment should
be sanity-checked against subtype splits before strong claims.

**On Q1b (difficulty).** "Harder problems route more experts" has a
confound: harder problems generate longer chains of thought, and unique
expert count grows with sequence length. The per-token normalization
(unique/token) and entropy are the cleaner metrics. If difficulty raises
entropy *within* category at comparable lengths, that suggests the router
responds to uncertainty; if not, difficulty may be invisible to routing.

**On Q1c (accuracy correlation).** Any correlation is correlational only:
difficulty drives both errors and (potentially) routing dispersion, so a
raw routing-accuracy link must be checked within difficulty strata before
reading it as a confidence signal.

## 6. Conclusion & Limitations

*Conclusion TBD from results. Known limitations, independent of outcome:*

- **Sample size**: 111 problems (~18-24/category) bounds the granularity of
  per-expert claims; per-expert-per-category estimates rest on few problems,
  so we emphasize layer-level and distribution-level statistics.
- **Single model, single scale**: findings describe Nemotron-3-Nano-30B-A3B,
  not MoE models in general; the hybrid Mamba backbone may make routing
  dynamics atypical relative to full-attention MoEs.
- **One sample per problem**: temperature 0.6 sampling with one generation
  per problem conflates sampling noise with routing signal; per-problem
  routing profiles average over hundreds of tokens, which mitigates but
  does not eliminate this.
- **Generated-phase only**: prefill routing is excluded by default (and in
  batched runs contains left-pad tokens). Conclusions are about routing
  *while generating*, not while reading.
- **Top-6 observability**: router scores are logged only for winners; we
  cannot see near-miss experts, so "routing distribution" means the
  renormalized winner distribution.
- **Truncations**: budget-capped generations are excluded from accuracy but
  their routing tokens are retained; ~severe truncation rates per category
  are reported alongside accuracy.
- **Patched inference stack**: all results depend on our runtime repairs to
  the model's cached-generation path (§3.3). The patches were validated
  behaviorally (verbatim mid-prompt recall, coherent long-form output), but
  an upstream-fixed reference implementation, when available, would be the
  cleaner baseline.

## Appendix

### A. Reproducibility
1. `python scripts/build_dataset.py` — rebuilds `data/problems.json`
   deterministically (seed 42; MMLU/GSM8K/MATH-500 fetched from the HF
   datasets-server; symbolic/creative/social generated).
2. `python scripts/capture_routing.py --model-path <M> --problems
   data/problems.json --out-dir outputs/logs/base --batch-size 8` — applies
   all runtime patches, runs instrumented inference, writes per-problem
   `routing_<pid>.npz` + `results.jsonl` + `run_config.json`.
3. `python scripts/analyze_routing.py --log-dir outputs/logs/base
   --problems data/problems.json --out-dir outputs/analysis/base` — all
   CSVs and figures.
- Environment pins: transformers==4.57.3, torch 2.8.0+cu128; install
  `causal-conv1d` and `mamba-ssm` with `--no-deps --no-build-isolation`
  (their resolver upgrades torch to an incompatible build otherwise), plus
  `einops`. Single H100 80GB suffices (~70GB peak).
- `scripts/probe_model.py`, `scripts/state_probe.py`,
  `scripts/debug_padding.py` — diagnostic tools used to locate the router
  gate and verify cache integrity.

### B. Error log
`RUNLOG.md` — chronological error/fix log (19 numbered errors), including
the full diagnosis narrative of the cached-generation repair.

### C. Full outputs
`outputs/analysis/` — per-problem metrics, per-expert specialization,
per-layer statistics, accuracy correlations, all figures.
