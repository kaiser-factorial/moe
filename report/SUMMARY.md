# NemoH — Consolidated Findings

Expert routing in **Nemotron-3-Nano-30B-A3B**, a hybrid Mamba-MoE model
(52 layers: 23 Mamba-2, 23 MoE with 128 routed experts + 1 shared, 6 attention;
top-6 sigmoid router with DeepSeek-V3-style load-balancing bias).

This is the single canonical summary. It supersedes scattered claims in the
phase reports where they conflict — in particular it carries the **two
verification-driven corrections** (Q1c metric-dependence; cross-layer NMI
deflation) as the authoritative reading. Full detail lives in:
`phase1_report.md` (Phase 1), §4.4 + `lora_target_modules.md` (Phase 2),
`phase3_pilot_gate.md` (Phase 3), `phase1_extended.md` (offline analyses A–I),
`errors_postmortem.md` (the inference-stack repair), and
`upstream_bug_investigation.md` / `upstream_bug_runtime_plan.md`.

---

## The one-paragraph story

Routing in this model is **finely and stably structured**. Experts specialize
by problem type — softly in the bulk (~2× uniform) and sharply in a thin tier
of near-pure specialists — and that specialization is **semantic** (it splits
by sub-task, not surface features), **organized into within-layer expert
teams**, **set per-problem and held constant across the generation**, and
**concentrated at mid-network, with layer 17 the nexus by four independent
measures**. A LoRA adapter fine-tuned on symbolic reasoning **re-weights this
structure without redrawing it** — because it never touches the router or
routed experts, every routing change is indirect, and the base model's map is
preserved while traffic is nudged onto a slightly more concentrated subset.

---

## Headline findings

**1. Two-tier specialization (Phase 1 §4.1; Extended A, I).**
A large bulk of generalist experts with real-but-soft category preferences
(corrected median selectivity 0.32 vs 0.167 uniform; bootstrap 95% CI
[0.333, 0.353]) plus **26 near-pure specialists** (selectivity > 0.8;
permutation p < 0.002 vs a null of ~4). At layer 17 the specialists are E63
(creative, 0.98), E103 (social/ethical, 0.97), E33 (symbolic, 0.82); the
highest-*mass* experts are moderate-selectivity generalists, not specialists.

**2. Specialization is semantic, not surface (Extended A, A2).**
Routing distinguishes *subtypes within* a category: within-subtype routing is
3.5× more similar than between-subtype for symbolic problems (permutation
p < 0.0005); all six symbolic subtypes form clean clusters. This retires the
pre-registered "maybe it's just digits/verse" caveat.

**3. Difficulty is invisible; routing is temporally stationary (Phase 1 §4.2;
Extended B, G).**
Within a category, routing entropy/concentration are flat across difficulty and
flat across the generation — including across the semantic `</think>` boundary
(Wilcoxon p = 0.36). The model sets a routing posture by problem *type* and
holds it; there is no late-generation "moment of commitment."

**4. Expert teams, and a (weak) cross-layer thread (Extended C, E).**
Experts co-fire in structured teams (modularity Q ≈ 0.20–0.26). By enrichment,
**social/ethical has the most distinctly organized team** (3.25× at L17, also
strong at L24/L38) — striking since it was the unscorable category; factual
gets small sharply-enriched teams late; symbolic spreads across many subtype
clusters. *Correction:* raw cross-layer NMI (0.71) is mostly small-sample bias
(null floor 0.60); true problem-specific coupling is only ~0.12 excess — real
for all pairs but **weak**, not the "coherent pathway" first claimed.

**5. Load balancing works; layer 17 is the nexus (Extended H).**
Zero dead experts in any layer (the anti-collapse bias does its job), yet
utilization is non-uniform (~40/128 experts carry 50% of mass). Gini peaks at
**layer 17** — which is also peak selectivity, sharpest social team, and (with
the L17 atlas) home to the pure specialists. Four independent signals name L17
the differentiation hotspot.

**6. The accuracy↔routing link is real but metric-dependent (Phase 1 §4.3;
Extended G — corrected).**
Concentration tracks correctness within the gradable categories when measured
as Phase 1 did (whole-generation aggregated; factual+reasoning pooled r = 0.44,
p = 0.008), but **not** as per-token mid-network thinking-region concentration
(r = 0.04). So it is a *global* property of how a problem is routed, consistent
with the stationarity finding — not a local confidence signal, and the pooled
all-category r = 0.31 is partly category composition. Per-category n (16–19) is
small; precise within-type strength is unsettled.

**7. The symbolic LoRA re-weights, never re-routes (Phase 2 §4.4; Extended
A2, A3, F, I).**
Authoritative cause (`adapter_config.json`): the adapter trains **attention
(q/k/v/o), Mamba (in/out_proj), and the shared expert** — **not** the 128
routed experts and **not** the router. (The SPEC was wrong; it implied routed
experts.) So all routing divergence is *indirect*. Consequences, confirmed at
every resolution: subtype map preserved (ratio 0.29→0.32); category map
preserved (25/25 specialists survive; pattern r = 0.93); the social team keeps
its function (2.2–3.0× enrichment) while absorbing modest sub-saturating
symbolic leakage; divergence is front-loaded on the native domain and
drift-driven off-domain; at L17 mass shifts *off* the old symbolic hubs *onto* a
reasoning expert. Net divergence is small (JSD 0.054 vs 0.110 null) and
**concentrates** routing (Gini 0.28→0.32) without collapse.

**8. Phase 3 premise validated, and steering is feasible (Phase 3 pilot;
Extended D).**
Because routing is only steerable indirectly (frozen router), residual-stream
control vectors are the right instrument — and the pilot moved specialist
occupancy ~50× more than the LoRA did, coherent at α≈0.5. The 16 banked
steering vectors are geometrically consistent across injection sites
(within-family cosine 0.53, cross-family 0.58), so a single well-chosen site
may suffice — trimming the eventual GPU sweep.

---

## Two corrections the data forced (and why they matter)

Both surfaced only because we ran nulls/controls before trusting a headline:

- **Q1c was nearly mis-reported as an artifact, then rescued.** A per-token
  metric showed the accuracy link vanishing under category control; re-running
  with Phase 1's *own* aggregated metric showed it surviving (r = 0.44,
  p = 0.008). Lesson: the link is metric-dependent and lives in global routing.
- **The cross-layer "pathway" was deflated.** Raw NMI 0.71 → bias floor 0.60 →
  real excess 0.12. Lesson: NMI over many labels with n = 111 is badly biased;
  always null it.

These are the project's epistemic backbone: every quantitative headline here
has survived a permutation/bootstrap/stratification check, and the two that
didn't were corrected in place.

---

## Limitations

- **One model, one scale**; the hybrid Mamba backbone may make routing atypical.
- **n = 111 problems** (~18–24/category); per-expert and per-subtype estimates
  rest on few items, so we lean on layer- and distribution-level statistics.
- **Gradable accuracy is thin** (computational/symbolic ≈ ceiling;
  creative/social unscored), so the within-type accuracy question is
  underpowered and unsettled.
- **All results run through our runtime patch** of the model's broken
  cached-generation path (`errors_postmortem.md`); validated behaviorally, but
  an upstream-fixed reference would be the cleaner baseline.
- **Community boundaries are unstable** where routing is diffuse (late layers);
  we trust frozen-set enrichment over membership Jaccard.
- **Steering results are a pilot** (single-prompt-class coherence checks); the
  Stage-1 sweep and causal accuracy test are not yet run.

---

## Open questions / cheap next steps

- *No GPU:* prefill-vs-generation routing (needs padding-aware handling);
  per-subtype specialist characterization.
- *GPU, ~$2–10:* confirm the cached-gen bug at runtime on the 4B/9B-v2
  (`upstream_bug_runtime_plan.md`); a larger gradable problem set to settle the
  within-type accuracy signal; Phase-3 Stage-1 sweep + causal test.
- *Write-up:* an upstream bug report (affects every HF-`generate()` user of the
  nemotron_h family).

---

*Reproduce:* `build_dataset.py` → `capture_routing.py` (applies the patches) →
`analyze_routing.py`; offline analyses via `extended_analysis.py`,
`subtype_lora_compare.py`, `temporal_divergence.py`, `coactivation_communities.py`,
`social_team_lora.py`, `thinking_vs_answer.py`, `load_balancing.py`,
`l17_atlas.py`, `nmi_null.py`. Seeds/configs in each `run_config.json`.
