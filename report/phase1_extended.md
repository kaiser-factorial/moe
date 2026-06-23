# Extended Offline Analyses (Phase 1.5)

**Status:** complete (2026-06-23). CPU-only; no new inference. Uses the existing
base-model routing logs (`outputs/logs/base/`, 111 problems, top-6 experts ×
23 MoE layers × every token) and the 16 banked Phase-3 steering vectors. All
reproducible via `scripts/extended_analysis.py`; numbers in
`outputs/analysis/extended/*.json`.

These four analyses mine dimensions of the data we captured but never used:
the per-*token* and cross-layer structure (Phase 1 only aggregated per-problem),
the sub-category semantics, and the geometry of the steering vectors.

---

## A. Subtype specialization — routing is *semantic*, not surface

Phase 1 pre-registered a caveat: category-level specialization might track
surface features (digits, verse line breaks) rather than meaning. We can now
test this directly by asking whether routing distinguishes *subtypes within a
category*. For each problem we build its mid-network routing distribution
(layers 13–24, the selectivity peak) and measure pairwise Jensen–Shannon
divergence; if subtypes are real, within-subtype pairs should be much more
similar than between-subtype pairs. Significance via label-permutation
(2000 shuffles).

| Category | within-subtype JSD | between-subtype JSD | gap | perm p |
|----------|-------------------:|--------------------:|----:|-------:|
| symbolic | 0.035 | 0.124 | 0.088 | <0.0005 |
| reasoning | 0.128 | 0.164 | 0.036 | <0.0005 |
| factual | 0.143 | 0.234 | 0.091 | <0.0005 |

The symbolic heatmap is unambiguous: each of the six subtypes
(base conversion, bit manipulation, numeral system, symbol transform, text
cipher, unit conversion) forms a tight low-divergence block on the diagonal.

![symbolic subtype JSD](../outputs/analysis/extended/figures/A_symbolic_subtype_jsd.png)

**Finding:** routing carries fine-grained, *task-semantic* structure below the
category level — within-subtype routing is 3.5× more similar than
between-subtype for symbolic problems. This retires the Phase-1 surface-feature
caveat: the router is distinguishing "Roman numerals" from "Caesar cipher,"
not merely "has digits." (Reasoning shows the weakest separation, consistent
with its subtypes — algebra, geometry, number theory — sharing more machinery.)

### A2. Does the symbolic LoRA preserve subtype structure?

Re-running the subtype test on the LoRA logs (`outputs/logs/lora/`) and
comparing to base directly answers a question Phase 2 raised: the adapter
*re-weights without re-routing* (§4.4) — but does it preserve the *fine-grained*
subtype map, or only the coarse category split?

| Category | base within/between (ratio) | LoRA within/between (ratio) |
|----------|----------------------------:|----------------------------:|
| symbolic | 0.035 / 0.124 (0.29) | 0.042 / 0.134 (0.32) |
| reasoning | 0.128 / 0.164 (0.78) | 0.142 / 0.174 (0.82) |
| factual | 0.143 / 0.234 (0.61) | 0.138 / 0.233 (0.59) |

All six within/between gaps remain highly significant (permutation p ≤ 0.001).
The side-by-side symbolic matrices show the same six diagonal blocks.

![base vs lora symbolic](../outputs/analysis/extended/figures/A2_base_vs_lora_symbolic.png)

**Finding:** the symbolic-reasoning LoRA **preserves the subtype routing map
essentially intact** — the separation ratio is unchanged (symbolic 0.29→0.32),
not collapsed. Fine-tuning on the symbolic domain did not merge its subtypes
into one undifferentiated "symbolic" blob, nor scramble them. This strengthens
the Phase-2 conclusion: the adapter nudges *how much* mass goes to the existing
sub-structure (within-subtype JSD ticks up slightly, +0.007 symbolic) without
redrawing the map. Specialization is a property of the base model's router that
LoRA rides on rather than rewrites — exactly the premise Phase 3's
control-vector approach depends on.

## B. Temporal dynamics — routing is stationary across a generation

Using per-token routing (mid-network), we binned each generation into deciles
and tracked top-1 concentration and entropy by position, per category.

![temporal](../outputs/analysis/extended/figures/B_temporal.png)

**Finding (a clean null):** routing statistics are essentially flat from the
first decile to the last — concentration ≈0.07 and entropy ≈3.4 nats
throughout, for every category. The model does **not** route diffusely while
"thinking" and then sharpen at the answer; its routing posture is set by
problem *type* and held constant over the whole trajectory. This complements
Phase 1's Q1b result (difficulty is invisible to routing) and Q1c
(concentration predicts correctness): the predictive concentration signal is a
stable property of the whole generation, not a moment of commitment near the
answer. (Caveat: thinking/answer boundaries were approximated by position, not
parsed from `</think>`; a token-aligned split is a cheap refinement.)

## C. Expert co-activation and cross-layer pathways

**Within a layer**, experts that fire on the same token form structured
"teams": the layer-17 co-activation matrix (spectral-ordered) shows clear
blocks rather than a uniform field — sets of experts that habitually co-fire.

![co-activation L17](../outputs/analysis/extended/figures/C_coactivation_L17.png)

**Across layers**, a problem's top-1 expert at one MoE layer strongly predicts
its top-1 at the next: normalized mutual information averages **0.71** and is
remarkably stable (0.66–0.79) across all 22 adjacent layer pairs.

![cross-layer NMI](../outputs/analysis/extended/figures/C_crosslayer_nmi.png)

**Finding:** routing is not 23 independent decisions — it is a coherent
*pathway*. A problem entering a specialist track tends to stay on it layer
after layer. This is the mechanistic substrate of the Phase-1 specialization:
specialists aren't isolated per-layer modules but links in consistent
cross-layer routes. (Caveat: part of the NMI reflects a few globally popular
top-1 experts shared across problems; a label-shuffle null would isolate the
problem-specific component and is a cheap next step.)

## D. Robustness and steering-vector geometry

**The specialists are not chance.** Phase 1 reported 26 expert-layer pairs with
selectivity > 0.8. A permutation null (shuffle category labels, recount;
500 draws) yields **4.3 ± expected, max 8** such pairs by chance — the observed
26 gives p < 0.002. The bulk specialization is likewise solid: bootstrap 95% CI
on the mid-network median selectivity is **[0.333, 0.353]**, well clear of the
0.167 uniform baseline.

**Steering-vector geometry (Phase-3 prep, offline).** The 16 banked
symbolic-vs-rest direction vectors are mutually positively aligned, with mean
cosine 0.53 within a site family (post-Mamba or post-MoE) and 0.58 across
families.

![vector cosine](../outputs/analysis/extended/figures/D_vector_cosine.png)

**Finding:** the "symbolic" direction is a broadly *consistent* direction in
residual space regardless of where it is read off — post-Mamba and post-MoE
sites point essentially the same way (cross-family alignment is as high as
within-family). For Phase 3 this means the steering signal is robust to
injection-site choice, and a single well-chosen site may suffice — useful for
trimming the eventual GPU sweep.

---

## What this adds to the headline story

Phase 1 established *that* experts specialize by problem type. The offline
mining sharpens it on four fronts, for free:

1. The specialization is **semantic and fine-grained** (subtype-level), not a
   surface-feature artifact (A).
2. It is **temporally stationary** — a fixed posture per problem type, not a
   late-generation event (B).
3. It is **organized into cross-layer pathways**, not independent per-layer
   picks (C).
4. The core claims are **statistically robust** (specialists p<0.002; bulk
   selectivity CI clear of baseline), and the Phase-3 steering direction is
   **geometrically consistent across sites** (D).

## Cheap follow-ups (still no GPU)

- Token-aligned thinking/answer split (parse `</think>`) to confirm B.
- Label-shuffle null for the cross-layer NMI in C.
- Repeat A on the **LoRA** logs (`outputs/logs/lora/`) — did fine-tuning
  preserve or blur subtype structure? Directly extends the Phase-2 story.
- Co-activation community detection (modularity) to name the expert "teams."
