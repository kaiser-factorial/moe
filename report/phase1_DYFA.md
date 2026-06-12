# Phase 1 — DYFA Summary

Base-model expert routing in Nemotron-3-Nano-30B-A3B. One DYFA block per
research question. Full prose report: `phase1_report.md`; all statistics from
`outputs/analysis/base/`.

---

## Q1a — Do experts specialize by problem type?

**D (did):** Captured the top-6 routed experts and weights for every
generated token at all 23 MoE layers across 111 problems in six categories,
then computed each expert's *selectivity* (largest share of its routing mass
attributable to one category) after normalizing category token masses;
analysis used generation-phase tokens only, treating each problem's
aggregated routing-mass vector as the unit of observation.

**Y (why):** Categories contribute wildly unequal token mass (symbolic 28%
vs. creative 5% of all generated tokens), so raw category shares would crown
every expert "symbolic" by exposure alone — normalization makes the uniform
baseline (1/6 ≈ 0.167) honest; generation-only because batched prefill rows
contain left-pad tokens and we care about routing while the model *reasons*,
not while it reads.

**F (found):** Median corrected selectivity is **0.32 ≈ 2× the uniform
baseline of 0.167**, with an inverted-U depth profile (≈0.23 at layer 1 →
**0.37 at layers 17-20** → 0.27 at layer 51); on top of this bulk sit **26
expert-layer pairs with selectivity > 0.8** (e.g. L10-E8: 100% factual,
L17-E63: 99% creative, L17-E103: 99% social/ethical), each carrying only
~0.5-1% of its layer's mass; meanwhile routing stays broad overall — a
typical problem touches 116/128 experts per layer, mean per-problem entropy
4.19 of 4.85 max nats. Figures: `base_heatmap_specialization.png`,
`base_selectivity_by_layer.png`.

**A (answer):** Yes — but it is a two-tier economy, not a division into
departments: a generalist bulk whose category preferences are real but soft
(~2× uniform), plus a thin tier of near-pure niche specialists concentrated
mid-network where type-differentiation peaks. Limitations: "category"
selectivity may partly track surface statistics (digits, verse line breaks)
rather than semantics; with ~18-24 problems per category, per-expert
estimates rest on few problems, so we lean on layer- and distribution-level
statistics rather than individual-expert claims.

---

## Q1b — Does difficulty change routing?

**D (did):** Compared routing entropy, top-1 concentration, and
unique-experts-per-token across difficulty levels within each category, with
a pre-registered check that any "unique experts" effect be tested against
generation length before interpretation.

**Y (why):** Harder problems generate longer chains of thought and
unique-expert counts saturate near the 128 ceiling, so the per-token ratio
*mechanically* falls with length — we pre-registered this confound to avoid
discovering it post hoc and dressing it up as a finding; entropy and
concentration are length-robust, so they carry the real test.

**F (found):** Entropy is flat across difficulty in every category
(**≈4.0-4.4 nats at every level**); top-1 concentration moves little, the
only deviation being hard factual rising to 0.14 (long ruminative trivia
deliberations); the apparent decline in unique experts per token with
difficulty disappears once length is accounted for — exactly the
pre-registered artifact. Figure: `base_difficulty_effects.png`.

**A (answer):** Given flat entropy and concentration across strata, routing
is effectively blind to difficulty: the router classifies *what kind* of
input it sees, not *how hard* it is. Limitation: our difficulty axes are
benchmark-defined (e.g. GSM8K step count, MATH level), which may not match
the model's internal sense of hardness; and with ~6-8 problems per
difficulty cell the test is powered for large effects only.

---

## Q1c — Does routing correlate with correctness?

**D (did):** Computed point-biserial correlations between per-problem
correctness (n = 95 scored problems; 16 budget-truncated generations
excluded; social/ethical unscored due to an answer-extraction mismatch) and
per-problem routing metrics averaged across layers, with a pre-registered
stratification by difficulty and category to rule out "hard problems are
both dispersed and wrong" as a common cause.

**Y (why):** Correctness is binary so point-biserial is the natural
statistic; stratifying was pre-registered because difficulty could plausibly
drive both routing dispersion and errors, making a raw correlation
uninterpretable; truncations are excluded because hitting a token budget is
a budget failure, not a model error.

**F (found):** Top-1 concentration correlates with correctness at
**r = 0.31, p = 0.005 (n = 95)**; the correlation is positive in *every*
difficulty stratum (easy r = 0.28, medium r = 0.25, **hard r = 0.46,
p = 0.03**) and in both scoreable categories with variance (factual
r = 0.33, reasoning r = 0.22); entropy and unique-expert counts correlate
negatively but not significantly. Figure: `base_accuracy_vs_routing.png`.

**A (answer):** Given r = 0.31 (p = 0.005), surviving stratification, I
conclude routing concentration carries a weak but real correctness signal —
when the router commits its weight to few experts, the model is more likely
right — and since it costs no extra forward passes it is a candidate cheap
confidence proxy. Limitations: causal direction is open (confident routing
may cause correctness, or familiar problems may cause both); one sampled
generation per problem (temperature 0.6) conflates sampling noise with
routing signal; accuracy ceiling effects (computational and symbolic at
100%) leave only two categories contributing variance; and alpha is not
adjusted across the several metrics tested, though the headline result
would survive a Bonferroni factor of 3 (p = 0.005 × 3 = 0.015 < 0.05).

---

## Q2a — Did LoRA fine-tuning change routing on its native (symbolic) domain?

**D (did):** Ran the identical 111 problems through the LoRA-adapted model
(same seeds, budgets, patched stack), built per-problem per-layer
routing-mass distributions for both models, and compared same-problem
base↔LoRA distributions via Jensen-Shannon divergence and top-6-expert
Jaccard, with a within-category base↔base between-problem null as the scale
reference; treated per-problem layer-averaged JSD as the unit of analysis.

**Y (why):** JSD over raw KL because it is symmetric and bounded (the two
runs generate different tokens, so neither direction is privileged); the
null is essential because "JSD = 0.05" is uninterpretable without knowing
how far apart two *unrelated* routing profiles sit; same-problem pairing
controls for content.

**F (found):** Symbolic divergence is the largest of all six categories
(mean JSD **0.0537** nats, top-6 Jaccard 0.424; n = 18), significantly
above the other 93 problems pooled (**Mann-Whitney U = 1404, one-sided
p < 10⁻⁵, rank-biserial r = 0.68**) and above the nearest category
(computational: U = 297, p = 0.020) — yet only **half** the
between-problem null (0.1096). Peak at layer 8 (0.070). Figure:
`divergence_by_layer.png`.

**A (answer):** Given p < 10⁻⁵ against other categories but a shift half
the size of the within-category null, I conclude LoRA measurably and
preferentially re-weighted routing on its training domain without
relocating it — a re-ranking within the base model's routing geography.
Limitations: the two models generate different token sequences, so
distributions aggregate over non-identical generations (content drift and
routing drift are partially confounded); n = 18 symbolic problems; JSD on
renormalized top-6 winner mass, not full router scores.

---

## Q2b — Does the routing shift leak into non-symbolic domains?

**D (did):** Computed the same divergence metrics per category and compared
each category's base↔LoRA divergence to its own between-problem null.

**Y (why):** Per-category nulls because categories differ in baseline
routing diversity (factual problems are far more routing-distinct from each
other, null 0.214, than symbolic ones, 0.110) — a shared threshold would
misread leakage.

**F (found):** Divergence is graded: factual **0.0239** < social/ethical
0.0268 < reasoning 0.0395 < creative 0.0434 < computational **0.0450** <
symbolic 0.0537; every category sits well under its null (factual: 9×
under). The ordering tracks surface adjacency to the training domain
(digit-heavy computational moves most among non-native types). Figure:
`jaccard_by_category.png`.

**A (answer):** Leakage exists — no category is routing-identical post-LoRA
— but it is bounded (all shifts ≪ between-problem distances) and ordered by
domain similarity, so the adapter's routing influence is largely contained
to symbolic-adjacent processing. Limitation: "adjacency" is informal; a
token-overlap covariate would make the gradient claim quantitative.

---

## Q2c — Are Phase 1's specialization patterns preserved?

**D (did):** Built per-layer expert×category routing-mass matrices
(category-normalized) for each model, correlated them (Pearson, flattened),
and tracked every base-model near-pure specialist (selectivity > 0.8)
across models.

**Y (why):** Pattern correlation captures whether the *map* of who-serves-
what survived, independent of small per-problem re-rankings that Q2a
already measures; specialist tracking tests the most fragile tier
(experts with ~0.5-1% mass could be silently reassigned without moving
aggregate statistics).

**F (found):** Mean per-layer pattern correlation **r = 0.927** (min 0.849,
at layer 51); **25 of 25** base specialists kept their category under LoRA
(0 flips), specialist count stable (25 → 26). Accuracy context: symbolic
fell 15/15 → 11/13 (truncation-excluded) while routing concentrated
slightly (Δconc +0.0023) — commitment without competence. Figures:
`specialist_survival.png`, `pattern_correlation_by_layer.png`.

**A (answer):** Given r ≈ 0.93 and complete specialist survival, I conclude
the specialization architecture is preserved: LoRA re-weights routing within
an intact expert-category map. The accuracy drop alongside increased
concentration also bounds Q1c's interpretation — concentration reads as
familiarity/commitment, not correctness per se. Limitations: specialist
selectivities rest on few problems each; the divergence pipeline counts 25
specialists where Phase 1's pipeline counted 26 (minor methodological
accounting difference, documented in scripts).

---

*Infrastructure footnote (affects all three answers):* every result above
sits on a runtime-patched inference stack — the model ships with cached
generation broken in five interacting ways, the worst being that attention
layers never receive the KV cache (prompt amnesia). Patches were validated
behaviorally (verbatim mid-prompt recall probe) before the capture run;
details in `phase1_report.md` §3.3 and `RUNLOG.md`.
