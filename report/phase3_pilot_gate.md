# Phase 3 Stage 0 Pilot — Gate Result: **PASS**

**Run**: 2026-06-12, pod `s0948eojx7s6gh` (H100 HBM3, EU-FR-1), ~38 min GPU,
~$2.40 total incl. fetch. Artifacts: `outputs/analysis/steering/`
(vectors.npz, sites.json, pilot/ with 14 conditions × 6 problems, run.log).
Design: `phase3_design.md` §4 Stage 0.

## Setup (as designed, mean-diff fallback chosen up front for budget)

Mean-difference symbolic-direction vectors (18 symbolic vs 18 stratified
non-symbolic, mean-pooled prompt states, base model), extracted at 16 sites
(8 MoE depths + nearest-preceding Mamba blocks) — all banked for Stage 1.
Pilot injected at **m16** (post-Mamba) and **e17** (post-MoE),
`h += α·σ_L·v̂`, α ∈ {±0.5, ±2, ±8}, greedy, ≤256 new tokens, 6 problems
(3 symbolic + 3 control). Noise floor (α=0 twice): JSD = 0.0000 exactly
(greedy determinism confirmed).

## Headline numbers

| α | JSD vs base (m16 / e17) | Δ sym-specialist occupancy (m16 / e17) | text |
|---|---|---|---|
| +0.5 | 0.092 / 0.088 | **+0.120 / +0.092** | coherent, on-task |
| −0.5 | 0.122 / 0.130 | **−0.115 / −0.117** | coherent, on-task |
| +2 | 0.492 / 0.515 | +0.101 / +0.087 | degenerate ("1010…" loops) |
| −2 | 0.444 / 0.450 | −0.119 / −0.119 | fluent but answers a *different, generic* question |
| ±8 | 0.51–0.57 | (saturated) | degenerate |

Anchors: LoRA effect 0.0537, between-problem null 0.110. Baseline
symbolic-specialist occupancy on symbolic problems: 0.1186 (so −0.1186 =
complete specialist evacuation; +0.12 = doubling).

## Findings

1. **H1 confirmed — routing is steerable from the residual stream.** At the
   *weakest* tested strength (α=0.5, ~50× smaller perturbation than α=8 in
   norm terms), JSD is already ~2× the LoRA effect while text stays fully
   coherent and on-task.
2. **The steering is semantically directional, not just disruptive.**
   Specialist occupancy tracks the sign of α almost symmetrically: +0.5
   *doubles* mass on the two symbolic specialists (L13/e21, L24/e20);
   −0.5 evacuates them to ~zero. LoRA, for comparison, moved occupancy by
   +0.0023. We have a ~50× stronger routing lever than fine-tuning produced,
   at zero weight changes.
3. **Locality signature (e17, α=+0.5)**: per-layer JSD peaks immediately
   downstream of the injection site (L20: 0.234, L24: 0.215, vs ~0.05 at
   L1–L17 from token-trajectory feedback) and decays with depth — the
   response map Stage 1 wants exists and is resolvable.
4. **Mamba-site ≈ MoE-site at L16/17** — no layer-type difference at this
   depth; discriminating the families needs Stage 1's depth sweep.
5. **Bonus interpretability finding**: at α=−2 (away-from-symbolic) the
   model fluently answers a *different, generic-sounding* problem ("the role
   of the government…", "the responsibility of the employer…") instead of
   the symbolic prompt it was given — the vector seems to move *what kind of
   problem the model believes it is solving*, not just surface tokens.
6. **Methodological correction**: mean token logprob is NOT a usable
   coherence metric — the degenerate "1010…" loops at α=+2 score *better*
   (−0.13) than honest baseline reasoning (−0.17). Confidence is not
   coherence. Stage 1/2 must use a repetition/loop detector (e.g. distinct
   n-gram ratio) + spot-reads instead.

## Decisions for Stage 1 (pending funding)

- Usable steering window is **|α| ≤ ~1**; revise grid to
  α ∈ {±0.25, ±0.5, ±1} (design doc §3.4's ±{2,4,8} is overcooked —
  degeneracy onset is between 0.5 and 2).
- Replace mean-logprob coherence gate with distinct-4-gram ratio
  (flag < ~0.3) + manual spot-read per condition.
- Vectors for all 16 sites already extracted and banked (vectors.npz) —
  Stage 1 needs no re-extraction, only steered captures.
- Stage 2 causal test inherits a strong setup: at α=+0.5 occupancy doubles
  with coherent text — exactly the lever needed to test
  competence-vs-familiarity. Mind the asymmetry: the negative direction has
  a floor (occupancy ~0.12 → 0), the positive direction has headroom.

## Ledger

Balance $5.12 → $3.03. Spent: pilot pod $2.09, fetch pod ~$0.25 (created
because EU-FR-1 had zero CPU instances; H100 for 4 min). Remaining $3.03 is
not enough for Stage 1 (~$5–6.5) — resume when the card issue is fixed.
