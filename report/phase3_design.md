# Phase 3 Design: Steering Expert Routing via Layer-Type Control Vectors

**Status**: Stage 0 pilot RUN 2026-06-12 — **gate PASSED**, see
`phase3_pilot_gate.md` (revises §3.4 strength grid to α ∈ {±0.25, ±0.5, ±1}
and replaces the logprob coherence metric with a loop detector)
**Depends on**: Phase 1 (`phase1_report.md`), Phase 2 (§4.4, divergence analysis in `outputs/analysis/divergence/`)
**SPEC reference**: §4 Phase 3, §7 (repeng notes)

---

## 1. Motivation

Phase 2 left a dangling causal question. On symbolic problems, LoRA routing
*concentrated* toward the symbolic specialists (Δconc +0.0023) while accuracy
*fell* (15/15 → 11/13) — we interpreted concentration as **familiarity, not
competence**. But Phase 2 was correlational: LoRA changed expert weights *and*
routing simultaneously, so the contributions can't be separated. And because
the adapter never touched routers, all observed divergence was indirect —
proving routing *can* be moved through the residual stream, but not from
where, or with what consequence.

Phase 3 breaks the confound: steer routing **without touching any weights**
via activation-addition control vectors, and observe (a) where routing
control lives in the hybrid stack, (b) whether routing toward specialists
causally affects competence.

## 2. Research Questions & Hypotheses

**Q3a (mechanistic)**: Which injection sites move routing most per unit of
steering — post-Mamba or post-MoE residual positions, and at which depths?

- **H1**: Routing is steerable from the residual stream at all (JSD response
  exceeding the within-condition noise floor). Phase 2 guarantees indirect
  influence *exists*; H1 tests that we can harness it deliberately.
- **H2**: Sensitivity is depth-structured, peaking near the specialist band
  (L13–L27, where near-pure specialists cluster), rather than uniform.

**Q3b (causal)**: Does steering routing toward the 25 symbolic specialists
change symbolic accuracy?

- **H3 (familiarity hypothesis, from Phase 2)**: accuracy is flat or *drops*
  under specialist-ward steering — concentration marks familiarity, not
  competence. The alternative (accuracy improves) would mean routing was a
  bottleneck and specialists carry real competence.

All three outcomes of Q3b are informative; H3 has directional support from
Phase 2 but the test is the point.

## 3. Method

### 3.1 Control vector construction

**Primary**: adapt **repeng** (PCA over paired hidden-state differences) to
the NemotronH hybrid stack. repeng's `ControlModel` wraps standard
`decoder.layers[i]`; Nemotron-3-Nano interleaves Mamba2 / attention / MoE
blocks, so wrapping needs surgery.
**Timebox: one working session.** If repeng fights the architecture, fall
back to **hand-rolled mean-difference vectors** (mean hidden state over
positive set − mean over negative set, per layer) injected via forward
hooks — same infrastructure as `capture_routing.py`. Mean-diff is the
standard ActAdd baseline and is fully transparent; the fallback costs us
nothing scientifically.

**Vector reading pass**: one forward pass per problem per condition over the
contrast set, capturing residual-stream states at every candidate injection
site. Cheap (no generation needed — last-token or mean-pooled states).

### 3.2 Contrast definition

Reuse the existing 111-problem dataset (`data/problems.json`):

- **Positive set**: 18 symbolic problems
- **Negative set**: 18 problems sampled evenly from the other five
  categories (seeded, stratified; record the sample)

This makes the vector *the symbolic-routing direction as it exists in our
own Phase 1–2 data* — directly comparable, zero new curation. (Instruction-
style pairs are a Phase 3.5 option if the direction proves too entangled
with surface features like math notation.)

### 3.3 Injection sites

Two site families, mirroring SPEC's layer-type question:

- **Post-Mamba**: residual stream immediately after Mamba2 block output
- **Post-MoE**: residual stream immediately after MoE block output

The 23 MoE layers sit at indices
{1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51}.
Stage 1 sweeps a pruned grid: **8 depths** spanning early / specialist-band /
late (e.g. 3, 8, 13, 17, 20, 27, 38, 47) × both site families. One site per
run (no stacking) so effects are attributable.

### 3.4 Steering strength

Vectors are unit-normalized; injected as `h ← h + α·σ_L·v̂` where σ_L is the
mean residual-stream norm at layer L (so α is comparable across depths).
Grid: **α ∈ {−8, −4, −2, +2, +4, +8}** (sign = away-from / toward-symbolic).
Calibrate the ceiling in the pilot: if α=8 already destroys coherence,
shrink; if α=8 does nothing, extend.

## 4. Staged Execution

### Stage 0 — Pilot (~0.5 h GPU)
Extract vectors; inject at 2 layers (one Mamba-site, one MoE-site, ~L17) ×
{±0.5, ±2, ±8} on 6 problems. Verify: hooks fire, generation stays coherent at low
α, routing JSD responds at some α. **Gate**: if routing JSD at *every*
pilot condition is below the noise floor (§5), stop and rethink before
spending the sweep budget.

### Stage 1 — Mechanistic sweep (~1.5–2 h GPU)
16 sites × 6 strengths × 24 probe problems (12 symbolic / 12 non-symbolic,
fixed seed), capture routing at all 23 MoE layers per run (existing capture
infra), greedy short-generation only (≤256 tokens — routing capture doesn't
need full solutions).
**Output**: per-site, per-α routing response map → pick the **2–3 most
routing-responsive sites** for Stage 2.

### Stage 2 — Causal test (~1–1.5 h GPU)
At winning sites only: full generation (matching Phase 1–2 decoding params)
on all 18 symbolic problems × {α=0 baseline re-run, 2–3 chosen α values},
plus 12 non-symbolic controls for collateral-damage check. Grade accuracy
with the Phase 1–2 protocol.

## 5. Metrics & Calibration

| Metric | Definition | Calibration anchor |
|---|---|---|
| Routing shift | JSD(steered ‖ unsteered), per layer, mean over probe set | LoRA effect = 0.0537 (symbolic); between-problem null = 0.110; noise floor = JSD between two α=0 runs (measure in pilot; greedy ⇒ expect ≈0) |
| Specialist occupancy | total routing mass on the 25 surviving specialists (from `specialist_survival.csv`), Δ vs unsteered | Phase 2 Δconc +0.0023 as the "LoRA-sized" reference |
| Top-6 overlap | Jaccard of selected experts, steered vs unsteered | Phase 2 per-category tables |
| Accuracy | Phase 1–2 grading protocol, symbolic subset | base 15/15, LoRA 11/13 |
| Coherence | mean logprob of generated tokens vs unsteered + manual spot-read; flag conditions with degenerate output | exclude incoherent conditions from accuracy claims |

**Headline comparisons**: (1) routing response: Mamba-sites vs MoE-sites by
depth (Q3a); (2) accuracy vs specialist occupancy across α (Q3b) — the
causal version of Phase 2's correlation.

## 6. Success / Kill Criteria

This phase is optional; pre-commit to exits.

- **Kill after pilot**: no pilot condition moves routing above noise floor
  → write a short negative-result note (§4.3 becomes "routing is robust to
  residual steering at tested scales"), stop.
- **Kill after Stage 1**: routing moves but only at α that destroys
  coherence → report the mechanistic map only; skip Stage 2.
- **Success (minimum)**: Stage 1 map with a clear layer-type/depth story —
  fills SPEC §4.3 regardless of Stage 2 outcome.
- **Success (full)**: Stage 2 accuracy result in either direction with
  specialist occupancy demonstrably moved (≥ LoRA-sized shift).
- **Budget kill**: hard stop at $20 spend; salvage whatever stage completed.

## 7. Ops Plan

- **Top-up**: +$20 (≈6 h H100 @ $3.29/hr, EU-FR-1; balance currently $5.35).
  Estimated need: ~3–4 h total across stages, leaving slack for env friction.
- **Volume**: `gdqj7o63ik` (EU-FR-1, 175 GB) already holds HF cache
  (`/workspace/hf`), repo, and Phase 2 artifacts — pod must be created in
  EU-FR-1. H100 HBM3 is the reliably available SKU there.
- **Env**: follow **recipe v2** (RUNLOG/memory): release torch 2.8.0 from
  cu128 index WITH deps, uninstall torchvision/torchaudio, then mamba-ssm +
  causal-conv1d with `--no-deps --no-build-isolation`. Use
  `capture_routing_h100.py` (ptxas patch neutered) as the capture base.
- **Safety rails**: account API key goes in a root-only file via ssh after
  boot — never in pod env at create (secret-scanner kills the pod).
  Self-deleting run script pattern from `phase2_pod.sh` (write
  `PHASE3_<STAGE>_COMPLETE`, sync results to volume, delete pod).
- **Hygiene**: stage scripts in repo (`scripts/phase3_*`), results synced to
  volume then pulled to `outputs/analysis/steering/`. The 5 dead pods from
  Phase 2 can be deleted during the first ssh session.

## 8. Deliverables (maps to SPEC §6 Phase 3)

- `scripts/phase3_extract_vectors.py`, `scripts/phase3_steer_capture.py`,
  `scripts/phase3_pod.sh`
- `outputs/analysis/steering/control_vector_routing_effects.csv` — per-site,
  per-α: JSD, specialist occupancy Δ, top-6 Jaccard
- `outputs/analysis/steering/steered_accuracy.csv` — Stage 2 grading
- Figures: routing-response heatmap (site × α), accuracy-vs-occupancy plot
- Report **§4.3** + Phase 3 DYFA (Q3a/b)

## 9. Risks

- **Direction entanglement**: symbolic-vs-rest vector may encode surface
  features (notation density) rather than "symbolic reasoning"; mitigation —
  inspect nearest-neighbor prompts of the direction, fall back to
  instruction pairs if needed.
- **repeng adaptation rabbit hole**: hard timebox (one session), proven
  fallback.
- **Small accuracy N** (18 symbolic problems): report effect sizes with
  exact binomial CIs; treat Stage 2 as preliminary, as SPEC already frames
  Phase 3.
- **Greedy-decoding noise floor ≈ 0** could make tiny JSDs look
  significant — that's why the gate uses the *LoRA-sized* effect (0.0537)
  as the meaningful-shift reference, not the noise floor alone.
