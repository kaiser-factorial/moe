# Final Report — Outline / Scaffold

**Status:** scaffold (2026-06-23). Section stubs + purpose + source docs. Runs
still pending (WS1 divergence on B/C; WS2 Family-D router training), so Results
has placeholders.

**Audience:** familiar with mechanistic interpretability + LoRA; **not** assumed
to know MoE internals or representation engineering / control vectors. The
Background must bridge *repEng-on-dense-models → repEng-on-MoE-routing* — that
bridge is the conceptual hinge of the whole project, so give it room.

**Two guiding principles for placing material:**
- *Debug content:* if not knowing it would make a reader **misread a result**,
  it goes in **Methods/Results** (state as resolved fact). If it only would have
  **saved us time**, it goes in the **Epilogue**.
- *Back matter:* the **Epilogue** is narrative denouement, read in order
  ("lessons from doing mech-interp on a hybrid MoE"). **Appendices** are
  jump-in-grab-a-number-leave reference. Epilogue first, appendices after.

`report/SUMMARY.md` is the canonical findings spine — most of Results/Discussion
is a polish + re-sequencing of it.

---

## 1. Abstract / TL;DR
One paragraph: the question, the headline (re-weight not re-route;
concentration = familiarity, not competence), and the family/steering evidence.
*Sources:* `SUMMARY.md`, `DYFA.md`, `phase1_DYFA.md`.

## 2. Introduction & Motivation
The question: when you fine-tune an MoE, does it **re-route** (change which
experts fire) or merely **re-weight** (same map, different inputs)? And does
routing concentration track **competence** or just **familiarity**? Why this
matters for interpreting MoE fine-tunes.
*Sources:* `SPEC.md` (motivation only — note SPEC's technical claims were later
corrected, see §4), `SUMMARY.md` intro.

## 3. Background  ← the audience ramp; give it room
- **3.1 MoE basics** — routed experts, top-k, the router/gate, load balancing.
- **3.2 NemotronH's hybrid stack** — 52 layers: Mamba-2 + attention + MoE;
  128 routed experts + 1 *shared* (always-on) expert per MoE layer; which layers
  are which.
- **3.3 The router in detail** — DeepSeek-style **sigmoid** top-k scoring;
  `e_score_correction_bias` (aux-loss-free balance, a frozen buffer); why the
  router is `gate.weight` and nothing else.
- **3.4 MoE design compare/contrast** — Switch / Mixtral / DeepSeek-V3 /
  NemotronH (softmax vs sigmoid routing, aux-loss vs aux-loss-free, shared
  experts, hybrid vs pure-transformer). Orients the MoE-naive reader.
  **Call out NemotronH's sigmoid (DeepSeek-style) routing explicitly** — most
  MoE intuition (Switch/Mixtral) assumes *softmax* gating where per-token expert
  probs sum to 1; NemotronH scores each expert independently via sigmoid (+ the
  aux-loss-free `e_score_correction_bias`). This isn't trivia: it's load-bearing
  for our Family-D router training — the textbook Switch load-balance loss had to
  be re-derived (softmax-normalize P_e) because the raw sigmoid version tracked
  gate *sharpness*, not allocation, and failed to penalize a routing collapse
  (see Methods §4.5 / Epilogue). A clean "theory detail → it bit us in practice"
  hook for the MoE-literate-but-not-MoE reader.
- **3.5 Representation engineering, traditional sense** — control vectors on
  *dense* models (how repEng was used before this project); set up the leap to
  steering *routing* via the residual stream.
- **3.6 The adapter families (A–D)** — what each LoRA actually trains; needed
  before any result. *Source:* `adapter_registry.md` (authoritative table).
*Sources:* `phase1_report.md` (model structure), `expert_anal_probe.md`
(layer-type map: Mamba/MoE/attention indices + module names), `adapter_registry.md`,
`lora_target_modules.md`, modeling_nemotron_h.py notes in
`router_training_plan.md` §2–3.

## 4. Methods
- **4.1 Adapter families & what they train** — the 3-family table; **the LoRA
  target-module correction** (SPEC §2 was wrong; routed-experts vs
  attn+shared). *Science-load-bearing debug — state cleanly here, not in the
  epilogue.* *Sources:* `adapter_registry.md`, `lora_target_modules.md`.
- **4.2 Routing capture pipeline** — instrumented inference, per-token top-6
  capture at 23 MoE layers, the gate hooks. *Source:* `phase1_report.md`,
  scripts/capture_routing.py.
- **4.3 Divergence metric** — per-problem per-layer expert-mass JSD; the
  between-problem null. *Source:* `SUMMARY.md` §4.4, scripts/divergence_analysis.py.
- **4.4 Steering protocol** — mean-diff control vectors, 16 sites, α-scaling,
  coherence detector. *Sources:* `phase3_design.md`, `phase3_pilot_gate.md`.
- **4.5 Router training (Family D)** — gate-only finetune, LR probe, aux-loss
  balance, locked decisions. *Source:* `router_training_plan.md`.
- **4.6 Known upstream bug (caveat)** — NemotronH cached-generation bug + how
  capture works around it. *Science-load-bearing — belongs here.* *Sources:*
  `upstream_bug_investigation.md`, `upstream_bug_runtime_plan.md`.

## 5. Experiments & Results
- **5.1 Phase 1 — routing structure** — two-tier specialization, L17–20
  specialists, difficulty invisible, concentration↔correctness. *Sources:*
  `phase1_report.md`, `phase1_extended.md` (A–I).
- **5.2 Phase 2 — fine-tuning divergence (Family A)** — re-weighting not
  re-routing; symbolic JSD 0.0537 vs null 0.110; specialists survive;
  concentration = familiarity. *Source:* `SUMMARY.md` §4.4.
- **5.3 Family comparison (WS1, B & C)** — does training the routed experts
  unlock *direct* re-routing? *[PENDING RUN]* *Source:* `family_comparison_plan.md`.
- **5.4 Phase 3 — steering** — residual-stream control vectors move occupancy
  ~50× the LoRA; α regimes; coherence findings. *Source:* `phase3_pilot_gate.md`.
- **5.5 Family D — router training** — direct re-routing vs capability.
  *[PENDING RUN]* *Source:* `router_training_plan.md`.
- **5.6 CoT capability across families** — the dependent variable for the
  "familiarity not competence" claim. Capability pass/fail is a **free byproduct
  of every routing capture** (`results.jsonl` per-problem P/F), so extract +
  harmonize it across base / A / B / C / D on the same 111-set (fix the UNKN
  non-binary cases + the unscored social/ethical category first). **Caveat:**
  captures are single-sample (do_sample, temp 0.6) → too noisy for a small-effect
  claim; back the headline with a focused **deterministic (greedy) or pass@k**
  eval on at least base + A + D. Keep `expert_anal_lora.md` (Family-B, 14-task
  set) as a qualitative *failure-mode* companion only — not the pass-rate source.
  *Sources:* `scripts/score_capability.py` (harmonized grader: per-type robust
  extraction, post-</think> answer region, PARSE_FAIL vs FAIL, NO_KEY excluded;
  validated — reproduces the symbolic 83→61% drop), capture `results.jsonl` (all
  families), `expert_anal_lora.md` (qual).
  **Micro-finding (the symbolic "drop"):** base 83% → Family-A 61% symbolic is
  *mostly a truncation artifact, not competence loss*. Of 6 base→A pass→fail
  flips, **4 are token-cap truncation** (A's CoT ran to the 3072 cap before
  boxing — e.g. symb-easy-13: 606→3072 tokens; symb-medi-17/hard-12/hard-06 all
  capped), only 2 are genuine reasoning errors. Aggregate: symbolic mean gen
  1405→1569 tok, cap-hits 3/18→5/18. I.e. the adapter taught the training set's
  verbose "hypothesis-explosion" CoT style (cf. `expert_anal_lora.md`), which
  collides with the fixed budget → truncation. The robust effect is *behavioral
  (longer CoT)*, supporting "re-weight not re-route"; the competence claim is
  confounded by the token cap. n=18, single-sample → magnitude soft, mechanism
  clear. **Recoverable** with a higher cap / CoT-compression prompt.

## 6. Discussion
Re-weight-not-re-route synthesis; concentration = familiarity not competence;
what direct routing control (steering, Family D) does and doesn't buy; limits
(quantization confounds, single model, dataset scope).
*Sources:* `SUMMARY.md`, `DYFA.md`.

## 7. Conclusion
What we learned about routing rigidity in a hybrid MoE; implications for
interpreting MoE fine-tunes; future work.

## 8. Epilogue — Engineering & Debugging Log  ("lessons from mech-interp on a hybrid MoE")
Narrative, read-in-order. The *cautionary-tale* version of the adapter-identity
confusion (the clean facts already live in §4.1); the NemotronH cached-gen bug
hunt; mamba/torch ABI; RunPod ops (secret scanner, DC pinning, PEP668, pkill
self-match, kagglehub-stdout); Kaggle 2×T4 (4-bit, the aux-loss
grad-checkpointing bug). *Sources:* `debug/KAGGLE_DEBUG.md`,
`debug/runpod_ws1_setup_2026-06-23.md`, `errors_postmortem.md`, `RUNLOG.md`.

## 9. Appendices  (reference; jump-in-grab-a-number)
- **A. Adapter registry** — the full 3-family table + verification method.
  *Source:* `adapter_registry.md`.
- **B. Hyperparameters & training recipes** — LoRA configs, the Wonderland SFT
  recipe, router-training settings.
- **C. Full result tables** — per-layer / per-category JSD, specialist lists,
  steering grid. *Source:* `phase1_extended.md`, `SUMMARY.md` tables.
- **D. Derivations** — JSD null construction; load-balance aux-loss form.

## 10. References

---

### Open TODOs for the report
- [ ] WS1 results (B & C divergence) → fill §5.3.
- [ ] Family D results → fill §5.5.
- [ ] §5.6: harmonized capability scorer over all families' `results.jsonl`
      (fix UNKN + unscored social/ethical), then a deterministic/pass@k eval on
      base + A + D for the headline. Demote `expert_anal_lora.md` 6/14 to qual.
- [ ] §4.5/§5.5: log router-training dynamics (aux 0.01 too weak on a sigmoid
      gate-only router; `lm` can't diagnose router health — collapsed router has
      low `lm`; see `router_training_plan.md` §10). Sigmoid-thread payoff for §3.4.
- [ ] §5.5 POSSIBLE HEADLINE: the **concentration sweep** (`router_training_plan.md`
      §11) — train routers across `AUX_COEF` from balanced→specialized, plot
      capability vs routing-concentration; check whether it concentrates onto the
      Phase-1 symbolic specialists. Cleanest direct test of "familiarity ≠ competence."
- [ ] §5.6 follow-up RUN: re-eval symbolic (base + A) with a **higher token cap**
      (and/or a "commit after 3 hypotheses" compression prompt) to separate
      truncation from true competence loss — quantify how much of the 83→61%
      "drop" is recoverable. (Cheap; only the symbolic 18 + a budget bump.)
- [x] Source triage (2026-06-23): `expert_anal_probe.md` = LIVE (layer-map ref,
      §3.2). `expert_anal_lora.md` = LIVE but Family-B/14-task, qual only.
      `HANDOFF.md` = navigation pointer, not a citable source (defers to
      SUMMARY.md; pre-dates WS1/WS2). `Nemo_Lora_Experts.md` = SUPERSEDED by
      `scripts/` (provenance only). Cite only the first two.
