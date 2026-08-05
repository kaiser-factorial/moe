# WS3 — RL vs SFT: does the training objective decide whether fine-tuning re-routes?

*Design doc **v2**, revised 2026-08-05 (v1 same day). Status: PROPOSED (no runs).
**v2 change:** substrate pivoted from WONDERLAND to the **octave-rl environment**
(`~/Projects/repo-teacher/lecture-4-prime-intellect/octave_RL`), and the router-unfrozen arms
promoted from "extension" to budget-contingent mainline. Rationale in §3a.
Infra claims verified against `PrimeIntellect-ai/prime-rl` @ `738eb50` (main, 2026-08-05) and Prime
Intellect docs the same day — this platform moves fast; re-verify before launch.*

## 1. Question

Phase 2's headline — **Family-A LoRA re-weights but never re-routes** (JSD 0.054 vs null 0.110,
25/25 specialists survive, pattern r = 0.93) — is a fact about *supervised LoRA fine-tuning on
symbolic reasoning*. WS3 asks two things:

1. **Objective:** does RL (policy-gradient with verifiable reward), matched as closely as possible
   in data, parameterization, and scale of behavioral change, move routing differently than SFT?
2. **Domain generalization (new in v2):** Arm S doubles as a test of whether
   re-weights-not-re-routes holds in a *new* domain (Octave code generation). If it does, the
   Phase-2 finding stops being about one dataset and becomes a property of the model — a stronger
   claim than replication.

Two literatures predict opposite outcomes for (1), so either result is a finding:

- **"RL touches less"**: RL fine-tuning has been reported to update only a small, sparse
  subnetwork of weights → predicts *even less* routing movement than SFT.
- **"Routers drift under RL"**: the MoE-RL stabilization literature (GSPO, routing replay) exists
  precisely because routers drift enough under policy gradients to break trainer/inference
  importance ratios → predicts *more* movement.

Tertiary (budget-contingent, §5): given the *ability* to re-route (trainable gate), does RL
exercise it more than SFT?

## 2. Infrastructure audit (Prime Intellect) — findings & conflicts

Run the RL side on Prime-adjacent infrastructure. Audit results:

**2a. Use self-hosted `prime-rl` on rented GPUs, NOT Hosted Training.** Hosted Training exposes
only `learning_rate` and `lora_alpha` — **no rank, no `target_modules`, no SFT mode**, adapter
export path undocumented, and no visibility into whether router replay (§2c) is enabled
server-side. Any of these alone kills the experiment's validity; together they rule it out.
Self-hosted `prime-rl` (the same engine underneath) exposes everything we need and writes
artifacts to local disk. (Hosted also only offers the Nano-30B **BF16** variant — moot, since BF16
is what we use anyway, §2g.)

**2b. NemotronH is a first-class `prime-rl` custom model.** Custom modeling code lives at
`src/prime_rl/trainer/models/nemotron_h/` (hybrid Mamba + LatentMoE experts + attention), with a
checkpoint converter (`converting_nemotron_h.py`) that renames HF `…mixer.gate.weight` →
`…mlp.router.gate`. Merged models require a trainer/vLLM KL-mismatch table < 0.015, so
trainer/inference parity has been checked upstream. `uv run sft` and `uv run rl` share the same
trainer, modeling code, and LoRA implementation → **all three arms run in one stack**.

⚠️ **Version-pin tension (new in v2):** the octave project validated against specific
prime-rl/verifiers revisions (its handoff warns against drifting). WS3 needs a prime-rl revision
with nemotron_h custom modeling. **First infra task: check whether octave's pinned revisions
include nemotron_h; if not, bump to the oldest revision that does and re-run the octave test suite
(37 tests) + a 20-task smoke eval against the bumped pin before anything else.** Record the final
pins in `run_config.json`.

**2c. ⚠️ Router replay exists in `prime-rl` — exactly the failure mode we feared — and is OFF by
default.** `trainer.enable_router_replay` (default `False`,
`packages/prime-rl-configs/…/trainer.py:604`) makes the trainer reuse the inference engine's
expert selections and only recompute gating weights (`trainer/models/layers/moe.py` ~L968). With
it ON, selection is mechanically pinned to the rollout policy's routing — an RL arm would
"discover" reduced routing drift by construction. **Main arms: replay OFF, and assert it**: run
`--dry-run` (writes resolved per-process configs) and grep the resolved trainer config for
`enable_router_replay = false` before every launch. This is the WS3 analog of the WS2 false-green
lesson: verify the mechanism, not the intention.

**2d. LoRA config — parity is possible but the defaults are wrong for us, twice.**
`[model.lora]` exposes `rank` (default 16), `alpha` (32), `dropout` (0.0), `target_modules`,
`modules_to_save`. Two traps:

1. **Defaults train the routed experts.** The default `target_modules` includes `experts`
   (LoRA-wraps the fused GroupedExperts) and the NemotronH latent projections
   (`fc1_latent_proj`, `fc2_latent_proj`). Family-A-parity requires routed experts FROZEN.
   Override explicitly; do not inherit defaults.
2. **Unknown module names are silently ignored** (documented behavior, for cross-architecture
   defaults). A typo'd target list trains a different family without erroring. Mitigation: the
   trainer logs the matched module list at launch (assert expected count), and every exported
   adapter gets the safetensors-header key audit from `report/adapter_registry.md` §"How to
   re-verify" before any analysis touches it.

Family-A-parity spec for all core arms: `rank=32, alpha=64, dropout=0.05`,
`target_modules = [q_proj, k_proj, v_proj, o_proj, in_proj, out_proj, <shared-expert up/down>]`.
Module names differ between HF and prime-rl's custom model — resolve the shared-expert pattern
against the trainer's logged module list at G0 (§7), and pin whatever regex matches *only* the
shared expert (Family B's bare `up_proj`/`down_proj` mistake is the cautionary tale; in the
prime-rl model routed experts are fused `experts` modules, so bare names *should* be safe — verify,
don't assume).

**2e. Router training is available for the gate-trainable arms.** The router gate is an
`nn.Parameter` (not `nn.Linear`), so LoRA cannot wrap it — but `modules_to_save = ["router"]`
marks it fully trainable (`lora.py::freeze_all_except_lora_and_specified`) and records it in the
exported `adapter_config.json`. Functionally Family D's 7.9M-param router training, inside the RL
stack. (WS2's collapse lessons presumably transfer: watch `max_load`, expect seed sensitivity, do
a short probe before a full run.)

**2f. Adapter export is PEFT-format** (`adapter_config.json` + safetensors;
`convert_adapter_to_hf` registered per model class), so `capture_routing.py` + the Phase-2 offline
pipeline should load WS3 adapters unchanged — **gated on a round-trip check** (G0), since the
prime↔HF name conversion is nontrivial for this architecture.

**2g. Model variant.** NVIDIA never published a "plain" checkpoint: the HF releases are `…-BF16`
(the full-precision weights), `…-FP8`, and `…-NVFP4` (ModelOpt-quantized *from* BF16). Same
trained weights, different numerical format. Prime Inference's cheaper lowercase alias is
therefore almost certainly a quantized serving build. All NemoH work already uses BF16
(`adapter_registry.md`) — never route routing-sensitive evals through a quantized build: FP8/FP4
rounding of sigmoid gate scores can flip near-tie top-6 selections, which is noise injected
exactly into our measurand.

**2h. Sandbox dependency (new in v2).** Octave scoring runs in Prime Sandboxes — a *separately
billed, separately failing* service (the 2026-07-30 octave run lost its post-transition eval to a
sandbox `Payment required` while GPU compute was healthy). Consequences: sandbox smoke test
**before** provisioning any GPU pod; sandbox spend is a per-rollout cost the RL/RFT arms pay and
the SFT arm doesn't (budget line, not a validity issue); the octave controller's payment-rejection
fail-fast stays enabled.

## 3. Data & environment: octave-rl (v2 pivot)

### 3a. Why the pivot from WONDERLAND

- **WONDERLAND's long gold CoTs are a validity problem for WS3, not just a cost problem.** SFT on
  rambling gold traces teaches verbose style while RL optimizes its own length → the arms diverge
  in *token distribution* before they diverge in anything interesting, and routing is measured
  over tokens. The on-policy-vs-fixed-prompt gap (§6) would be dominated by style drift, not
  objective. Octave's tight output structure (one code block, 1,536-token cap) keeps arms
  comparable.
- **The environment already exists and is validated**: native `verifiers.v1` taskset, 10 task
  families × 3 levels × 500 tasks, 6 hidden cases each, **9,000/9,000 reference cases pass** in
  pinned GNU Octave 10.2.0; constant-output hack audited; graduated reward. Deletes the entire
  "package WONDERLAND as a verifiers env" workstream.
- **Graduated reward** (`case_fraction`, weight 1.0 + small execution bonus) gives denser RL
  signal than binary boxed-match.
- What is lost: direct dataset parity with Phase 2 — addressed in §3d; it costs one gate
  reinterpretation (G2) and buys the domain-generalization claim (§1.2).

### 3b. Configuration for WS3 (deviations from the octave/Qwen setup)

The octave curriculum machinery (3-attempt debugging interaction, attempt multipliers, guided
third turn, staged controller) was built to support Qwen3.5-4B. WS3 uses the *environment*, not
the curriculum:

- **Single-turn for all arms** (`max_turns=1`). Multi-turn RL vs single-turn SFT would be a
  structure asymmetry; single-turn is also ~2–3× cheaper per rollout at 30B.
- **Thinking mode fixed uniformly across arms** — recommend OFF (`[sampling]` config), both for
  arm comparability (SFT gold has no CoT; letting R think while S learns terse code would rebuild
  the WONDERLAND length confound) and for rollout cost. If thinking-on is ever wanted, it is a
  *separate* arm pair, not a mid-experiment toggle.
- **Static level mix, no live controller.** The curriculum controller adapts the training
  distribution to the policy — an adaptive data distribution per arm would un-match the arms.
  Fix the mix at G1 and freeze it for every arm.
- Concurrency envelope: the octave handoff's stable settings (enforce_eager, TRITON_ATTN,
  batch 8 / group 2 / inflight 2) are **RTX-6000-Ada + Qwen-GDN specific**. Re-derive with a short
  concurrency ladder on the actual pod (H200 + Nemotron) before the long run.

### 3c. Splits, gold traces, and the three supervision structures

- **G1 first (baseline gate):** evaluate base Nemo-30B-A3B-BF16 per level (≥50 tasks/level,
  static, concurrency per smoke test). Nemo ≫ Qwen-4B, so expect Level 1 near ceiling and
  possibly Level 2 too. Choose the training mix that lands pooled baseline in the **10–35%**
  band (e.g. weighted L2/L3, or generate a Level 4 with the existing difficulty knobs — tighter
  tolerance, mandatory vectorization, cooperating functions). Freeze: mix, task pools, splits
  (TRAIN / HELD-OUT-EVAL / PROBE, disjoint, committed seed).
- **Arm S gold = the reference solutions.** The validation loop ("every reference answer passes
  its own harness") implies per-task reference implementations exist at build time.
  **Verify they were retained** (not just used transiently); if not, regenerate from the family
  generators — they are deterministic and seeded. S trains on prompt → reference code block,
  format-matched to what the rubric rewards.
- **Arm F (RFT) filter = full pass only** (`case_fraction == 1.0`). Training on partially-correct
  code would supervise on wrong answers. Note the resulting inherent asymmetry: R's gradient sees
  partial credit, F's data doesn't — that is part of "objective structure," document it, don't
  patch it.
- WONDERLAND is retained as a **contingency control** (§7 G2): a small WONDERLAND-SFT run in the
  prime-rl stack, only if S's routing behavior diverges qualitatively from Family A's.

### 3d. What the pivot does to Phase-2 parity (the honest accounting)

Two parities were in play. **Cross-arm parity (S vs F vs R on identical prompts, spec, stack)** is
the load-bearing one for the objective question — fully preserved; it never depended on the
dataset. **Cross-study parity with Phase 2** is reduced from *replication* to *generalization*:
Arm S no longer re-establishes the Family-A result on Family-A data; it tests the same regime in
a new domain. The routing measurement pipeline is domain-agnostic either way (§6 keeps the frozen
111-problem set for atlas continuity). The one attribution risk this creates: if S re-routes where
Family A didn't, domain and stack are confounded — resolved by the G2 contingency control, not by
giving up the pivot.

## 4. Core arms (router frozen — Phase-2-regime parity)

All arms: base `…-A3B-BF16`, LoRA per §2d, router + routed experts frozen, same frozen TRAIN
mix (§3c), same sampling config, fixed seed (documented; cheap seed-repeat if a result is
surprising), W&B on.

| Arm | Objective | Data seen by the loss | Isolates |
|---|---|---|---|
| **S** (SFT-gold) | NLL, `uv run sft` | reference Octave solutions | supervised regime, in-domain baseline + Phase-2 generalization test |
| **F** (RFT) | NLL, `uv run sft` | model's own *full-pass* samples (best-of-n from base, verifier-filtered) | self-generated data distribution, supervised objective |
| **R** (RL) | GRPO-family policy gradient (`uv run rl`, default DPPO loss + KL, replay OFF) | on-policy rollouts, graduated `case_fraction` reward | policy-gradient objective on top of F's data shift |

S→F difference = data-distribution effect. F→R difference = objective effect.

**Matching rule (the KL-asymmetry problem).** RL's KL-to-reference term is an anchor SFT doesn't
have, so "routing moved less per step" is uninterpretable. Primary comparison at **matched
competence gain**: checkpoint each arm at fixed intervals, pick per-arm checkpoints with equal
held-out improvement (case_fraction and full-solve rate; e.g. base+10pts, base+20pts), compare
routing there. Report per-step and per-weight-displacement (‖ΔW_eff‖ = ‖BA‖·α/r summed over
modules) curves as secondary views. Log `kl_tau`, entropy, reward; don't tune per-arm beyond what
stability requires; document any deviation.

## 5. Gate-trainable arms (budget-contingent mainline in v2)

Corina's call 2026-08-05: include an unfrozen arm if budget allows. The pivot changes its cost
structure — **Family D (WS2) no longer supplies the SFT-side corner for free** (different data);
the honest 2×2 on Octave needs both arms run fresh:

| Arm | Spec | Priority |
|---|---|---|
| **R+g** | Arm R + `modules_to_save=["router"]` (LoRA elsewhere unchanged) | first — the novel cell |
| **S+g** | Arm S + `modules_to_save=["router"]` | second — completes the 2×2 |

If only R+g fits the budget, it still answers "does RL exercise a trainable router," with WS2's
Family D as a qualitative (different-data) SFT cross-reference — label it as such, never as the
matched corner. WS2 collapse hygiene applies: short probe first, watch `max_load`, seed 123
sensitivity in mind. Register every WS3 adapter in `adapter_registry.md` **at creation** (proposed:
Family E = A-targets via prime-rl on octave; E+g = E + full router).

Optional cheap control if Arm R shows routing movement: **R-replay** (Arm R +
`enable_router_replay=true`) — movement vanishing under replay ⇒ selection-driven drift;
persisting ⇒ score/weight-driven. Decide after the core readout.

## 6. Measurement

Reuse the Phase-2 suite unchanged (JSD vs permutation null, specialist survival + pattern r,
Gini/concentration, subtype-preservation ratio, L17 atlas, per-layer divergence profile), applied
to each arm's adapter via `capture_routing.py` (with `apply_nemotron_patches()`, per the survival
kit). WS3 additions:

1. **Three capture sets per arm**: (a) the frozen 111-problem set — atlas continuity,
   cross-domain routing change, directly comparable to Phase-2 numbers; (b) held-out Octave
   prompts — on-domain divergence (the analog of Phase 2's "front-loaded on the native domain"
   finding); (c) **on-policy** — routing on each arm's own sampled generations
   (distribution-driven change). The (a/b)–(c) gap is itself a result, and with the length
   confound designed out (§3a) it is interpretable.
2. **Training-time routing telemetry, free**: the prime-rl trainer computes a routing-confidence
   statistic (selected-probability mass) in the MoE forward — surface it in W&B for per-arm drift
   *during* training, not just at checkpoints.
3. For +g arms additionally: direct router-weight delta (Δgate norms per layer/expert), routing
   change decomposition (how much survives with the base router swapped back in = weight-driven
   share).
4. Report everything at the matched-competence checkpoints (§4) with reward/eval curves alongside.

## 7. Gates (in order; each blocks the next)

- **G-1 — pins & sandbox (new).** Resolve the prime-rl/verifiers pin (§2b tension); octave test
  suite (37) green on the final pin; `prime sandbox create` smoke test + billing check;
  `prime pods list` clean.
- **G0 — stack round-trip.** ~20-step throwaway LoRA in prime-rl; export; safetensors-header
  audit (Family-A key pattern, expected module count, no `experts`/router keys); load via PEFT on
  the HF checkpoint; logit parity vs prime-rl eval on ~10 prompts; `capture_routing.py`
  end-to-end. Plus `--dry-run` config audit (replay off, LoRA spec, matched-module log). Also
  here: verify reference solutions are retrievable (§3c) and the Nemo concurrency envelope
  (§3b).
- **G1 — baseline band.** Per-level Nemo baseline; choose + freeze the mix/splits (§3c). Watch
  for infrastructure zeros (instant 0.0 + missing usage = grader never reached — octave lesson).
- **G2 — Arm S sanity.** S should show the Phase-2 regime qualitatively (small JSD vs null,
  specialists survive). If it clearly doesn't: run the small WONDERLAND-SFT control in this stack
  to split domain vs stack **before** interpreting R. Either control outcome is informative
  (domain-dependence of Phase 2 is a finding); an uninvestigated ambiguity is not.
- **G3 — RL health.** Reward climbs; entropy doesn't collapse; no expert-collapse signature
  (`max_load`-style stats, not just loss); sandbox error rate ~0 in trace rows (the 324/326-error
  batch is the cautionary tale — exclude any such partial batch from everything).
- **G4 — analysis** only on adapters that individually passed the G0-style audit.

## 8. Infra & cost sketch (rough, verify at launch)

- **SFT/RFT arms**: trainer only, 1× H200 (BF16 weights ≈ 63 GB; LoRA optimizer small;
  activation checkpointing on). RFT adds one batched best-of-n sampling pass (vLLM, same pod,
  sequential) + sandbox scoring for the filter.
- **RL arms**: trainer + vLLM concurrently → 2 GPUs (`--deployment.num-train-gpus 1
  --num-infer-gpus 1`), H200×2 preferred. Single-turn 1,536-token rollouts are far cheaper than
  WONDERLAND CoT rollouts would have been; sandbox cost per rollout is new. Prime marketplace or
  RunPod both fine; the EU-FR-1 volume (`gdqj7o63ik`) pins RunPod work to that DC.
- Very rough per-arm: S/F ≈ $50–100 (1× H200, hours not days at LoRA scale); R ≈ $150–400
  probe-first (1–2 h probe before any full run, house rule); each +g arm ≈ its frozen
  counterpart + margin. Sandbox budget separate — prepay/verify (§2h).
- RunPod survival kit applies (torch/mamba env recipe, no API key in create-env, pkill
  bracketing); prime-rl's `uv` env may sidestep the torch-dev landmine — still validate mamba
  kernels + `state_probe.py` on first boot.

## 9. Known limitations / threats

- **Endogenous data in RL** — decomposed by Arm F, not eliminated. New v2 wrinkle: F filters to
  full passes while R's gradient sees partial credit (§3c) — inherent to the objectives,
  documented.
- **Domain–stack confound if G2 trips** — handled by the WONDERLAND-SFT contingency control, run
  only if needed.
- **One model, one domain, one seed** per arm in the mainline; surprises get a seed-repeat before
  being believed.
- **Trainer/vLLM routing mismatch** during RL (async off-policy tolerates; DPPO masks ratios) —
  measurements all run in one stack (HF + patches) identically across arms, so cross-arm
  comparisons stay internally consistent.
- **Sandbox as a second failure domain** (§2h) — preflight + fail-fast + partial-batch exclusion.
- **Adapter identity risk** now spans 5–7 adapters across two projects — registry discipline is
  load-bearing; register at creation, not post-hoc.
- Platform drift: re-verify every §2 claim at launch; pin and record all revisions.

## 10. Relation to existing workstreams

Independent of WS1 and the Phase-3 steering sweep; decoupled from WS2 by the pivot (Family D
becomes a qualitative cross-reference, not a matched corner). Side benefit: G1 produces the first
Nemo-30B calibration of the octave environment, which feeds the lecture-4 project regardless of
WS3's outcome. Report home: new §5.6 ("Does the objective matter?") + this doc as
`report/ws3_rl_design.md`. Pre-registered headline candidates: "RL re-routes where SFT cannot" /
"Neither objective re-routes: routing stability is a property of the model, not the loss" /
"RL moves routing only through the data distribution (F ≈ R ≫ S)" / (+g) "Given a trainable
router, RL uses it; SFT doesn't" — all publishable.
