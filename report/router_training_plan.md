# Router Training Plan (WS2 → Family D)

**Status:** draft 2026-06-23. Detail for WS2 in `report/family_comparison_plan.md`.
Produces **Family D**: the only adapter that trains the router gate directly,
so routing changes are *direct* (selection geometry moved) rather than the
*indirect* re-weighting seen in Families A–C (router frozen in all of those).

## 1. Comparison discipline — same data, same recipe

To keep the only variable "what gets trained," reuse exactly what the other
adapters used (verified from `lora3` `training_args.bin` + `trainer_state.json`):

| Setting | Value (match the others) |
|---|---|
| Dataset | **`WONDERLAND_FINAL_MASTER.jsonl`** — 9,500 chat-format symbolic CoT records (gravity/cipher/bit-manip/unit-conversion/roman-numeral/symbolic-transform) |
| Format | chat SFT (`messages` = user problem → assistant CoT), LM loss on assistant tokens |
| Epochs / steps | 1 epoch ≈ **1,188 steps** |
| Effective batch | 8 (per_device 1 × grad_accum 8) |
| LR / schedule | **1e-4**, cosine, warmup_ratio 0.05 |
| Optimizer | AdamW (torch), weight_decay 0, max_grad_norm 1.0 |
| Precision / seed | bf16, seed 42 |

Router-specific deviations from this recipe are flagged in §4 (LR, aux loss).

## 2. What is actually trainable (verified from modeling_nemotron_h.py)

Per MoE layer, `mixer.gate` (`NemotronHTopkRouter`) has exactly:
- **`gate.weight`** — `nn.Parameter`, `[128, 2688]`, **fp32**. The gating matrix.
  This is the *only* gradient-trainable router parameter. ×23 MoE layers =
  **~7.9M params** total (trivial vs 30B).
- `gate.e_score_correction_bias` — `[128]` **registered buffer (no grad)**.

## 3. Router mechanics that shape the training problem

From `NemotronHTopkRouter.forward` / `get_topk_indices`:

```
router_logits = F.linear(hidden.fp32, weight.fp32)     # the trainable part
scores        = sigmoid(router_logits)                  # DeepSeek-V3: SIGMOID, not softmax
topk_indices  = topk( scores + e_score_correction_bias )# bias affects SELECTION only
topk_weights  = scores.gather(topk_indices); normalize; * routed_scaling_factor
```

Three consequences:

1. **Gradient does reach `gate.weight`.** The combination weights (`topk_weights`)
   come from the differentiable `scores`, and they scale the selected experts'
   outputs → loss backprops into `gate.weight` for the selected experts. So
   training works — it sharpens/softens affinity for *currently selected* experts.
2. **Selection is discrete** (top-k argmax, no straight-through). Which experts
   fire changes only as logits cross thresholds → expect *gradual* re-routing,
   not instant. This is the whole point (direct lever), but it means short runs
   may move occupancy modestly; watch the trajectory.
3. **No safety net for load balance.** The bias that balances load
   (`e_score_correction_bias`) is a frozen buffer and is **not updated in
   forward**, and `NemotronHMOE.forward` has **no aux-loss / router_logits
   plumbing**. So training `gate.weight` alone, with the bias frozen, can push
   mass onto a few experts → **collapse / dead experts**. This must be handled.

## 4. Mitigations (the part that needs real implementation)

- **Add an explicit load-balance auxiliary loss.** Capture per-layer
  `router_logits`/`scores` with forward hooks on `mixer.gate` — **reuse the hook
  infrastructure already in `scripts/capture_routing.py`** — and add the standard
  Switch/DeepSeek load-balance term `α · N · Σ_e f_e · P_e` (f_e = fraction of
  tokens routed to e, P_e = mean gate prob to e) to the LM loss. Start α≈0.01.
- **Monitor per-expert load every N steps**; hard-stop if any expert exceeds a
  cap (e.g. >5–10% of tokens) or if >X experts go dead.
- **Consider a lower LR than 1e-4.** Directly moving selection geometry is more
  sensitive than LoRA on side-paths; a sweep of {1e-4 (recipe), 3e-5, 1e-5} on a
  100-step probe, picking the highest LR that stays balanced, is worth the hour.
- Keep `gate.weight` in **fp32** (it already is) for stable router optimization.

## 5. Mechanism: how to make it a loadable adapter

We want Family D to load through the *same* `PeftModel.from_pretrained` path the
capture pipeline uses, so analysis is apples-to-apples.

- **Preferred: PEFT `modules_to_save=["gate"]`.** Saves the trained gate modules
  inside the adapter; `PeftModel.from_pretrained` restores them. Note `LoraConfig`
  also requires a `target_modules` — give it a throwaway/no-op LoRA target (e.g. a
  single `in_proj` at rank 1) or set the LoRA scaling so it's inert, so the
  *only meaningful* trained weights are the gate. (Decision point — see §8.)
- **Alt: plain unfreeze + custom save.** `requires_grad=True` on every
  `mixer.gate.weight`, freeze all else, train with HF `Trainer`, then save just
  those 23 tensors and write a 3-line loader patch for `capture_routing.py`.
- Either way: **gradient checkpointing is required** — backward to the gates
  needs activations through the full 30B forward; 60 GB weights + activations fit
  an 80 GB H100 only with checkpointing on.

## 6. Eval & the decisive read-out

After training Family D:
1. **WS1 divergence on D** (reuse `capture_routing.py` + `divergence_analysis.py`)
   — now you should see *direct* re-routing: higher JSD(base↔D), shifted/possibly
   new specialists, vs Families A–C's indirect-only shifts.
2. **CoT capability pass-rate** on the same set as `expert_anal_lora.md`.

Interpretation (this is the headline):
- **D re-routes strongly but capability does NOT beat base/A** → confirms
  "concentration = familiarity, not competence": you can move the selection map
  directly and it doesn't buy capability.
- **D re-routes and capability climbs** → partially falsifies it; direct routing
  control is a real lever the indirect path under-exploited.

## 7. Compute & cost (H100, EU-FR-1, ~$3.29/hr)

- Training: 1 epoch / 1,188 steps, full 30B forward+backward (only 7.9M trainable)
  with grad checkpointing. Estimate **~4–8 GPU-h** (timebox; checkpoint every
  ~200 steps to the volume).
- + WS1 capture on D (~2–3 h) + CoT eval (~1 h).
- Round total ≈ **$25–40**. Pre-commit a cap; self-stop on collapse.

## 8. Decisions — LOCKED 2026-06-23

1. **Mechanism: plain unfreeze.** `requires_grad=True` on the 23 `gate.weight`
   only, everything else frozen; save those 23 tensors as a state_dict; add a
   ~3-line `strict=False` gate-loader to `capture_routing.py`. Purest
   "router-only" attribution; no PEFT dummy-target confound.
2. **LR: 100-step probe first** over {1e-4, 3e-5, 1e-5}; pick the highest LR that
   stays load-balanced. Use 1e-4 in the full run only if the probe shows it's
   stable. Report the chosen LR + rationale.
3. **Load balance: aux-loss only** (Switch/DeepSeek term from gate-score hooks,
   α≈0.01) for the primary run. Keep clean attribution: any re-routing = learned
   `gate.weight` preference. Hold `e_score_correction_bias` refresh as a
   contingency, triggered only if load monitoring shows collapse.
4. **Scope: D1 now, D2 conditional.** Run router-only (D1) first, ~$45 cap for the
   round (probe + train + capture + CoT eval). Run D2 (router + Family-A LoRA,
   separately-authorized ~$25) ONLY if D1 returns a null capability result — that's
   the case where D2 is needed to rule out "router-alone underpowered."
   Watch cumulative session spend (WS1 also running).

## 9. Suggested order
1. (zero-GPU) Write `scripts/train_router.py` (SFT loader for
   `WONDERLAND_FINAL_MASTER.jsonl` + gate-only trainable + load-balance hooks).
2. 100-step LR/balance probe → pick LR.
3. Full 1-epoch run → Family D adapter on the volume.
4. WS1 divergence on D + CoT eval → compare to A–C.

## 10. Probe / training-dynamics findings (Kaggle RTX Pro 6000, 2026-06-23)
Ran the LR probe (BF16, gate-only, the FIXED softmax-normalized aux). Results:
- **1e-4** collapses ~step 25–30: `max_load` 0.51 → 0.95; `lm` erratic (spikes to
  1.5). **3e-5** collapses too, ~same step: `max_load` → 0.84. Both at the
  textbook `AUX_COEF=0.01`. Binding constraint = **aux strength, not LR.**
- **Aux-fix validated:** the normalized aux sits ~1.6–2.4 and *rises* at the
  collapse moment (e.g. 2.27 at the step-25 tip) — an honest signal, vs the old
  raw-sigmoid version that *fell* during collapse.
- **METHODS NOTE 1 (for §4.5 / §3.4 sigmoid thread):** a textbook-strength (0.01)
  load-balance loss is **too weak** to keep gate-only training stable on a
  frozen-bias *sigmoid* router. Needs a heavier hand (testing `AUX_COEF` 0.05 →
  0.1). The correctly-aimed-but-gentle aux can't fight rich-get-richer once it tips.
- **METHODS NOTE 2 (important):** `lm` does **not** diagnose router health — a
  collapsed router achieves *low* `lm` (3e-5: at step 30 `lm`=0.62 while
  `max_load`=0.845 and still rising). The model just funnels everything through a
  few experts and models the data fine. **Selection must watch `max_load`, not
  `lm`.** (A Family-D adapter picked by `lm` could ship a degenerate collapsed
  router that tells us nothing about re-routing.)
- Timing: ~17 min / 40 steps at MAX_LEN=2048 → full 1,188-step run ≈ ~8 h (more at
  4096) — at Kaggle's session limit; plan the resume path.

## 11. PLANNED — the "concentration sweep" (specialized vs balanced routers)
Idea (Corina, 2026-06-23): instead of only fighting the collapse, **use it.**
Train a *spectrum* of routers by varying `AUX_COEF` from strong (balanced,
`max_load`~0.55) to weak/zero (specialized → concentrated, `max_load`→0.85+), then
measure **capability vs. routing-concentration**. This turns the "annoying
collapse" into the project's cleanest test of the central thesis:
- If capability is **flat or falls** as the router concentrates onto fewer experts
  → strong "concentration = familiarity, not competence."
- If a deliberately-**specialized** router **improves** symbolic accuracy → direct
  re-routing *can* buy competence (partial falsification).
Also analyze **where** it concentrates: does it funnel symbolic tokens onto the
Phase-1 symbolic specialists (map rigid, training just amplifies existing
structure) or onto new experts (real re-routing)? Unifies with Phase-3 steering
(both push expert occupancy — one trained, one injected).
**Caveat:** the fully-collapsed end (few experts) likely degrades *generation*
coherence even at low `lm` — so sweep the *moderate* concentration band, and
score capability + output coherence, not just `lm`. The probe's saved
`router_state_probe_*.pt` files are already routers at different concentrations —
cheap first data points before any dedicated run.
