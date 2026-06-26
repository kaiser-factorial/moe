# Family Comparison Plan — filling the analysis gaps + training the router

**Status:** draft 2026-06-23. Companion to `report/adapter_registry.md`
(adapter families) and `report/SUMMARY.md` (Family-A results).

## 0. The scientific question

All Family-A results showed *indirect* routing change: the router was frozen,
so re-routing could only happen because the LoRA moved the hidden states the
router reads ("re-weighting, not re-routing"). Two things were never separated
from that:

1. **Does training the routed experts themselves change the routing picture?**
   Families B and C actually adapted the 128 routed experts (A did not). The
   router is *still* frozen in B/C, so any routing-index shift is still
   indirect — BUT the experts' *functions* changed, and deeper-layer hidden
   states change cumulatively, which could amplify indirect re-routing. This is
   a clean, cheap test of how rigid the routing map is when the experts are no
   longer frozen.
2. **What happens if we re-route on purpose?** Directly training the gate is the
   only intervention that can move *selection* rather than just *inputs*. If
   capability fails to follow direct re-routing, that's strong confirmation of
   "concentration = familiarity, not competence."

## 1. Current coverage

| Analysis | A (attn+mamba+shared) | B (+routed) | C (everything) | D (router-trained) |
|---|:--:|:--:|:--:|:--:|
| Routing capture + divergence vs base | ✅ | ❌ → **WS1** | ❌ → **WS1** | ❌ → **WS2** |
| Steering / control vectors | ✅ pilot | optional | optional | optional |
| CoT capability pass-rate | ❌ | ✅ `expert_anal_lora` | ❌ | ❌ |

`D` = a new router-trained adapter produced by WS2.

---

## Workstream 1 — Divergence on Families B & C (cheap, high-information)

Reuses `scripts/capture_routing.py` + `scripts/divergence_analysis.py`
**unchanged** — only the adapter path changes. The router hook
(`mixer.gate`, `NemotronHTopkRouter`) is identical across families.

### Adapters
- **Family B**: `kagglehub.model_download("corinakaiser/lora3-checkpt800-1188/pyTorch/v3")`
  → adapter at `…/1/lora-adapter3/checkpoint-1188` (note the subfolder).
  (`nemotron-lora-adapter2/v2` is the same recipe — only run if we suspect the
  two diverged in training; otherwise skip.)
- **Family C**: `kagglehub.model_download("corinakaiser/nemo-lora-4-all/pyTorch/v0")`
  → adapter at `…/1/`.

`--adapter-path` feeds `PeftModel.from_pretrained`, which takes a **local dir**,
so `model_download` first, then pass the resolved path (not the Kaggle slug).

### Runs (same 111-problem base set as Phase 2)
```bash
# on the pod, base model already cached on volume gdqj7o63ik
python scripts/capture_routing.py --model-path "$BASE" \
   --adapter-path "$FAMILY_B_DIR" --out outputs/logs/lora_B/
python scripts/capture_routing.py --model-path "$BASE" \
   --adapter-path "$FAMILY_C_DIR" --out outputs/logs/lora_C/
# then, locally / cheap:
python scripts/divergence_analysis.py  # point at lora_B, lora_C vs base
```

### Read-outs (what each result means)
- **JSD(base↔B) and JSD(base↔C) vs the Family-A anchor (0.0537, null 0.110).**
  - If B/C ≈ A and still ≪ null → routing map is rigid even with experts
    trainable: indirect re-weighting dominates regardless of intervention scope.
    Strengthens the SUMMARY thesis.
  - If B/C ≫ A (toward/past null) → training experts unlocks real re-routing;
    bounds how much of A's story was a frozen-expert artifact.
- **Specialist survival (the 25/25 test) and dead-expert count.** Did training
  experts create dead experts or new specialists? A found neither.
- **Caveat to write up:** routing-index divergence *understates* B/C behavioral
  change, because expert *functions* changed even where indices didn't. Consider
  a secondary metric: per-expert output-norm shift on shared tokens. (Optional;
  flag, don't block.)

### Cost
~2–3 GPU-h per adapter capture (eager per-token decode through 23 MoE layers is
the bottleneck — `RUNLOG.md` L101). 2 adapters ≈ 4–6 h on H100 HBM3
(~$3.29/h) ≈ **$15–20**. Divergence analysis is CPU/local, free.

---

## Workstream 2 — Train the router directly (Family D)

### Step 0 — confirm the target parameter names (do FIRST)
Run the existing probe and dump the gate's parameters:
```bash
python scripts/probe_model.py   # already locates mixer.gate / out_features==128
# add a one-liner: for n,p in mixer.gate.named_parameters(): print(n, tuple(p.shape))
```
Expectation (DeepSeek-V3 style): a gating weight `[128, hidden]` and likely an
`e_score_correction_bias` `[128]`. Whether it's an `nn.Linear` or a raw
`nn.Parameter` decides the training mechanism.

### Step 1 — choose the mechanism
- **Preferred: `modules_to_save=["gate"]`** (full-finetune the router; it's tiny
  — 128×hidden per MoE layer ×23 layers). Clean, no PEFT-parameter fiddliness.
- Alt: PEFT `target_parameters=["gate.weight"]` (the config field already
  exists) if we want LoRA-style low-rank on the gate.
- Either way, **freeze everything else** to isolate the router's contribution.

### Step 2 — handle the two failure modes
- **Discrete top-k:** gradient reaches gate weights only for *selected* experts,
  so selection flips slowly. Train long enough / warm up LR; expect gradual
  re-routing, not instant.
- **Load balancing / collapse:** DeepSeek routing balances via the
  `e_score_correction_bias` (aux-loss-FREE, updated by its own rule, not grad).
  Options: (a) leave the bias on its native update and only train the weight;
  (b) add a standard load-balance auxiliary loss. Monitor per-expert load each
  step; abort on collapse (one expert >X% mass).

### Step 3 — data & eval
- Train on the same symbolic-reasoning set used for the other adapters (keeps
  the comparison honest).
- After training: run **WS1 divergence** on D (now you should see *direct*
  re-routing — the discriminating result) **and** the CoT pass-rate eval.
- **Decisive read-out:** if D re-routes strongly (high JSD, new specialists) but
  capability does **not** improve over base/A → confirms routing concentration
  is familiarity, not competence. If capability *does* climb with re-routing →
  partially falsifies it. Either is a headline result.

### Cost
Training run (router only, tiny param count, but full forward through 30B):
estimate **1 short epoch ≈ 2–4 GPU-h**, plus a WS1 capture (~2–3 h) and CoT eval
(~1 h). ≈ **$20–30**. Bigger unknown than WS1; timebox and checkpoint.

---

## 3. Operational notes (from prior runs — see project memory / RUNLOG)
- **Env recipe v2:** runpod/pytorch 2.8.0 image ships a *dev* torch with
  ABI-incompatible mamba-ssm/causal-conv1d wheels. Fix: reinstall
  `torch==2.8.0` (cu128, **with** deps), uninstall dev torchvision/torchaudio,
  then mamba-ssm + causal-conv1d `--no-deps --no-build-isolation`.
- `capture_routing.py` needs the `NemotronHBlock` KV-cache plumbing patch
  (already in the script); `patch_ptxas()` is neutered for H100.
- **RunPod:** never put the account API key in pod env (secret-scanner kills the
  pod) — ssh it into a root-only file. Volume `gdqj7o63ik` (EU-FR-1) pins the DC
  and already holds the base model + HF cache. H100 HBM3 ~$3.29/h usually
  available; EU-FR-1 A100s often not.
- **Also pending (separate):** runtime-confirm the cached-gen bug on 4B/9B-v2
  per `report/upstream_bug_runtime_plan.md` — don't let it block WS1.

## 4. Suggested order
1. **WS1 on B and C** (cheapest, reuses everything, directly tests rigidity).
2. **WS2 Step 0** (probe gate params) — near-free, unblocks everything.
3. **WS2 full** (router training → Family D → divergence + capability).
4. (Optional) capability eval on A & C and steering on B/C/D to complete the
   matrix.

## 5. Open decisions
- Family B: run `lora3` only, or also `adapter2/v2` to check they're identical?
- Router training: full-finetune gate (`modules_to_save`) vs LoRA the gate
  (`target_parameters`)?
- Load balancing: native bias only, or add an explicit aux loss?
- Budget cap for this round (WS1+WS2 ≈ $35–50 total at H100 rates)?
