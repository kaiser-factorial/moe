# What the symbolic LoRA actually trained (authoritative)

**Source:** `adapter_config.json` of
`brick-factorial/nemotron-lora-symbolic-reasoning` (the adapter used in
Phase 2/3). Fetched 2026-06-23. This settles "experts-run vs attention-run."

## target_modules (verbatim)

```
v_proj, q_proj, k_proj, o_proj          # attention (Q/K/V/O projections)
in_proj, out_proj                       # Mamba-2 mixer projections
shared_experts.down_proj, shared_experts.up_proj   # the SHARED expert MLP
```

Hyperparameters: LoRA rank r=32, alpha=64, dropout=0.05, bias none.

## What this means

| Component | Trained? |
|-----------|----------|
| Attention (q/k/v/o_proj) | **Yes** |
| Mamba-2 mixer (in_proj/out_proj) | **Yes** |
| Shared expert (always-on MLP, up/down_proj) | **Yes** |
| Routed experts (the 128 per MoE layer) | **No** |
| Router gate (`gate` / `NemotronHTopkRouter`) | **No** |

So this is an **attention + Mamba + shared-expert** adapter. It is **not** an
"experts" run in the sense of the 128 routed experts — those weights, and the
router that selects them, were frozen.

## Correction to the SPEC

SPEC.md §2 listed trainable layers as "Experts (up_proj/down_proj), Mamba
(in_proj/out_proj)." The adapter config shows the trained MLP is
`shared_experts.up_proj/down_proj` — the **shared** expert, not the routed
experts — and that attention (q/k/v/o_proj) was also trained (not mentioned in
the SPEC). The routed experts were never adapted.

## Why this is the linchpin of the Phase 2/3 interpretation

Because **neither the router nor the routed experts were trained**, every
routing change we measured in Phase 2 is necessarily **indirect**: the adapter
altered hidden states (via attention, Mamba, and the shared pathway), and the
*frozen* router then selected different experts in response. This is exactly
why Phase 2 found "re-weighting, not re-routing," why the 25/25 specialists
survived, and why subtype structure was preserved (§A2): the map is baked into
frozen weights; fine-tuning only changed the inputs the map reads.

It is also the premise Phase 3 rests on: if routing is steerable only
indirectly through the residual stream, then control vectors injected into the
residual stream are the right instrument — and indeed the pilot moved expert
occupancy ~50× more than the LoRA did.

**One caveat for honesty:** this describes the adapter at the HF path we
uploaded. The memory notes the original checkpoint was
`~/Downloads/nemotron-sub/lora-adapter3/checkpoint-1188`; this config's
`base_model_name_or_path` correctly points at the 30B-A3B BF16 base, and the
Phase-2 inference loaded this adapter, so it is the one our results describe.
If there were *other* training runs (e.g. a routed-experts variant) they are
not this adapter and were not analyzed.
