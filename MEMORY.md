---
name: nemoh-moe-routing-project
description: "NemoH MoE expert-routing study — Phases 1-2 + Phase-3 pilot DONE; live front is WS1/WS2 (Family D router training); last pod run was a FALSE GREEN (see ws2_probe_log.md); key infra facts for resuming"
metadata: 
  node_type: memory
  type: project
  originSessionId: 567a73e5-289f-410b-b846-d3cab6975e1c
---

Corina's MoE routing research on Nemotron-3-Nano-30B-A3B (SPEC.md in repo;
`report/SUMMARY.md` is authoritative where docs conflict).

**Done (as of 2026-06-26):**
- **Phase 1** (2026-06-11): 111-problem dataset, per-token top-6 routing at all
  23 MoE layers. Two-tier specialization (soft bulk ~2× uniform + 26 near-pure
  specialists), semantic not surface, L17 = nexus, difficulty invisible.
- **Phase 1 extended**: 9 offline analyses (`report/phase1_extended.md`).
- **Phase 2** (2026-06-11/12): Family-A LoRA **re-weights, never re-routes**
  (JSD 0.054 < 0.110 null; pattern r=0.93; 25/25 specialists survive). Cause:
  adapter trains attn+Mamba+shared only — router/routed experts frozen.
- **Phase-3 pilot: GATE PASSED** — steering vectors move specialist occupancy
  ~50× the LoRA; all 16 vectors banked (`outputs/analysis/steering/vectors.npz`).

**Live front — family comparison (`report/family_comparison_plan.md`):**
- WS1 (Families B & C divergence capture): launched 2026-06-23, no results in tree.
- WS2 → Family D (router-only, 23 `gate.weight`, 7.9M params): **does not exist
  yet**. Stuck at probe stage. ⚠️ The 2026-06-26 pod sweep reported "8/8 STABLE"
  but ALL 8 probes crashed at step 0 (`element 0 of tensors does not require
  grad`) — launcher scored STABLE on absence-of-COLLAPSE-file only. Full story +
  2026-07-31 fixes: `outputs/analysis/ws2_probe_log.md`. The probe grid must be
  RE-RUN. Kaggle Run 4 remains valid: collapse is seed/data-triggered (seed=42
  tips at step ~25; seed=123 oscillates and survives). Select on `max_load`,
  NOT `lm`.
- Pod `xe0nbeg1wsj6pg` (H200) was left up 2026-06-26 01:44 UTC — verify killed.

**Adapter identity (from `report/adapter_registry.md`):** Phase 2/3 routing
results = Family A; `expert_anal_lora.md` 6/14 CoT = Family B. Never cite as one
model.

Key infra facts:
- RunPod network volume `gdqj7o63ik` (EU-FR-1, /workspace) persists across pods:
  model cache (~59GB, HF_HOME=/workspace/hf), scripts, logs. Stop/start wipes
  container disk. New pod = new IP/port only.
- `capture_routing.py:apply_nemotron_patches()` is MANDATORY for generation —
  NVIDIA's modeling code ships with cached generation broken (5 bugs, worst:
  attention layers never receive the KV cache). Without patches: 2 tok/s +
  prompt amnesia. Details in RUNLOG.md + `report/errors_postmortem.md`.
- Env pins: transformers==4.57.3 (5.x incompatible — and opbdh's bootstrap
  installs 5.8.1 first, so VERIFY the pin took; train_router.py now asserts it),
  torch 2.8.0+cu128; install causal-conv1d and mamba-ssm with
  `--no-deps --no-build-isolation` + einops.
- NEVER put RUNPOD_API_KEY in pod create-env (secret scanner kills the pod);
  write /root/.podenv post-boot — self-stop/delete silently no-ops without it.
- Git: deploy key in my sandbox `~/.ssh/id_ed25519` pushes to
  git@github.com:kaiser-factorial/moe.git.
- Gotcha that bit 5×: remote `pkill -f`/`pgrep -f` patterns match their own
  ssh/bash command line — use bracket patterns AND separate ssh calls.
