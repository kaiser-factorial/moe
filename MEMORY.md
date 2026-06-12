---
name: nemoh-moe-routing-project
description: "NemoH MoE expert-routing study — Phase 1 done 2026-06-11, Phase 2 (LoRA comparison) pending; key infra facts for resuming"
metadata: 
  node_type: memory
  type: project
  originSessionId: 567a73e5-289f-410b-b846-d3cab6975e1c
---

Corina's MoE routing research on Nemotron-3-Nano-30B-A3B (SPEC.md in repo).
**Phase 1 complete (2026-06-11)**: 111-problem dataset, per-token top-6
routing captured at all 23 MoE layers, analysis + report in
`report/phase1_report.md`. Findings: two-tier specialization (soft bulk ~2×
uniform + ~26 near-pure specialists, peak mid-network L17-20), difficulty
invisible to routing, concentration↔correctness r=0.31.

**Phase 2 next**: same dataset through LoRA adapter
(`brick-factorial/nemotron-lora-symbolic-reasoning`) via
`capture_routing.py --adapter-path`, then divergence analysis (Jaccard/KL,
SPEC §4 Phase 2). Raw base npz logs are local in `outputs/logs/base/`.

Key infra facts:
- RunPod network volume `/workspace` persists across pods: model cache
  (~59GB, HF_HOME=/workspace/hf), scripts, logs. New pod = new IP/port only.
- `capture_routing.py:apply_nemotron_patches()` is MANDATORY — NVIDIA's
  modeling code ships with cached generation broken (5 bugs, worst:
  attention layers never receive the KV cache). Without patches: 2 tok/s +
  prompt amnesia. Details in RUNLOG.md (20 numbered errors).
- Env pins: transformers==4.57.3, torch 2.8.0+cu128; install causal-conv1d
  and mamba-ssm with `--no-deps --no-build-isolation` + einops (their
  resolver upgrades torch to a broken cu130 build otherwise).
- Git: deploy key in my sandbox `~/.ssh/id_ed25519` pushes to
  git@github.com:kaiser-factorial/moe.git.
- Gotcha that bit 4×: remote `pkill -f`/`pgrep -f` patterns match their own
  ssh/bash command line — use bracket patterns AND separate ssh calls.
