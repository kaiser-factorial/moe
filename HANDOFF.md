# HANDOFF — NemoH MoE Routing Project

Start here next session. Repo: `git@github.com:kaiser-factorial/moe.git`
(everything below is committed & pushed; clean tree).

## Status in one line

Phases 1, 2, and the Phase-3 **pilot** are COMPLETE; 9 offline analyses done and
consolidated. All remaining work needs a GPU (RunPod credits exhausted last
session). Nothing is mid-flight — safe to pick up cold.

## Read these first (in order)

1. `report/SUMMARY.md` — canonical synthesis of all findings. **Authoritative
   where anything conflicts.**
2. `report/DYFA.md` — per-question D-Y-F-A with a figure each (Q1a–Q3 + E1–E4).
3. `report/phase1_extended.md` — the 9 offline analyses (A, A2, A3, B, C, E, F,
   G, H, I) in detail.
4. `report/errors_postmortem.md` — the inference-stack repair (read before any
   GPU run; you WILL need the patches).

## The findings, compressed

- **Two-tier specialization**: soft generalist bulk (~2× uniform selectivity)
  + 26 near-pure specialists (perm p<0.002). Semantic (splits by subtype, not
  surface), temporally stationary, peaks mid-network.
- **Layer 17 is the nexus** — 4 independent signals (peak selectivity, sharpest
  social/ethical team 3.25×, peak Gini, the pure specialists).
- **LoRA re-weights, never re-routes** — confirmed cause: adapter trains
  attn+Mamba+shared-expert only, NOT routed experts/router (see
  `lora_target_modules.md`; the SPEC was wrong). Map preserved (pattern r=0.93,
  25/25 specialists survive); divergence small (JSD 0.054 < 0.110 null), graded
  by domain adjacency, concentrates load without collapse.
- **Phase-3 pilot GATE PASSED**: residual-stream control vectors move routing
  ~2× the LoRA and specialist occupancy ~50× the LoRA, coherent at |α|≤~1.

## TWO corrections — do not regress to the old wording

1. **Q1c (accuracy↔routing)**: metric-dependent. Survives category control with
   Phase-1's whole-generation aggregated metric (factual+reasoning pooled
   r=0.44, p=0.008) but NOT with per-token thinking-region (r=0.04). It is a
   *global* routing property, not a local confidence signal. Earlier "confidence
   signal" framing was downgraded.
2. **Cross-layer NMI**: raw 0.71 is mostly small-sample bias (null floor 0.60);
   real excess only ~0.12. "Coherent pathway" was downgraded to "weak but real";
   the firm finding is within-layer expert *teams*.

## Next actions (all GPU-gated) — ranked

1. **Phase-3 Stage-1 steering sweep** — highest value; pilot already passed.
   Vectors for all 16 sites are banked (`outputs/analysis/steering/vectors.npz`,
   local + on volume), so **no re-extraction needed**. Revised grid:
   α ∈ {±0.25, ±0.5, ±1} (|α|≥2 degenerate). Then Stage-2 causal accuracy test
   (does steering toward specialists move symbolic accuracy? H3: flat/drops =
   familiarity confirmed). Plan: `report/phase3_design.md`. Use a
   distinct-n-gram coherence detector, NOT mean-logprob (loops score better than
   reasoning).
2. **Confirm the cached-gen bug at runtime** on the 4B/9B-v2 — cheap (~$2),
   high external value (affects every HF-`generate()` user of the nemotron_h
   family; confirmed present in source). Full plan + paste-ready steps:
   `report/upstream_bug_runtime_plan.md`. If confirmed → file upstream.
3. **Larger gradable problem set** to settle the within-type accuracy question
   (Q1c is underpowered at n=16-19/category).

## GPU/RunPod survival kit (hard-won — read before launching)

- **Env recipe v2** (the runpod/pytorch:2.8.0 image ships a *dev* torch that
  breaks the mamba kernels): `pip install torch==2.8.0 --index-url
  https://download.pytorch.org/whl/cu128` (WITH deps), `pip uninstall
  torchvision torchaudio`, then `causal-conv1d` + `mamba-ssm`
  `--no-deps --no-build-isolation`, plus `einops`. Pin `transformers==4.57.3`.
- **`capture_routing.py::apply_nemotron_patches()` is MANDATORY** — without it
  generation runs cacheless (2 tok/s, prompt amnesia). Validate with
  `state_probe.py` (Beatrix/Marco prompt → must answer "Marco").
- **NEVER put the account RUNPOD_API_KEY in pod create-env** — secret scanner
  kills the pod in 3-11 min (looks like mystery deletion). Put it in a root-only
  file via ssh after boot.
- Network volume `gdqj7o63ik` (EU-FR-1) holds the HF cache (`/workspace/hf`),
  repo, and all phase2/3 results; it pins pods to EU-FR-1. Stop/start WIPES
  container disk (rebuild env) and changes the SSH port.
- `pkill`/`pgrep -f`: bracket the pattern AND use separate ssh calls (the
  command line self-matches — bit us 5×).
- Full detail in project memory + `RUNLOG.md`.

## Data inventory (all local, all CPU-analyzable)

- `outputs/logs/base/` + `outputs/logs/lora/` — 111 npz each (top-6 routing ×
  23 MoE layers × every token) + `results.jsonl` + `run_config.json`.
- `outputs/analysis/{base,lora,divergence,steering,extended}/` — all CSVs,
  jsons, figures.
- `data/problems.json` — the 111-problem dataset (seed 42, regenerable).
- LoRA adapter: `brick-factorial/nemotron-lora-symbolic-reasoning` (HF, public).

## Loose ends / caveats to remember

- `report/phase1_DYFA.md` is the OLD per-question DYFA (Q1-Q2 only, no figures);
  `report/DYFA.md` is the new complete one. Keep DYFA.md; phase1_DYFA.md is
  superseded but left for provenance.
- social/ethical accuracy is unscored (prompts lacked the boxed-answer suffix) —
  fix in dataset if you want that category gradable.
- Community membership boundaries are unstable at late layers; trust frozen-set
  enrichment over Jaccard.
