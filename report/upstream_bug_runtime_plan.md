# Runtime Investigation Plan: confirm the Nemotron-H cached-generation bug across the family

**Prereq:** RunPod credits (~$10 covers everything below with margin).
**Goal:** turn the free source-inspection finding (bug present verbatim in
9B-v2 and 4B) into runtime-confirmed evidence, then file a clean upstream
report. See `upstream_bug_investigation.md` for the static findings this builds
on, and `errors_postmortem.md` §1 for the five-defect description.

**Headline hypothesis (H):** unpatched `transformers.generate(use_cache=True)`
on any `nemotron_h` model loses the prompt mid-context (attention KV never
stored), producing a measurable accuracy/recall collapse vs. the patched model.

---

## Budget & sizing

| Stage | Model | GPU | Est. time | Est. cost |
|-------|-------|-----|-----------|-----------|
| 0 | env build (one-time, cached on volume) | any | 15 min | ~$0.50 |
| 1 | 4B (smallest) — confirm bug + fix | 1× any 24GB+ (e.g. A40/4090) | 20 min | ~$0.50 |
| 2 | 9B-v2 — confirm bug + fix | 1× 40-48GB (A6000/A100) | 30 min | ~$1 |
| 3 | quantitative recall benchmark, both | same pod as 1/2 | 45 min | ~$1.50 |
| 4 | transformers-mainline check (no GPU needed) | local/sandbox | — | $0 |
| 5 | (optional) 30B re-confirm | 1× H100 80GB | 30 min | ~$1.50 |

**Hard stop: $10.** Stages 1–4 are the core; 5 is nice-to-have (we already
patched the 30B, so it's just belt-and-suspenders).

The 4B is the smartest first target: cheapest, newest (Mar 2026 — proves the
bug is still shipping), same code. If it reproduces there, the family claim is
all but settled.

---

## Stage 0 — Environment (reuse the proven recipe)

Use **env recipe v2** (in project memory / RUNLOG). On `runpod/pytorch:2.8.0`:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # WITH deps
pip uninstall -y torchvision torchaudio                                       # dev build breaks transformers import
pip install transformers==4.57.3 accelerate einops
pip install --no-deps --no-build-isolation causal-conv1d mamba-ssm
```

Put `HF_HOME=/workspace/hf` on the network volume so model downloads persist.
Account API key goes in a root-only file *after* boot, never in create-env
(secret-scanner kills the pod — Error 26). Bracket all pkill/pgrep patterns and
split kill/launch into separate ssh calls.

---

## Stage 1 — Behavioral confirmation on the 4B (the decisive test)

The diagnostic already exists: `scripts/state_probe.py` (the Beatrix/Marco
prompt — fact mid-prompt, question at the end). Run it twice, unpatched vs
patched, on `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`.

**Predicted result if H is true:**
- *Unpatched:* model guesses the parrot's name (e.g. "Polly"); cache dump
  shows attention layers with empty KV `(1, 0)`.
- *Patched* (`apply_nemotron_patches()`): KV populated `(1, n_heads, T, d)`;
  model quotes "Marco" verbatim.

This is a yes/no result in ~5 minutes of GPU. If it reproduces, H is confirmed
for the 4B.

To make `state_probe.py` toggle the patch, add a `--no-patch` flag that skips
the `apply_nemotron_patches(model)` call (one line). Run:

```bash
python scripts/state_probe.py --model-path nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --no-patch   # expect failure
python scripts/state_probe.py --model-path nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16              # expect "Marco"
```

## Stage 2 — Repeat on the 9B-v2

Same two runs on `nvidia/NVIDIA-Nemotron-Nano-9B-v2`. Confirms the bug spans
both the oldest (Aug 2025) and newest (Mar 2026) text releases, i.e. it's a
standing family-wide defect, not a one-version regression.

## Stage 3 — Quantitative recall benchmark (turns anecdote into a number)

A single probe is a vivid demo but n=1. Add a small, automatic recall test so
the upstream report has a statistic.

**`scripts/needle_recall.py` (to write, ~40 lines):** generate 25 synthetic
"needle-in-context" prompts — a unique fact (random name/number) placed at a
controlled depth in a 200–600 token filler context, with the question at the
end. Score exact-match recall, unpatched vs patched, on both models. Also
sweep needle depth (early/mid/late) to show the signature: unpatched recall
should collapse for early/mid needles and survive only for late ones (the
Mamba-conv-state tail), while patched recall stays high across depths.

**Deliverable:** a 2×2×3 table (model × patch × depth) of recall %. Expected
shape:

| | unpatched | patched |
|---|---|---|
| needle early | ~0% | ~high |
| needle mid | ~low | ~high |
| needle late | moderate | ~high |

That depth-dependent collapse is the fingerprint of "attention sees nothing,
only Mamba conv tail survives" — much stronger evidence than a single prompt.

## Stage 4 — Is mainline `transformers` affected? (no GPU)

Check whether a recent `transformers` ships a *native* (non-`custom_code`)
`NemotronHForCausalLM`. If it does:
- diff its `NemotronHBlock.forward` attention branch against the repo's
  `modeling_nemotron_h.py`;
- test whether loading with `trust_remote_code=False` (forcing the library
  implementation) avoids the bug entirely.

If mainline is fixed but the HF repos still ship the broken `custom_code`,
that's the cleanest possible upstream ask: "sync the repo's modeling file to
mainline." This stage is free — do it first, even before credits, if possible.

## Stage 5 (optional) — 30B re-confirm

We already patched and ran the 30B successfully, so this only matters if a
reviewer wants the probe on the exact model the study used. Cheap to add to a
Stage-1/2 pod if an H100 is already up.

---

## If confirmed: the upstream report

File on the **9B-v2 community/discussions tab** (highest traffic, oldest, most
likely to get NVIDIA eyes). Include:
1. One-paragraph symptom: HF `generate(use_cache=True)` loses prompt context;
   attention KV never populated.
2. Minimal repro: the `state_probe.py` prompt + the `--no-patch` vs patched
   outputs, plus the Stage-3 recall table.
3. Root cause: `NemotronHBlock.forward` calls the attention mixer without
   `past_key_value` (quote the 5 lines).
4. Minimal fix: the block-forward diff from `apply_nemotron_patches()` (and
   note the supporting cache-class defects).
5. Scope: same code in 4B / 9B-v2 / 30B-A3B (and likely 12B, Omni).

Optionally open a `transformers` issue too if Stage 4 shows mainline is also
affected.

---

## Pre-committed kill criteria

- If Stage 1 (4B) does **not** reproduce the failure unpatched — i.e. the
  model recalls the needle fine without our patch — **stop and re-examine.**
  That would mean the runtime path differs from the source we read (e.g.
  transformers is silently substituting its own attention), which is itself
  the finding; pivot Stage 4 to the front and don't spend on 9B/30B.
- If env build burns more than ~$2 / 45 min, fall back to the exact pinned
  image hash from the Phase-2 success (volume `gdqj7o63ik`, recipe v2).
- Respect the $10 hard stop; Stages 1–2 + 4 alone (~$2) already answer the
  scientific question.

---

## Ready-to-run checklist (paste-ready when credits land)

1. [ ] Stage 4 first if doable offline (free): check mainline transformers.
2. [ ] Spin smallest viable pod; build env (recipe v2); cache to volume.
3. [ ] Add `--no-patch` flag to `state_probe.py`; add `scripts/needle_recall.py`.
4. [ ] 4B: probe unpatched → patched (Stage 1).
5. [ ] 9B-v2: probe unpatched → patched (Stage 2).
6. [ ] Run `needle_recall.py` on both, both modes, 3 depths (Stage 3).
7. [ ] Save tables/figures to `outputs/analysis/upstream_bug/`; commit.
8. [ ] If confirmed, draft the 9B-v2 discussion post from the template above.
9. [ ] Tear down pod; verify volume holds results; note balance.
