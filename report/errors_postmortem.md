# Errors & Fixes: A Postmortem

**Project:** MoE expert-routing study on Nemotron-3-Nano-30B-A3B (NemoH)
**Scope:** Phases 1–3, three working sessions (2026-06-10 → 06-12)
**Companion:** `RUNLOG.md` (chronological, with raw error strings)

This document catalogs the 27 distinct errors encountered while building and
running the routing-capture pipeline, grouped by root cause. The throughline:
very little of the difficulty was in the science. The hard part was a model
whose cached-generation path had apparently never been exercised in
Hugging Face transformers, wrapped in a GPU dependency stack that actively
fought itself, run on cloud infrastructure with several non-obvious traps.

---

## 1. The big one: cached generation is broken in the shipped model code

**Errors 11, 13, 14, 15, 19 — one root cause, many faces.**

Symptoms, in the order they appeared and confused us:
- generation crawled at ~2 tok/s with GPU utilization ~46% (error 11),
- an out-of-memory crash partway through the first batch (error 13),
- a tensor-shape mismatch when reassembling per-token logs, because the
  per-step routing calls had *growing* row counts (error 14),
- and, after those were bridged, degenerate output: repetition loops and
  fluent rambling about a question the model couldn't actually see (error 17).

The underlying cause was a chain of five defects in
`modeling_nemotron_h.py` that interact so that `model.generate()` silently
runs **without a usable cache**:

1. `prepare_inputs_for_generation` returns the cache under the key
   `past_key_values`, but `forward()` only accepts it as `cache_params`. The
   cache fell into `**kwargs` and was dropped — every step.
2. Once bridged, `generate()` round-trips the cache as
   `model_kwargs["cache_params"]`, but `prepare_inputs_for_generation` only
   reads its `past_key_values` parameter, so a fresh empty cache was built at
   every decode step.
3. The cache class (`HybridMambaAttentionDynamicCache`) lacks the
   `conv_kernel_size` attribute its own mixers read.
4. Its `update_conv_state` / `update_ssm_state` methods (and `torch_forward`)
   call `.device` on a Python *list*, raising `AttributeError` on first real
   use.
5. **The actual villain:** `NemotronHBlock.forward` never passes the cache to
   the attention mixers at all — `self.mixer(hidden_states, cache_position=…)`
   with no `past_key_value`. All six attention layers therefore stored no KV
   at prefill and attended to a single token during decode.

### Why it was so hard to see

The failure is graceful, which is the worst kind. With the cache broken, the
model could still see the *tail* of its prompt through the Mamba conv states,
so it produced plausible-looking reasoning — about a question it had largely
forgotten. It didn't crash or emit obvious garbage from token one; it decayed
into loops over hundreds of tokens.

The decisive diagnostic (`state_probe.py`) used a prompt with a fact in the
middle ("a parrot called Marco") and the question at the end. The broken model
knew it was being asked for a name and guessed "Polly"; dumping the cache
showed **all six attention layers holding empty KV `(1, 0)`** after prefill.
After the patch: KV `(1, 2, 120, 128)` on every attention layer, and the model
quoted the prompt verbatim and answered "Marco."

A final twist explains why nobody caught this upstream: the *original*,
accidental cacheless mode re-runs the full prefill every step, so attention
layers do see the whole sequence and output is coherent — just quadratically
slow. Any test that tolerated the slowness would pass. The bug only manifests
once caching is actually on.

**Fix:** all five defects are patched at runtime in
`apply_nemotron_patches()` (`scripts/capture_routing.py`); no model files are
modified. Post-patch throughput went from 2 tok/s to 8–16 tok/s batched, with
coherent long-form generation.

A related wrinkle: `generate()` validates kwargs by *inspecting the signature*
of `prepare_inputs_for_generation`, so a naive `(*args, **kwargs)` wrapper
broke validation (error 16). Fixed with `functools.wraps`.

---

## 2. Dependency and environment hell

The GPU stack repeatedly sabotaged itself. These are the errors most likely to
bite anyone reproducing the work.

- **Error 6 — no slow path.** `modeling_nemotron_h.py` hard-raises
  `ImportError: mamba-ssm is required`; there is no pure-PyTorch fallback, so
  `mamba-ssm` + `causal-conv1d` must be installed.
- **Error 7 — mamba-ssm silently upgrades torch.** Installing `mamba-ssm`
  pulled torch `2.8.0+cu128 → 2.12.0+cu130`, which broke CUDA
  (`torch.cuda.is_available() == False`) and produced an ABI-mismatch
  `undefined symbol` from the prebuilt conv kernel. Fix: reinstall the pinned
  torch, then build the kernels from source with
  `--no-deps --no-build-isolation`.
- **Error 9 — transformers 5.x is incompatible.** `TypeError: 'NoneType' is
  not subscriptable` at `cache_position[-1]`. The model card specifies 4.57.3;
  pinning it fixed this.
- **Error 10 — our own Blackwell workaround backfired on Hopper.** The
  `shadow_mamba()` stub lacked `__version__`, which transformers 4.57.3's
  `is_mamba_2_ssm_available()` reads. Fix: only shadow on Blackwell (SM ≥ 10)
  and give the stub a version string.
- **Error 22 — the base image now ships a *dev* torch.** `runpod/pytorch:2.8.0`
  carries `2.8.0.dev20250319`, whose ABI the prebuilt mamba/conv wheels don't
  match (`undefined symbol …SetDeviceEab`) — even when fetched under
  `--no-binary`. Fix: install release `torch==2.8.0` from the cu128 index.
- **Error 23 — but install torch *with* its deps.** Using `--no-deps` left a
  mismatched NCCL (`ncclCommWindowRegister` missing).
- **Error 24 — torchvision breaks the import.** The image's dev-built
  torchvision raised `operator torchvision::nms does not exist` at transformers
  import. Fix: `pip uninstall torchvision torchaudio` (not needed for a text
  model).
- **Error 25 — `patch_ptxas()` crashes on release-torch triton.** The Blackwell
  ptxas redirect demanded a `get_ptxas(arch)` signature that release triton
  doesn't have. H100 never needs it; neutered on Hopper pods, kept for
  Blackwell.

**Net recipe (env v2):** install release `torch==2.8.0` from the cu128 index
*with* deps; `pip uninstall torchvision torchaudio`; then `mamba-ssm` and
`causal-conv1d` with `--no-deps --no-build-isolation`; also install `einops`
(an undeclared mamba-ssm runtime dependency). Pin `transformers==4.57.3`.

---

## 3. The fused-kernel layout shim

**Errors 15.6 and 18.** The fused decode kernel `causal_conv1d_update` rejected
the modeling code's tensor layout (`weight must have shape (dim, width)`): the
model passes the hidden state as `(B, 1, dim)` while the kernel expects
`(B, dim, 1)`. Initially we sidestepped this by forcing the pure-PyTorch Mamba
path — but that path corrupts prefill SSM state on longer prompts (the model
only "sees" the prompt tail), so it was *not* a safe workaround. The correct
fix was to keep the fast path and add a transpose-in/transpose-out shim around
`causal_conv1d_update`.

---

## 4. Infrastructure & RunPod operations

- **Error 26 — "the pod assassin."** Pods that carried the account
  `RUNPOD_API_KEY` in their create-time environment were terminated by
  RunPod's secret scanner within 3–11 minutes — and the deletions were
  audit-logged *as the account owner*, so they looked exactly like a human
  deleting pods from the console. (Corina was, understandably, baffled.)
  Evidence that exonerated everyone: only key-carrying pods died, and the
  key's `lastUsed` predated the deletions. Fix: write the account key to a
  root-only file over SSH *after* boot, never into the create env. Note also
  that the auto-injected pod-scoped key cannot stop or delete its own pod
  (REST 403).
- **Pods dropping mid-run.** Twice the pod became unreachable mid-task. No work
  was lost because the model cache, scripts, and logs all live on a network
  volume that survives pod restarts — only the SSH port changes. This is the
  single most valuable operational habit from the project.
- **Stop/start wipes the container disk.** The pip environment and `/root`
  files are lost on stop→start (volume survives); the env must be rebuilt and
  the SSH port re-read each time.
- **Datacenter pinning.** A network volume pins its pods to one datacenter;
  EU-FR-1 reliably stocks H100 HBM3 but rarely A100.

---

## 5. The opbdh detour (Phase 2)

**Error 27.** A brief attempt to use the `opbdh` RunPod launcher cost more than
it saved: hard-coded `minVCPUPerGPU: 8, minRAMPerGPU: 64` made every create
fail with "no pods with required specifications" (patched to 4/24); a global
config silently pinned runs to the wrong datacenter (US-MD-1); and its monitor
process can't survive in the sandbox (bwrap `--die-with-parent` kills
background processes between calls). Resolution: drive the RunPod REST API
directly, which worked cleanly.

---

## 6. The running gag: pkill/pgrep self-match

**Errors 8, 12, 20, and two more in Session 3 — at least five victims.**

Every time a remote command did `pkill -f "capture_routing"` (or `pgrep`), the
kill command's *own* command line contained the search string, so it matched
and killed its own shell (exit 255). The bracket trick (`[c]apture_routing`)
only half-helps, because the launch command in the same line legitimately
contains the string too. Error 20 was the most poetic: an auto-analysis watcher
hung forever polling for a process whose name its own command line matched —
the curse claimed its author. The durable fix is two things together: bracket
patterns *and* separating the kill and launch into distinct SSH invocations.

---

## 7. Data and minor issues

- **Error 21 — the SPEC's adapter repo was empty.** The Hugging Face repo named
  in the SPEC contained only `.gitattributes`. The real LoRA adapter was found
  locally (`checkpoint-1188`, confirmed against the prior inference log) and
  uploaded to the SPEC path with corrected metadata before Phase 2 could run.
- **Errors 1–5 — routine setup friction.** Missing `scipy`; a matplotlib rename
  (`boxplot(labels=)` → `tick_labels=`); a stale `.git/index.lock` the mount
  wouldn't let us delete (resolved via the file-delete permission); PEP 668
  blocking `pip` (`--break-system-packages`); and a PATH-ordering bug in a
  one-liner where the `hf` CLI wasn't yet on PATH.
- **Data integrity.** `routing_symb-medi-17.npz` synced as 0 bytes after one
  session and was restored from the volume original; the base set is 111/111
  intact.

---

## Full catalog

| # | Area | One-line summary | Status |
|---|------|------------------|--------|
| 1 | Setup | `scipy` missing | fixed |
| 2 | Setup | matplotlib `boxplot(labels=)` renamed | fixed |
| 3 | Git | stale `.git/index.lock` undeletable on mount | fixed |
| 4 | Setup | PEP 668 blocks pip | fixed (`--break-system-packages`) |
| 5 | Setup | `hf` CLI not yet on PATH (ordering) | fixed |
| 6 | Env | model hard-requires mamba-ssm, no fallback | fixed |
| 7 | Env | mamba-ssm silently upgrades torch → CUDA broken | fixed |
| 8 | Ops | pkill self-match killed shell | fixed |
| 9 | Env | transformers 5.x incompatible w/ model code | pinned 4.57.3 |
| 10 | Env | our shadow_mamba stub broke version check | fixed |
| 11 | Perf | cacheless generation → ~2 tok/s | root cause = 15 |
| 12 | Ops | pkill self-match (again) | fixed |
| 13 | Mem | OOM during batch | symptom of 15 |
| 14 | Bug | growing per-step routing row counts | symptom of 15 |
| 15 | Bug | **5-defect broken-cache chain** | patched at runtime |
| 16 | Bug | generate() inspects PIFG signature; wrapper broke it | `functools.wraps` |
| 17 | Bug | degenerate long output | root cause = 19 |
| 18 | Bug | torch Mamba path corrupts long prefill | kept fast path + shim |
| 19 | Bug | **attention layers never receive KV cache** | patched at runtime |
| 20 | Ops | watcher self-match hang | fixed |
| 21 | Data | SPEC adapter repo empty | re-uploaded real adapter |
| 22 | Env | base image ships dev torch; ABI mismatch | install release torch |
| 23 | Env | torch `--no-deps` breaks NCCL | install with deps |
| 24 | Env | dev torchvision breaks transformers import | uninstall it |
| 25 | Env | patch_ptxas crashes on release triton | neutered on Hopper |
| 26 | Ops | **secret scanner kills key-carrying pods** | key in post-boot file |
| 27 | Ops | opbdh launcher misconfigured | drive REST API directly |

---

## Lessons that generalized

1. **Put the cache cost on the volume, not the container.** Surviving two
   mid-run pod deaths with zero lost work was entirely due to keeping the model
   cache, scripts, and logs on a network volume.
2. **Pin the whole stack, build kernels from source.** Install release torch
   with its deps, remove torchvision, build mamba/conv with
   `--no-deps --no-build-isolation`, pin transformers 4.57.3.
3. **Never ship a secret in a pod's create env.** Write it to a root-only file
   after boot.
4. **Probe state, don't guess.** The cache bug was invisible from output alone;
   dumping the actual KV/SSM state after prefill found it in one shot.
5. **Bracket your pkill patterns *and* split the SSH calls.** Five separate
   incidents say so.
