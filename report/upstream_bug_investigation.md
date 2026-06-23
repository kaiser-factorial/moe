# Investigation: Does the cached-generation bug affect other Nemotron-H models?

**Question (from Corina):** the `modeling_nemotron_h.py` cached-generation bug
we patched on Nemotron-3-Nano-30B-A3B — is it also present in newer / other
Nemotron versions?

**Short answer:** Yes — by source inspection, the central defect is present
**verbatim** across the entire `nemotron_h` text-model family, from the
Aug 2025 9B-v2 through the Mar 2026 4B. This was checked for free (static
code inspection, no GPU). Runtime confirmation on each model is the remaining
cheap step.

---

## What was checked

The bug that actually corrupts output (Error 19 in our postmortem) is in
`NemotronHBlock.forward`: the attention mixer is invoked **without the KV
cache**, so all attention layers store nothing at prefill and attend to a
single token during decode. The offending lines in our 30B-A3B were:

```python
elif self.block_type == "attention":
    hidden_states = self.mixer(
        hidden_states, cache_position=cache_position   # <-- no past_key_value!
    )
    hidden_states = hidden_states[0]
```

We fetched the `modeling_nemotron_h.py` shipped with two other family members
and grepped the same site.

### Result: byte-identical in all three

| Model | Released | Bug site | Status |
|-------|----------|----------|--------|
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (ours) | Dec 2025 | block forward, attention branch | **present** (patched) |
| NVIDIA-Nemotron-Nano-9B-v2 | Aug 2025 | lines 783–785 | **present, identical** |
| NVIDIA-Nemotron-3-Nano-4B-BF16 | Mar 2026 | lines 782–784 | **present, identical** |

In both the 9B-v2 and the 4B, the attention branch reads exactly:

```python
elif self.block_type == "attention":
    hidden_states = self.mixer(
        hidden_states, cache_position=cache_position
    )
    hidden_states = hidden_states[0]
```

No `past_key_value` / cache argument — the same omission. The supporting
defect (`cache_params.ssm_states.device` called on a Python list, our Error
15.4) is also present in the newer files (line 566 of the 4B's module).

## Scope of the family (all `nemotron_h`, all `custom_code`)

The same architecture string and custom modeling file ship with, at least:
9B-v2 (+ Base, FP8, NVFP4, Japanese), 12B-v2 (+ Base, VL variants), 4B
(+ FP8, GGUF), 30B-A3B (+ Base, FP8, NVFP4), and the Omni-30B-A3B reasoning
multimodal models. Because they share `modeling_nemotron_h.py`, they almost
certainly share the defect. (The VL/Omni variants wrap a different top-level
class but reuse the same hybrid block — worth a separate check.)

## Why does a bug this severe persist across releases?

Hypothesis, worth stating in any upstream report: **almost nobody runs these
models through `transformers.generate()`.** NVIDIA's model cards steer users
to vLLM and TRT-LLM, which have their own, independent attention/cache
implementations and never touch `modeling_nemotron_h.py`. The HF custom-code
path is the fallback for research/inspection — exactly our use case — so the
bug survives in a code path the mainstream deployment stack bypasses. This
also matches our Session-2 finding that *cacheless* generation (the silent
default before our cache-plumbing fix) re-runs full prefill each step and
stays coherent, just slow, so casual HF testing wouldn't surface it either.

## What this does and doesn't establish

- **Established (free):** the source code containing the bug is identical
  across the family; there is no version in which the attention branch was
  fixed.
- **Not yet established (needs a GPU):** that each model exhibits the
  behavioral failure at runtime, and that no `transformers` version since
  has added a native (non-custom-code) `nemotron_h` implementation that
  routes around it. Our patch was validated behaviorally on the 30B only.

## Cheap next steps (when budget allows)

1. **Runtime confirmation, ~$2 and 20 min on the smallest model.** The 4B
   fits on a cheap GPU. Run `state_probe.py` (Beatrix/Marco prompt): if the
   unpatched model guesses the parrot's name instead of quoting it, the bug
   is live; apply `apply_nemotron_patches()` and confirm the fix. The 4B is
   the natural target — cheapest, newest, same code.
2. **Check `transformers` mainline.** See whether a recent transformers
   release ships a built-in `NemotronHForCausalLM` (not `custom_code`); if so,
   compare its block-forward and test whether `trust_remote_code=False` avoids
   the bug entirely.
3. **Check the model repos' commit history / community tab** for an already-
   filed issue or a newer `modeling_nemotron_h.py` revision (the files carry a
   snapshot hash; ours was `cbd3fa9…`).
4. **If confirmed, file upstream.** A minimal repro (the probe prompt + the
   five-line diff) on the 9B-v2 community tab would be a clean, useful report
   — it affects every HF-`generate()` user of the family.

## Repro pointers (in this repo)

- `scripts/state_probe.py` — the one-shot diagnostic (empty KV after prefill).
- `scripts/capture_routing.py::apply_nemotron_patches()` — the runtime fix.
- `report/errors_postmortem.md` §1 — full description of the five-defect chain.
