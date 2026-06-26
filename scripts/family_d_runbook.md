# Family D Training — Runbook (Kaggle RTX Pro 6000, BF16)

Notebook: `scripts/train_router_kaggle_rtxpro.ipynb`. Produces **Family D** — the
router-only adapter (23 `gate.weight` tensors trained, everything else frozen),
BF16, no quantization confound. Probe first to pick LR, then the full run.

---

## Phase 1 — Set up the Family D kernel
1. New notebook → **File → Import Notebook** → `train_router_kaggle_rtxpro.ipynb`.
2. **Settings:** Accelerator = **RTX Pro 6000** (your comp GPU); Internet = **Off**.
3. **Add inputs:**
   - base model,
   - your Wonderland dataset (`corinakaiser/master-cot-train3`),
   - `ryanholbrook/nvidia-utility-script` **as a Utility Script** (Add Input →
     Utility Scripts). Kaggle mounts its prebuilt Blackwell stack at
     `/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script` — works
     offline, no online build step. Cell 1 just `sys.path.insert`s it.
4. **Check paths:** `MODEL_PATH` + `DATA_PATH` (cell 3) match the input panel;
   `DEPS_DIR` (cell 1) is the utility-script mount above (default already set).

## Phase 2 — Probe run (pick the LR)
Set `MODE="probe"` (cell 3). Run cells top-to-bottom and watch the checkpoints:
- **Cell 1:** prints `torch … capability: (12, 0)` and the RTX Pro 6000 name.
- **Cell 2:** `ptxas-blackwell patched` (and `mamba_ssm shadowed` only if you set
  `SHADOW_MAMBA=True`).
- **Cell 4:** model loads in BF16 (~60 GB).
- **Cell 5:** `trainable router params: 7,913,472 across 23 gates`.
- **Cell 7:** `examples: 9500`.
- **Cell 8:** training log — `lm` should **fall**, `max_load` should stay **under
  0.08** (the red line in the cell-9 plot).

Run it **three times**, changing only `LR` (cell 3): `1e-4` → `3e-5` → `1e-5`.
**Pick the highest LR whose loss falls and stays balanced.**

Troubleshooting:
- Blackwell init is baked into cell 2 (shadow mamba3 + ptxas patch) and cell 4
  (force slow path) — ported from your `nemotron-train-4-0-all`. If cell 2 errors
  on the ptxas copy, the utility-script input isn't attached / `DEPS_DIR` is wrong.
- OOM → lower `MAX_LEN` (cell 3) 4096 → 2048 → 1024.

## Phase 3 — Full run (= Family D)
Set `MODE="full"` and `LR=<chosen>` (cell 3); re-run all cells. This is 1 epoch
(~1,188 steps). Your original LoRA run estimated **~12 h** for this — and step
time is dominated by the frozen 30B forward + forced-slow-path Mamba (same for
router-only), so expect **~12 h, right at Kaggle's session limit**. Plan for a
resume (below); it may take two sessions.
- Checkpoints save every 200 steps (`save_total_limit=3`).
- **If it hits Kaggle's session limit before finishing:** re-run the notebook and
  change cell 8's `trainer.train()` → `trainer.train(resume_from_checkpoint=True)`
  to continue from the last checkpoint.
- Output: **`router_state_full_lr<LR>.pt`** in `/kaggle/working/familyD/` — these
  23 gate tensors *are* Family D. *Save Version* (or save as a dataset) so they
  persist; download or note the path.

## Phase 4 — Hand off for analysis
Tell me the chosen LR + that Family D is saved (and where). I then load it into
the base model with `model.load_state_dict(torch.load('router_state_full_*.pt'),
strict=False)` and run the **routing-divergence** + **capability** captures on it
(on RunPod, via `capture_routing.py` + a 3-line gate-loader) → fills report §5.5,
and tests the headline: does *directly* re-routing buy capability, or confirm
"familiarity, not competence"?

---

### Quick reference — what each knob is
| knob | cell | default | when to change |
|---|---|---|---|
| `MODE` | 3 | `"probe"` | `"full"` for the real Family-D run |
| `LR` | 3 | `1e-4` | probe each of 1e-4/3e-5/1e-5; full = chosen |
| `MASK_PROMPT` | 3 | `False` | `False` = full-seq loss (matches your LoRA recipe); `True` = assistant-only |
| `MAX_LEN` | 3 | `4096` | lower on OOM |
| `AUX_COEF` | 3 | `0.01` | raise if an expert collapses; lower if balance fights the loss |

(Blackwell init — shadow mamba3, ptxas patch, force-slow-path — is baked into
cells 2 & 4, not knobs.)
