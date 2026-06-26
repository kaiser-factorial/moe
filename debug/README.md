# debug/ — Debug & Error Logs

Running log of issues hit while building/running this project and how they were
fixed. Start here when something breaks; add to the relevant file (or a new
dated one) when you burn time on a fix worth remembering.

## In this folder
- [KAGGLE_DEBUG.md](KAGGLE_DEBUG.md) — Kaggle notebook / 2×T4 issues. Latest:
  3-pass review of the router-probe notebook (aux-loss grad-checkpointing bug,
  bf16-on-Turing, multi-GPU reduction, etc.).
- [runpod_ws1_setup_2026-06-23.md](runpod_ws1_setup_2026-06-23.md) — WS1 launch
  gotchas (PEP668 pip, pkill self-match, kagglehub stdout-in-var, slow-capture
  budget reality, reusable REST+SSH launch recipe).

## Related debug/error docs elsewhere in the repo (not moved, to keep links intact)
- `RUNLOG.md` (root) — chronological run log; RunPod env-recipe v2, operational
  gotchas (secret scanner, DC pinning, env pins), Mamba/torch ABI fixes.
- `report/errors_postmortem.md` — earlier errors & fixes postmortem.
- `report/upstream_bug_investigation.md` — NemotronH cached-generation bug:
  does it affect other family members?
- `report/upstream_bug_runtime_plan.md` — runtime plan to confirm that bug
  across the model family.

> If we want everything physically in `debug/`, these can be moved here and the
> references in `report/` updated — say the word.

## Convention
- One file per environment/theme (kaggle, runpod, …) or per big incident.
- Newest entry first within a file; date each entry.
- For each issue: **symptom → root cause → fix** (and "not a bug" confirmations,
  which save the next person from re-investigating).
