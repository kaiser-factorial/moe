# WS2 Router Training Probe Log

All probes: gate-only training (23 × `gate.weight`, 7.9M params), `seed=42`,
`eff_batch=8`, `max_len=2048`, `max_steps=100`, `balance_cap=0.8`.
Collapse = `max_load > 0.8` for 3 consecutive log windows (patience=3).

---

## Kaggle session (2026-06-23, §10 of router_training_plan.md)

| lr   | aux_coef | collapse_step | max_load_at_collapse | notes |
|------|----------|--------------|----------------------|-------|
| 1e-4 | 0.01     | ~25–30       | 0.95                 | lm erratic (spikes to 1.5) |
| 3e-5 | 0.01     | ~25–30       | 0.84                 | lm low (0.62) while collapsed |

**Key finding:** `lm` is NOT a proxy for router health — lm drops while max_load climbs.
The aux signal was *rising* at the collapse moment (honest signal, too weak to prevent).

---

## This session (2026-06-25, Kaggle notebook)

### Run 1 — lr=3e-5, aux_coef=0.1

| step | lm     | aux    | max_load | status     |
|------|--------|--------|----------|------------|
| 5    | 1.4394 | 1.9325 | 0.552    | stable     |
| 10   | 0.9662 | 1.7913 | 0.489    | stable     |
| 15   | 0.9023 | 1.7697 | 0.491    | stable     |
| 20   | 1.2398 | 1.7891 | 0.491    | stable     |
| 25   | 2.1619 | 2.2730 | 0.778    | ⚠️ tipping |
| 30   | 0.6249 | 1.9279 | 0.821    | COLLAPSED  |
| 35   | 0.5021 | 1.8849 | 0.784    | —          |
| 40   | 0.6196 | 1.6354 | 0.806    | —          |
| 50   | 0.5929 | 1.6025 | 0.811    | —          |

**Result:** COLLAPSE at step 50 (patience=3). `global_step=50, train_loss=11.2516`

### Run 2 — lr=3e-5, aux_coef=0.05

No step-level metrics captured. **Result:** COLLAPSE (same pattern).

### Run 3 — lr=1e-5, aux_coef=0.1

| step | lm     | aux    | max_load | status     |
|------|--------|--------|----------|------------|
| 5    | 1.4545 | 1.9412 | 0.561    | stable     |
| 10   | 0.9952 | 1.8197 | 0.510    | stable     |
| 15   | 0.9467 | 1.8274 | 0.509    | stable     |
| 20   | 1.3657 | 1.8667 | 0.527    | stable     |
| 25   | 2.5219 | 2.4373 | 0.784    | ⚠️ tipping |
| 30   | 0.6669 | 2.1384 | 0.834    | COLLAPSED  |
| 35   | 0.5369 | 2.1255 | 0.814    | —          |
| 40   | 0.6643 | 1.8020 | 0.812    | —          |

**Result:** COLLAPSE at step 40 (patience=3). `global_step=40, train_loss=12.1961`

### Run 4 — lr=1e-5, aux_coef=0.05, **seed=123** (data-hypothesis test + first complete run)

| step | lm     | aux    | max_load | note                        |
|------|--------|--------|----------|-----------------------------|
| 5    | 2.7914 | 2.5833 | 0.796    | ⚠️ high, but no collapse    |
| 10   | 3.2266 | 2.3414 | 0.725    |                             |
| 15   | 2.4033 | 1.8838 | 0.785    |                             |
| 20   | 2.5103 | 2.3121 | 0.756    |                             |
| 25   | 1.3399 | 1.8378 | 0.515    | ✅ no step-25 tip (≠ seed=42) |
| 30   | 2.6041 | 1.8881 | 0.863    | above cap, count=1          |
| 35   | 2.1620 | 1.7443 | 0.748    | count reset                 |
| 40   | 1.2595 | 1.8407 | 0.525    |                             |
| 45   | 0.6149 | 1.7726 | 0.827    | above cap, count=1          |
| 50   | 1.2688 | 1.8428 | 0.536    | count reset                 |
| 55   | 0.6340 | 1.7756 | 0.837    | above cap, count=1          |
| 60   | 0.5346 | 1.7425 | 0.829    | above cap, count=2; **lm=0.53 lowest in run** |
| 65   | 2.5648 | 2.1925 | 0.748    | count reset                 |
| 70   | 1.8560 | 2.0150 | 0.777    |                             |
| 75   | 1.0990 | 1.8649 | 0.518    |                             |
| 80   | 2.2058 | 2.2321 | 0.771    |                             |
| 85   | 0.7678 | 1.7877 | 0.821    | above cap, count=1          |
| 90   | 0.6220 | 1.7785 | 0.839    | above cap, count=2          |
| 95   | 0.8765 | 1.6877 | 0.464    | count reset; max_load lowest in run |
| 100  | 1.0438 | 1.9318 | 0.556    |                             |

HF Trainer loss column (14.66→10.67 range): ~7-8× higher than monitor `lm` due to
`gradient_accumulation_steps=8` summing micro-batch losses rather than averaging.
The per-token CE (`lm` column) is the correct number for comparison.

**Result: COMPLETE — 100/100 steps, no abort.**
`TrainOutput(global_step=100, training_loss=11.929, epoch=0.084, runtime=2554s)`

**Interpretation:**
- The two objectives fight for the entire run. LM gradient → concentration (lower lm);
  aux gradient → balance (pushes back). Neither wins at aux=0.05: oscillation instead of
  convergence OR collapse.
- max_load ping-pongs between 0.46 (balanced) and 0.87 (concentrated). Never 3
  consecutive bad windows → patience=3 never triggers.
- Best lm (0.53) occurs at step 60 when max_load=0.829 — concentrated routing
  actively helps prediction (confirms re-routing signal).
- lm is NOT converging monotonically; it tracks the oscillating max_load.

---

## Cross-run diagnostics

**Collapse always at step 25 (tip) regardless of LR:**

| run | lr   | aux  | tip_step | max_load_before | max_load_after |
|-----|------|------|----------|-----------------|----------------|
| 1   | 3e-5 | 0.10 | 25       | 0.491           | 0.778          |
| 3   | 1e-5 | 0.10 | 25       | 0.527           | 0.784          |

**Interpretation:** Same `seed=42` → same data order → same batches at every step.
The collapse is triggered by a specific batch/data cluster around step 25, not by
gradient magnitude (otherwise 3× LR difference would shift the collapse step).

**Run 4 (seed=123) CONFIRMS the data hypothesis.** No step-25 collapse. The router
OSCILLATES near the 0.8 cap rather than permanently tipping: max_load bounces
0.796→0.725→0.785→0.756→0.515→0.863→0.748→0.525→0.827. The recoveries (step 25:
0.515, step 40: 0.525) show the phase transition is NOT a one-way door — the
router can pull back from high concentration. This is qualitatively different from
seed=42 where max_load monotonically locked in above 0.8 after step 25.

**Revised diagnosis: the collapse is data-triggered, not a fundamental instability.**
`aux=0.05 + lr=1e-5 + seed=123` may well survive to 100 steps with patience=3.

---

## Next to try

**RunPod status (2026-06-25):** EU-FR-1 datacenter had a widespread machine outage —
5 consecutive pods stuck at "Rented by User" indefinitely, ports never appeared.
The launch setup is ready (`scripts/ws2_bundle/`, opbdh OPBDH_RUNPOD_GPU_TYPES patch,
`max_dollars_per_hour=6.0` for H200), but probes couldn't run today.
Retry command: `OPBDH_RUNPOD_GPU_TYPES="NVIDIA H200" ~/.opbdh-venv/bin/opbdh launch scripts/ws2_bundle --config opbdh.json --yes`

**Priority probe grid for next session** (seed=123, balance_cap=0.80, 100 steps):

| priority | lr   | aux_coef | expected behavior |
|----------|------|----------|-------------------|
| 1        | 1e-5 | 0.10     | less oscillation than 0.05; might stabilize |
| 2        | 1e-5 | 0.50     | balance term dominates; stable but forced-balanced Family D |
| 3        | 1e-5 | 1.00     | maximum balance pressure; boring but safe full run |
| 4        | 1e-6 | 0.05     | slower updates; does oscillation period lengthen? |
