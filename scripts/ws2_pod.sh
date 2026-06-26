#!/usr/bin/env bash
# NemoH WS2: router-only training → Family D adapter.
#
# Decisions locked 2026-06-23 (report/router_training_plan.md §8):
#   - Plain unfreeze of 23 gate.weight tensors only (~7.9M params).
#   - LR probe: aux_coef∈{0.1,0.05} × lr=3e-5 (100 steps each).
#     1e-4+aux=0.01 collapsed at step ~25-30 on Kaggle (§10);
#     testing heavier aux first, lr=3e-5 only since 1e-4 also collapsed.
#   - Full run at first stable (non-COLLAPSE) config.
#   - Self-DELETE on WS2_COMPLETE; self-STOP on failure (debuggable).
#
# Deploy (same pattern as WS1):
#   1. Create pod in EU-FR-1 attached to volume gdqj7o63ik.
#      Do NOT put RUNPOD_API_KEY in pod env (secret scanner).
#   2. After boot, write /root/.podenv (umask 077): POD_ID=... RP_KEY=...
#   3. scp scripts/ws2_pod.sh and scripts/train_router.py and
#      WONDERLAND_FINAL_MASTER.jsonl to /workspace/ws2/ on the pod.
#   4. nohup bash /workspace/ws2/ws2_pod.sh &
set -uo pipefail
ROOT=/workspace/ws2
LOG=$ROOT/run.log
mkdir -p $ROOT
exec >> "$LOG" 2>&1
echo "=== WS2 pod start $(date -Is) ==="

export HF_HOME=/workspace/hf HUGGINGFACE_HUB_CACHE=/workspace/hf/hub PYTHONUNBUFFERED=1
export PIP_BREAK_SYSTEM_PACKAGES=1

echo "--- verify preinstalled torch (image ships release 2.8.0) ---"
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo FATAL-NOTORCH; exit 1; }

echo "--- install deps ---"
pip install -q "transformers==4.57.3" peft accelerate "numpy<2.4" einops
# mamba kernels (needed for model forward pass):
pip install -q causal-conv1d --no-deps --no-build-isolation
pip install -q mamba-ssm     --no-deps --no-build-isolation
python3 -c "import mamba_ssm, causal_conv1d; print('mamba kernels OK')" \
  || { echo FATAL-KERNELS; exit 1; }

echo "--- verify volume: model cache ---"
ls /workspace/hf/hub | head -3 || { echo "FATAL: /workspace/hf missing — wrong volume?"; exit 1; }

echo "--- verify data + script ---"
[ -f "$ROOT/train_router.py" ] || { echo "FATAL: train_router.py not found in $ROOT"; exit 1; }
[ -f "$ROOT/WONDERLAND_FINAL_MASTER.jsonl" ] || { echo "FATAL: WONDERLAND_FINAL_MASTER.jsonl not found"; exit 1; }
NRECS=$(wc -l < "$ROOT/WONDERLAND_FINAL_MASTER.jsonl")
echo "dataset: $NRECS records"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# ─── pod self-management ───
# Must be defined before any call site (called in the probe-all-collapsed path below).
_self_stop() {
  PODID=""; PODKEY=""
  [ -f /root/.podenv ] && . /root/.podenv && PODID="${POD_ID:-}" && PODKEY="${RP_KEY:-}"
  if [ -z "$PODID" ] || [ -z "$PODKEY" ]; then
    echo "no creds in /root/.podenv — leaving pod up for manual cleanup"
    return
  fi
  if grep -q WS2_COMPLETE "$ROOT/DONE" 2>/dev/null; then
    echo "--- success: self-DELETE pod $PODID ---"
    curl -s -X DELETE -H "Authorization: Bearer $PODKEY" \
      "https://rest.runpod.io/v1/pods/$PODID" -w "delete HTTP %{http_code}\n"
  else
    echo "--- incomplete/failure: self-STOP pod $PODID ---"
    curl -s -X POST -H "Authorization: Bearer $PODKEY" \
      "https://rest.runpod.io/v1/pods/$PODID/stop" -w "stop HTTP %{http_code}\n"
  fi
}

# ─── probe: find a stable (aux_coef, lr) ───
# Kaggle pre-runs (2026-06-25):
#   lr∈{3e-5,1e-5} × aux∈{0.1,0.05} all collapsed at step ~25 (max_load 0.49→0.78).
#   Collapse is phase-transition triggered by specific batch (seed=42, same data order).
#   LR is NOT the variable: switching from 3e-5→1e-5 doesn't delay collapse at all.
#   Normal baseline max_load is ~0.55 (model already has concentrated routing).
#   balance-cap must be >0.8 so probe runs past the natural baseline.
#   Next to try: much stronger aux (0.5, 1.0) at lr=1e-5.
CHOSEN_AUX=""
CHOSEN_LR=""

# Probe order: strongest aux first (most likely to hold the phase transition back)
for AUX in 1.0 0.5 0.1 0.05; do
  for LR in 1e-5 1e-6; do
    PDIR="$ROOT/probe_aux${AUX}_lr${LR}"
    echo "--- probe: aux=$AUX lr=$LR ---"
    python3 "$ROOT/train_router.py" \
      --model-path "$MODEL" \
      --data "$ROOT/WONDERLAND_FINAL_MASTER.jsonl" \
      --out-dir "$PDIR" \
      --lr "$LR" --aux-coef "$AUX" \
      --max-steps 100 \
      --balance-cap 0.80 \
      --max-len 2048
    if [ -f "$PDIR/COLLAPSE" ]; then
      echo "  probe COLLAPSED (aux=$AUX lr=$LR) — $(cat $PDIR/COLLAPSE)"
    else
      echo "  probe STABLE (aux=$AUX lr=$LR)"
      CHOSEN_AUX=$AUX
      CHOSEN_LR=$LR
      break 2
    fi
  done
done

if [ -z "$CHOSEN_AUX" ]; then
  echo "FATAL: all probes (lr∈{1e-5,1e-6} × aux∈{1.0,0.5,0.1,0.05}) collapsed"
  echo "WS2_ALL_COLLAPSED $(date -Is)" > $ROOT/DONE
  sync
  _self_stop; exit 1
fi

echo "=== selected config: aux_coef=$CHOSEN_AUX lr=$CHOSEN_LR ==="

# ─── full training run ───
FULL_DIR="$ROOT/familyD"
echo "--- full run: 1 epoch, aux=$CHOSEN_AUX lr=$CHOSEN_LR ---"
python3 "$ROOT/train_router.py" \
  --model-path "$MODEL" \
  --data "$ROOT/WONDERLAND_FINAL_MASTER.jsonl" \
  --out-dir "$FULL_DIR" \
  --lr "$CHOSEN_LR" --aux-coef "$CHOSEN_AUX" \
  --epochs 1 \
  --balance-cap 0.82 \
  --max-len 4096
TRAIN_EXIT=$?
echo "full run exit: $TRAIN_EXIT"

# ─── check outcome ───
if [ "$TRAIN_EXIT" -eq 0 ] && [ -f "$FULL_DIR/router_state.pt" ] && [ ! -f "$FULL_DIR/COLLAPSE" ]; then
  echo "WS2_COMPLETE aux=$CHOSEN_AUX lr=$CHOSEN_LR $(date -Is)" > $ROOT/DONE
  echo "router_state.pt: $(du -sh $FULL_DIR/router_state.pt)"
else
  echo "WS2_INCOMPLETE exit=$TRAIN_EXIT $(date -Is)" > $ROOT/DONE
fi
sync

# ─── self-manage the pod ───
_self_stop
