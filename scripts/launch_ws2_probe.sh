#!/usr/bin/env bash
# opbdh launcher: WS2 probe-only (no full training run).
# Runs aux∈{1.0,0.5,0.1,0.05} × lr∈{1e-5,1e-6} probe grid, saves per-probe
# logs + metrics to /workspace/ws2/probes/, then self-STOPs the pod.
# Results synced back to outputs/opbdh_runs/ by opbdh.
set -uo pipefail
echo "=== launch_ws2_probe.sh: staging ==="

ls /workspace/hf/hub | head -3 || { echo "FATAL: /workspace/hf not found — wrong volume?"; exit 1; }

export HF_HOME=/workspace/hf
export HUGGINGFACE_HUB_CACHE=/workspace/hf/hub
export PIP_BREAK_SYSTEM_PACKAGES=1
export PYTHONUNBUFFERED=1

mkdir -p /workspace/ws2
WS2=/workspace/ws2
REPO=/workspace/moe

# Stage scripts from volume repo
cp "$REPO/scripts/train_router.py"        "$WS2/train_router.py"
cp "$REPO/WONDERLAND_FINAL_MASTER.jsonl"  "$WS2/WONDERLAND_FINAL_MASTER.jsonl"

echo "--- install deps ---"
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "FATAL: torch missing"; exit 1; }
pip install -q "transformers==4.57.3" peft accelerate "numpy<2.4" einops
pip install -q causal-conv1d --no-deps --no-build-isolation
pip install -q mamba-ssm     --no-deps --no-build-isolation
python3 -c "import mamba_ssm, causal_conv1d; print('mamba kernels OK')" \
  || { echo "FATAL: mamba kernels failed"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
PROBES="$WS2/probes"
mkdir -p "$PROBES"

echo "=== starting probe sweep ==="
# Strongest aux first — LR is not the variable (both 3e-5 and 1e-5 tip at same
# step with same data seed). We want to know if higher aux survives step ~25.
for AUX in 1.0 0.5 0.1 0.05; do
  for LR in 1e-5 1e-6; do
    TAG="aux${AUX}_lr${LR}"
    PDIR="$PROBES/$TAG"
    echo "--- probe $TAG ---"
    python3 "$WS2/train_router.py" \
      --model-path "$MODEL" \
      --data "$WS2/WONDERLAND_FINAL_MASTER.jsonl" \
      --out-dir "$PDIR" \
      --lr "$LR" --aux-coef "$AUX" \
      --max-steps 100 \
      --balance-cap 0.80 \
      --seed 123 \
      --max-len 2048
    if [ -f "$PDIR/COLLAPSE" ]; then
      echo "  COLLAPSED — $(cat $PDIR/COLLAPSE)"
    else
      echo "  STABLE — $TAG is a candidate for full run"
    fi
  done
done

echo "=== probe sweep complete $(date -Is) ==="
echo "results in $PROBES:"
for d in "$PROBES"/*/; do
  tag=$(basename "$d")
  if [ -f "$d/COLLAPSE" ]; then status="COLLAPSE"; else status="STABLE"; fi
  echo "  $tag → $status"
done

# Self-stop so Corina can inspect results / opbdh syncs before pod dies
PODID=""; PODKEY=""
[ -f /root/.podenv ] && . /root/.podenv && PODID="${POD_ID:-}" && PODKEY="${RP_KEY:-}"
if [ -n "$PODID" ] && [ -n "$PODKEY" ]; then
  echo "--- self-STOP pod $PODID ---"
  curl -s -X POST -H "Authorization: Bearer $PODKEY" \
    "https://rest.runpod.io/v1/pods/$PODID/stop" -w "stop HTTP %{http_code}\n"
else
  echo "no creds in /root/.podenv — pod stays up for manual inspection"
fi
echo "=== launch_ws2_probe.sh done ==="
