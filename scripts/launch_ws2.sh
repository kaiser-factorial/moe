#!/usr/bin/env bash
# opbdh launcher for WS2 (router training → Family D).
#
# opbdh uploads this script as the user code, runs it inside /opbdh-run/user/
# via job.sh. This launcher:
#   1. Copies ws2_pod.sh + train_router.py + WONDERLAND to /workspace/ws2/
#   2. Starts the actual training under nohup (self-managing)
#   3. Exits 0 immediately so opbdh sees success and keeps the pod alive
#      (opbdh.json: keep_pod_on_success=true)
#
# The pod then runs for ~8-12h and self-DELETEs when WS2_COMPLETE or
# self-STOPs on failure. Check /workspace/ws2/run.log for progress.
#
# opbdh overrides HF_HOME → /workspace/opbdh-cache/huggingface; we need to
# point it back at our existing model cache at /workspace/hf.
set -uo pipefail
echo "=== launch_ws2.sh: setting up WS2 ==="

# Verify our model cache is present (wrong volume would fail here)
ls /workspace/hf/hub | head -3 || { echo "FATAL: /workspace/hf not found — wrong volume?"; exit 1; }

# Re-point HF to our existing cache (opbdh sets it to /workspace/opbdh-cache)
export HF_HOME=/workspace/hf
export HUGGINGFACE_HUB_CACHE=/workspace/hf/hub

mkdir -p /workspace/ws2

# Copy scripts from the moe repo on the volume
WS2=/workspace/ws2
REPO=/workspace/moe

cp "$REPO/scripts/ws2_pod.sh"    "$WS2/ws2_pod.sh"
cp "$REPO/scripts/train_router.py" "$WS2/train_router.py"
cp "$REPO/WONDERLAND_FINAL_MASTER.jsonl" "$WS2/WONDERLAND_FINAL_MASTER.jsonl"

echo "files staged in $WS2:"
ls -lh "$WS2/"

# Write pod credentials (needed for self-delete/stop)
# RUNPOD_API_KEY and POD_ID are injected via Desktop Commander after boot
# (see ws2_pod.sh deploy instructions); this launcher just starts the job.
echo "starting ws2_pod.sh under nohup..."
nohup bash "$WS2/ws2_pod.sh" > "$WS2/launch.log" 2>&1 &
WS2_PID=$!
echo "ws2_pod.sh PID: $WS2_PID"
sleep 3
if kill -0 $WS2_PID 2>/dev/null; then
  echo "ws2 job is running — launcher exiting (opbdh keeps pod alive)"
else
  echo "WARN: ws2 job exited quickly — check $WS2/launch.log"
  cat "$WS2/launch.log" 2>/dev/null | tail -10
fi

echo "=== launch_ws2.sh done ==="
