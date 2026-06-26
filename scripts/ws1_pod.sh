#!/usr/bin/env bash
# NemoH WS1: routing capture on Family B (lora3) + Family C (nemo-lora-4-all),
# to compare routing divergence across adapter families (see
# report/family_comparison_plan.md). Adapted from phase2_pod.sh (proven path).
#
# Runs under nohup ON the RunPod pod; persists to the network volume;
# self-DELETEs the pod on full success, self-STOPs on failure.
#
# Deploy (same as Phase 2 — see RUNLOG.md S3):
#  1. Create pod attached to volume gdqj7o63ik (holds base model + /workspace/hf
#     + /workspace/moe repo). Do NOT put the account RUNPOD_API_KEY in pod env
#     (secret scanner kills the pod in 3-11 min).
#  2. After boot, write /root/.podenv (umask 077): POD_ID=<id> and RP_KEY=<acct key>.
#  3. cp scripts/capture_routing.py /workspace/ws1/capture_routing_h100.py and
#     neuter patch_ptxas() (insert `return` as first line) — Hopper doesn't need it.
#  4. nohup bash scripts/ws1_pod.sh &
set -uo pipefail
ROOT=/workspace/ws1
LOG=$ROOT/run.log
mkdir -p $ROOT/logs/lora_B $ROOT/logs/lora_C
exec >> "$LOG" 2>&1
echo "=== WS1 pod start $(date -Is) ==="

export HF_HOME=/workspace/hf HUGGINGFACE_HUB_CACHE=/workspace/hf/hub PYTHONUNBUFFERED=1

echo "--- env (image runpod/pytorch torch280 already ships release torch 2.8.0 + cu128) ---"
export PIP_BREAK_SYSTEM_PACKAGES=1   # image python is PEP668 externally-managed
python3 -c "import torch; print('preinstalled torch', torch.__version__)" || { echo FATAL-NOTORCH; exit 1; }
pip install -q "transformers==4.57.3" peft accelerate "numpy<2.4" einops kagglehub
# prebuilt mamba/causal-conv1d wheels are ABI-OK against RELEASE torch (recipe v2)
pip install -q causal-conv1d --no-deps --no-build-isolation
pip install -q mamba-ssm --no-deps --no-build-isolation
python3 -c "import torch,mamba_ssm,causal_conv1d; print('torch',torch.__version__,'kernels OK')" || { echo FATAL-KERNELS; exit 1; }
ls /workspace/hf/hub | head -3 || { echo "FATAL: /workspace/hf missing — wrong volume?"; exit 1; }

echo "--- resolve adapters (public Kaggle models, no creds needed) ---"
# write paths to files so kagglehub's stdout logging can't pollute the var
python3 - <<'PY'
import kagglehub, os
pb=kagglehub.model_download('corinakaiser/lora3-checkpt800-1188/pyTorch/v3')
open('/workspace/ws1/B_DIR','w').write(os.path.join(pb,'lora-adapter3','checkpoint-1188'))
pc=kagglehub.model_download('corinakaiser/nemo-lora-4-all/pyTorch/v0')
open('/workspace/ws1/C_DIR','w').write(pc)
PY
B_DIR=$(cat /workspace/ws1/B_DIR)
C_DIR=$(cat /workspace/ws1/C_DIR)
echo "B (Family B, mamba+shared+routed): $B_DIR"
echo "C (Family C, everything):         $C_DIR"
[ -f "$B_DIR/adapter_config.json" ] || { echo "FATAL: B adapter_config missing"; exit 1; }
[ -f "$C_DIR/adapter_config.json" ] || { echo "FATAL: C adapter_config missing"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /workspace/moe
BASE="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

run_capture () {  # $1=family-label  $2=adapter-dir  $3=out-dir
  echo "--- capture $1: $2 ---"
  python3 /workspace/ws1/capture_routing_h100.py \
    --model-path "$BASE" --adapter-path "$2" \
    --problems data/problems.json --out-dir "$3" \
    --batch-size 8 --skip-existing
  echo "capture $1 exit: $?  npz: $(ls $3/routing_*.npz 2>/dev/null | wc -l)/111"
}

run_capture B "$B_DIR" "$ROOT/logs/lora_B"
run_capture C "$C_DIR" "$ROOT/logs/lora_C"

NB=$(ls $ROOT/logs/lora_B/routing_*.npz 2>/dev/null | wc -l)
NC=$(ls $ROOT/logs/lora_C/routing_*.npz 2>/dev/null | wc -l)
echo "totals: B=$NB/111  C=$NC/111"
if [ "$NB" -ge 111 ] && [ "$NC" -ge 111 ]; then
  echo "WS1_COMPLETE $(date -Is)" > $ROOT/DONE
else
  echo "WS1_INCOMPLETE B=$NB C=$NC $(date -Is)" > $ROOT/DONE
fi
sync

PODID=""; PODKEY=""
[ -f /root/.podenv ] && . /root/.podenv && PODID="$POD_ID" && PODKEY="$RP_KEY"
if [ -z "$PODID" ] || [ -z "$PODKEY" ]; then
  echo "no creds in /root/.podenv; leaving pod up for manual cleanup"
elif grep -q WS1_COMPLETE $ROOT/DONE; then
  echo "--- success: self-DELETE pod $PODID ---"
  curl -s -X DELETE -H "Authorization: Bearer $PODKEY" "https://rest.runpod.io/v1/pods/$PODID" -w "delete HTTP %{http_code}\n"
else
  echo "--- failure: self-STOP pod $PODID (disk-only billing, debuggable) ---"
  curl -s -X POST -H "Authorization: Bearer $PODKEY" "https://rest.runpod.io/v1/pods/$PODID/stop" -w "stop HTTP %{http_code}\n"
fi
