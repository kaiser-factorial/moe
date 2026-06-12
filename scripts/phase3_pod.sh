#!/usr/bin/env bash
# NemoH Phase 3 Stage 0: vector extraction + steering pilot (phase3_design.md §4).
# Same operational pattern as phase2_pod.sh (see RUNLOG S3): runs under nohup ON
# the pod, persists everything to the network volume, self-deletes on success,
# self-stops on failure. Account key goes in /root/.podenv AFTER boot (umask 077),
# NEVER in pod env at create (secret scanner kills the pod in 3-11 min).
#
# Deploy:
#   scp scripts/phase3_extract_vectors.py scripts/phase3_pilot.py -> /workspace/phase3/
#   scp scripts/capture_routing.py -> /workspace/phase3/  (imported by both;
#       patch_ptxas is NOT called by phase3 scripts, no neutering needed)
#   write /root/.podenv: POD_ID=<id> RP_KEY=<account key>
#   nohup bash /workspace/phase3/phase3_pod.sh &
#
# BUDGET NOTE: balance ~$5; hard timeouts below keep total pod life < ~75 min.
set -uo pipefail
PH=/workspace/phase3
LOG=$PH/run.log
mkdir -p $PH/pilot
exec >> "$LOG" 2>&1
echo "=== phase3 pilot pod start $(date -Is) ==="

export HF_HOME=/workspace/hf HUGGINGFACE_HUB_CACHE=/workspace/hf/hub PYTHONUNBUFFERED=1

echo "--- env pins (RUNLOG recipe v2) ---"
pip install -q "transformers==4.57.3" peft accelerate "numpy<2.4" einops
pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip uninstall -q -y mamba-ssm causal-conv1d torchvision torchaudio 2>/dev/null
pip install -q causal-conv1d --no-deps --no-build-isolation
pip install -q mamba-ssm --no-deps --no-build-isolation
python3 -c "import torch; print('torch', torch.__version__); import mamba_ssm, causal_conv1d; x=torch.randn(2,4,device='cuda'); print('kernels OK', x.sum().item() is not None)" || { echo FATAL-KERNELS; STATUS=fail; }

ls /workspace/hf/hub | head -3 || { echo "FATAL: /workspace/hf missing — wrong volume?"; STATUS=fail; }

STATUS=${STATUS:-ok}
MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
cd /workspace/moe   # repo on volume: data/problems.json lives here

if [ "$STATUS" = ok ]; then
  echo "--- step 1/2: extract vectors (timeout 25m) ---"
  timeout 25m python3 $PH/phase3_extract_vectors.py \
    --model-path "$MODEL" --problems data/problems.json --out-dir $PH
  [ $? -eq 0 ] && [ -f $PH/vectors.npz ] || STATUS=fail
fi

if [ "$STATUS" = ok ]; then
  echo "--- step 2/2: steering pilot (timeout 40m) ---"
  timeout 40m python3 $PH/phase3_pilot.py \
    --model-path "$MODEL" --vectors $PH/vectors.npz \
    --problems data/problems.json --out-dir $PH/pilot
  [ $? -eq 0 ] && [ -f $PH/pilot/summary.json ] || STATUS=fail
fi

if [ "$STATUS" = ok ]; then
  echo "PHASE3_PILOT_COMPLETE $(date -Is)" > $PH/DONE
else
  echo "PHASE3_PILOT_INCOMPLETE $(date -Is)" > $PH/DONE
fi
sync

PODID=""; PODKEY=""
[ -f /root/.podenv ] && . /root/.podenv && PODID="$POD_ID" && PODKEY="$RP_KEY"
if [ -z "$PODID" ] || [ -z "$PODKEY" ]; then
  echo "no credentials in /root/.podenv; leaving pod running for manual cleanup"
elif grep -q PHASE3_PILOT_COMPLETE $PH/DONE; then
  echo "--- success: self-DELETE pod $PODID ---"
  curl -s -X DELETE -H "Authorization: Bearer $PODKEY" "https://rest.runpod.io/v1/pods/$PODID" -w "delete HTTP %{http_code}\n"
else
  # budget is ~$5: a stopped pod still bills disk; delete here too — log +
  # results are already on the volume, container disk holds nothing we need.
  echo "--- failure: self-DELETE pod $PODID (budget mode; all state on volume) ---"
  curl -s -X DELETE -H "Authorization: Bearer $PODKEY" "https://rest.runpod.io/v1/pods/$PODID" -w "delete HTTP %{http_code}\n"
fi
