#!/usr/bin/env bash
# NemoH Phase 2: autonomous LoRA routing capture (as run 2026-06-11, pod ynkqgx8vm6qhfh).
# Runs under nohup ON the RunPod pod; everything persists to the network volume;
# self-deletes the pod on success, self-stops on failure. See RUNLOG.md Session 3
# for the error history that shaped every line of this file.
#
# Deploy notes (RUNLOG S3):
# - Create the pod WITHOUT the account RUNPOD_API_KEY in env (secret scanner
#   terminates such pods in 3-11 min). After boot, scp/write /root/.podenv
#   (umask 077) containing POD_ID=<id> and RP_KEY=<account key> — a file on
#   container disk is invisible to the scanner and dies with the pod.
# - Make the pod-local capture copy first:
#     cp scripts/capture_routing.py /workspace/phase2/capture_routing_h100.py
#   then neuter patch_ptxas() (insert `return` as its first line): the
#   Blackwell ptxas redirect crashes on release-torch triton (get_ptxas
#   signature changed) and H100/Hopper doesn't need it.
set -uo pipefail
LOG=/workspace/phase2/run.log
mkdir -p /workspace/phase2/logs/lora /workspace/phase2/logs/base_recovered
exec >> "$LOG" 2>&1
echo "=== phase2 pod start $(date -Is) ==="

export HF_HOME=/workspace/hf HUGGINGFACE_HUB_CACHE=/workspace/hf/hub PYTHONUNBUFFERED=1

echo "--- env pins (RUNLOG recipe v2: image ships a DEV torch; pin release) ---"
pip install -q "transformers==4.57.3" peft accelerate "numpy<2.4" einops
# Release torch WITH its own deps (--no-deps breaks NCCL: ncclCommWindowRegister)
pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# Dev-built torchvision breaks transformers import on release torch
# ("operator torchvision::nms does not exist") — and we don't need it.
pip uninstall -q -y mamba-ssm causal-conv1d torchvision torchaudio 2>/dev/null
# Prebuilt mamba wheels target release-torch ABI — fine now that torch is pinned.
pip install -q causal-conv1d --no-deps --no-build-isolation
pip install -q mamba-ssm --no-deps --no-build-isolation
python3 -c "import torch; print('torch', torch.__version__); import mamba_ssm, causal_conv1d; import torch as t; x=t.randn(2,4,device='cuda'); print('mamba kernels OK, cuda OK', x.sum().item()!=None)" || { echo FATAL-KERNELS; exit 1; }

echo "--- redirect HF cache to existing NemoH volume cache ---"
ls /workspace/hf/hub | head -3 || { echo "FATAL: /workspace/hf missing — wrong volume?"; exit 1; }

echo "--- recover base symb-medi-17 (synced to repo as 0 bytes after Phase 1) ---"
cp -v /workspace/moe/outputs/logs/base/routing_symb-medi-17.npz /workspace/phase2/logs/base_recovered/

echo "--- capture: LoRA-adapted, 111 problems ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /workspace/moe
python3 /workspace/phase2/capture_routing_h100.py \
  --model-path "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" \
  --adapter-path "brick-factorial/nemotron-lora-symbolic-reasoning" \
  --problems data/problems.json \
  --out-dir /workspace/phase2/logs/lora \
  --batch-size 8 --skip-existing
CAPTURE_EXIT=$?
echo "capture exit: $CAPTURE_EXIT"

N=$(ls /workspace/phase2/logs/lora/routing_*.npz 2>/dev/null | wc -l)
echo "npz files: $N / 111"
if [ "$CAPTURE_EXIT" -eq 0 ] && [ "$N" -ge 111 ]; then
  echo "PHASE2_COMPLETE $(date -Is)" > /workspace/phase2/DONE
else
  echo "PHASE2_INCOMPLETE exit=$CAPTURE_EXIT npz=$N $(date -Is)" > /workspace/phase2/DONE
fi
sync

# account key lives in /root/.podenv (container-disk file — invisible to the
# env secret-scanner; dies with the pod). Pod-scoped key can't manage pods (403).
PODID=""; PODKEY=""
[ -f /root/.podenv ] && . /root/.podenv && PODID="$POD_ID" && PODKEY="$RP_KEY"
if [ -z "$PODID" ] || [ -z "$PODKEY" ]; then
  echo "no credentials in /root/.podenv; leaving pod running for manual cleanup"
elif grep -q PHASE2_COMPLETE /workspace/phase2/DONE; then
  echo "--- success: self-DELETE pod $PODID ---"
  curl -s -X DELETE -H "Authorization: Bearer $PODKEY" "https://rest.runpod.io/v1/pods/$PODID" -w "delete HTTP %{http_code}\n"
else
  echo "--- failure: self-STOP pod $PODID (debuggable, disk-only billing) ---"
  curl -s -X POST -H "Authorization: Bearer $PODKEY" "https://rest.runpod.io/v1/pods/$PODID/stop" -w "stop HTTP %{http_code}\n"
fi
