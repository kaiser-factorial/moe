#!/usr/bin/env python3
"""Probe NemotronH module structure to locate the MoE router gate.

Run on the GPU pod BEFORE capture_routing.py. Prints the module tree of one
MoE mixer and identifies candidate router modules (Linear with out_features
== n_experts). Does a tiny forward pass to verify the gate hook sees
per-token logits.

Usage: python probe_model.py --model-path /path/to/model
"""
import argparse, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MOE_LAYERS = [1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51]
N_EXPERTS = 128

p = argparse.ArgumentParser()
p.add_argument("--model-path", required=True)
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
    trust_remote_code=True, low_cpu_mem_usage=True)
model.eval()

mixer = model.backbone.layers[MOE_LAYERS[0]].mixer
print(f"=== Mixer class: {type(mixer).__name__} ===")
print("--- attributes ---")
for name, mod in mixer.named_children():
    print(f"  .{name}: {type(mod).__name__}")

print("--- candidate router modules (Linear-ish with out_features==128) ---")
candidates = []
for name, mod in mixer.named_modules():
    out = getattr(mod, "out_features", None) or (
        mod.weight.shape[0] if hasattr(mod, "weight") and mod.weight is not None
        and mod.weight.dim() == 2 else None)
    if out == N_EXPERTS and "expert" not in name:
        candidates.append(name)
        print(f"  mixer.{name}: {type(mod).__name__} weight={tuple(mod.weight.shape)}")

# --- router (mixer.gate) trainable surface: what to target for router training ---
# From modeling_nemotron_h.py NemotronHTopkRouter.__init__:
#   self.weight = nn.Parameter([n_routed_experts, hidden])   <- ONLY trainable param
#   self.e_score_correction_bias = register_buffer([n_routed_experts])  <- BUFFER (no grad)
gate = getattr(mixer, "gate", None)
if gate is not None:
    print(f"--- mixer.gate = {type(gate).__name__}: parameters (trainable surface) ---")
    for n, prm in gate.named_parameters():
        print(f"  param  gate.{n:30s} shape={tuple(prm.shape)} dtype={prm.dtype} requires_grad={prm.requires_grad}")
    for n, buf in gate.named_buffers():
        print(f"  buffer gate.{n:30s} shape={tuple(buf.shape)} dtype={buf.dtype} (NOT grad-trained)")
    # count router params across all MoE layers (what router-only training would touch)
    total = 0
    for L in MOE_LAYERS:
        g = model.backbone.layers[L].mixer.gate
        total += sum(p.numel() for p in g.parameters())
    print(f"  >> total router params across {len(MOE_LAYERS)} MoE layers: {total:,}")
    print(f"  >> recommended: modules_to_save=['gate'] (full-finetune; param count is trivial)")
    print(f"  >> alt: PEFT target_parameters=['mixer.gate.weight'] for LoRA on the gate")

# verify hook output shape with a tiny forward
captured = {}
def hook(module, inp, out):
    o = out[0] if isinstance(out, tuple) else out
    captured["shape"] = tuple(o.shape) if torch.is_tensor(o) else str(type(o))
    if torch.is_tensor(o):
        captured["sample_row_sum_softmax"] = float(
            torch.softmax(o.float().flatten(0, -2)[0], -1).sum())

if candidates:
    target = dict(mixer.named_modules())[candidates[0]]
    h = target.register_forward_hook(hook)
    ids = tok("Hello, the capital of France is", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model(**ids)
    h.remove()
    print(f"--- hook test on mixer.{candidates[0]} ---")
    print(json.dumps(captured, indent=2))
else:
    print("NO ROUTER CANDIDATE FOUND — will need expert-level hooks fallback.")
    print("Full mixer tree:")
    for name, mod in mixer.named_modules():
        if name.count(".") <= 1:
            print(f"  {name}: {type(mod).__name__}")
