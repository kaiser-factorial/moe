#!/usr/bin/env python3
"""Inspect cache state after prefill: are SSM states actually populated?

Captures the cache used by generate(), then prints per-layer:
  - mamba layers: ssm_state abs-mean, conv_state abs-mean
  - attention layers: key_cache shape (should cover the prompt)

Usage: python state_probe.py --model-path M
"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from capture_routing import apply_nemotron_patches, shadow_mamba, MOE_LAYERS

p = argparse.ArgumentParser()
p.add_argument("--model-path", required=True)
args = p.parse_args()

shadow_mamba()
tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
    trust_remote_code=True, low_cpu_mem_usage=True)
model.eval()
apply_nemotron_patches(model)

# capture the cache object generate() actually uses
import functools
captured = {}
_prep = model.prepare_inputs_for_generation
@functools.wraps(_prep)
def prep(*a, **k):
    mi = _prep(*a, **k)
    if isinstance(mi, dict) and mi.get("cache_params") is not None:
        captured["cache"] = mi["cache_params"]
    return mi
model.prepare_inputs_for_generation = prep

PROMPT = ("My grandmother's name is Beatrix and she lives in Lisbon. "
          "She has a parrot called Marco who only speaks Italian. "
          "Every Sunday she bakes almond cake for the neighbors. "
          "Question: what is the name of my grandmother's parrot? "
          "Answer in one word.")
msg = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                              tokenize=False, add_generation_prompt=True,
                              enable_thinking=True)
enc = tok(msg, return_tensors="pt").to("cuda")
T = enc["input_ids"].shape[1]
print(f"prompt tokens: {T}", flush=True)

with torch.inference_mode():
    out = model.generate(**enc, max_new_tokens=64, do_sample=False,
                         pad_token_id=tok.eos_token_id)
print("OUTPUT:", repr(tok.decode(out[0][T:], skip_special_tokens=True)), flush=True)

cache = captured.get("cache")
if cache is None:
    raise SystemExit("no cache captured!")

cfg = model.config
pattern = cfg.hybrid_override_pattern
print("\nlayer | type | ssm_state absmean | conv_state absmean | kv shape", flush=True)
for i, ch in enumerate(pattern):
    ssm = cache.ssm_states[i]
    conv = cache.conv_states[i]
    kv = cache.key_cache[i]
    ssm_m = float(ssm.float().abs().mean()) if ssm.numel() else -1
    conv_m = float(conv.float().abs().mean()) if conv.numel() else -1
    if ch == "M":
        print(f"  {i:2d}   mamba  {ssm_m:.6f}   {conv_m:.6f}", flush=True)
    elif ch == "*":
        print(f"  {i:2d}   attn   kv={tuple(kv.shape)}", flush=True)
print("done", flush=True)
