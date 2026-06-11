#!/usr/bin/env python3
"""Isolate the batched-generation degeneracy: single row vs uniform batch vs
heavily padded batch. Greedy decoding for determinism.

Usage: python debug_padding.py --model-path M
"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from capture_routing import apply_nemotron_patches, shadow_mamba

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

SHORT = "What is the capital of France? Answer in one word."
LONG = ("Here is some context. " * 40
        + "Now: What is the capital of France? Answer in one word.")

def gen(texts, label, max_new=48):
    msgs = [tok.apply_chat_template([{"role": "user", "content": t}],
                                    tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False) for t in texts]
    enc = tok(msgs, return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    print(f"\n=== {label} (prompt_len={enc['input_ids'].shape[1]}) ===")
    for i in range(out.shape[0]):
        txt = tok.decode(out[i][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  row{i}: {txt[:160]!r}")

gen([SHORT], "A: single row, no padding")
gen([SHORT] * 4, "B: batch 4, identical rows (no padding)")
gen([LONG, SHORT, SHORT, SHORT], "C: batch 4, heavy left-padding on rows 1-3")
print("\ndone")

# ── Long-generation tests: replicate actual failure conditions ──────────────
MMLU = ("As of 2016, about what percentage of adults aged 18 years or older "
        "were overweight?\nA. 10%\nB. 20%\nC. 40%\nD. 80%\n\n"
        "Answer with the letter of the correct choice.\n\n"
        "Think step by step, then give your final answer as \\boxed{letter}.")

def gen2(texts, label, max_new=1024, thinking=True, sample=True):
    msgs = [tok.apply_chat_template([{"role": "user", "content": t}],
                                    tokenize=False, add_generation_prompt=True,
                                    enable_thinking=thinking) for t in texts]
    enc = tok(msgs, return_tensors="pt", padding=True).to("cuda")
    kw = dict(temperature=0.6, top_p=0.95, do_sample=True) if sample else dict(do_sample=False)
    torch.manual_seed(42)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new,
                             pad_token_id=tok.eos_token_id, **kw)
    print(f"\n=== {label} ===")
    for i in range(out.shape[0]):
        txt = tok.decode(out[i][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  row{i} ({len(txt)} ch): head={txt[:100]!r}")
        print(f"        tail={txt[-120:]!r}")

gen2([MMLU], "D: single row, thinking, sampled, 1024 tok")
gen2([MMLU] * 8, "E: batch 8 identical, thinking, sampled, 1024 tok")
print("\ndone2")
