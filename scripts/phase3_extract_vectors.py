#!/usr/bin/env python3
"""Phase 3: extract mean-difference control vectors (symbolic direction).

Mean residual-stream difference between the 18 symbolic prompts and an
18-problem stratified non-symbolic sample (seeded), captured at the OUTPUT
of selected backbone blocks (i.e. the residual stream entering the next
block). Two site families per target MoE depth d:
  e<d>  post-MoE   : output of MoE block d itself
  m<j>  post-Mamba : output of the nearest mamba block j < d
Prefill-only forward, batch size 1 (no padding -> clean mean pooling).
Vectors come from the BASE model (Phase 3 steers the base model).

Outputs (--out-dir):
  vectors.npz : v_mean_<site> (H,) fp32  mean-pooled diff (primary)
                v_last_<site> (H,) fp32  last-token diff (alternative)
                sigma_<site>  ()   fp32  mean per-token L2 norm at site
  sites.json  : site -> {layer, block_type, family}, contrast problem ids

Usage:
  python phase3_extract_vectors.py --model-path M [--problems P]
      [--out-dir /workspace/phase3] [--depths 3 8 13 17 20 27 38 47]
"""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_routing import (MOE_LAYERS, SEED, ANSWER_SUFFIX, shadow_mamba)

DEFAULT_DEPTHS = [3, 8, 13, 17, 20, 27, 38, 47]
NEG_QUOTA = {"factual": 4, "computational": 4, "reasoning": 4,
             "creative": 3, "social_ethical": 3}          # = 18


def render(tok, prob):
    content = prob["prompt"] + (ANSWER_SUFFIX.get(prob["expected_output_type"], "")
                                if prob["answer"] else "")
    msgs = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=True)
    except Exception:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


def contrast_sets(problems):
    sym = sorted([p for p in problems if p["category"] == "symbolic"],
                 key=lambda p: p["problem_id"])
    assert len(sym) == 18, f"expected 18 symbolic, got {len(sym)}"
    rng = np.random.default_rng(SEED)
    neg = []
    for cat, n in sorted(NEG_QUOTA.items()):
        pool = sorted([p for p in problems if p["category"] == cat],
                      key=lambda p: p["problem_id"])
        idx = rng.choice(len(pool), size=n, replace=False)
        neg += [pool[i] for i in sorted(idx)]
    return sym, neg


def resolve_sites(base, depths):
    """site name -> layer index; verifies block types at runtime."""
    layers = base.backbone.layers
    types = [getattr(l, "block_type", type(l.mixer).__name__).lower()
             for l in layers]
    sites = {}
    for d in depths:
        assert d in MOE_LAYERS, f"depth {d} not a MoE layer"
        sites[f"e{d}"] = d
        j = next(i for i in range(d - 1, -1, -1) if "mamba" in types[i])
        sites[f"m{j}"] = j
    meta = {name: {"layer": L, "block_type": types[L],
                   "family": "post-moe" if name[0] == "e" else "post-mamba"}
            for name, L in sites.items()}
    return sites, meta


class StateTap:
    """Read-only hooks on block outputs; per-prompt mean/last vectors."""
    def __init__(self, base, sites):
        self.buf = {}      # site -> (sum_vec fp32 cpu, n_tok, last_vec)
        self.norm_sum = {s: 0.0 for s in sites}
        self.norm_n = {s: 0 for s in sites}
        self.handles = [base.backbone.layers[L].register_forward_hook(
            self._mk(name)) for name, L in sites.items()]

    def _mk(self, name):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out   # (1, T, H)
            h = t.detach().float().squeeze(0)               # (T, H)
            self.buf[name] = (h.mean(0).cpu(), h[-1].cpu())
            n = h.norm(dim=-1)
            self.norm_sum[name] += float(n.sum()); self.norm_n[name] += n.shape[0]
        return hook

    def detach(self):
        for h in self.handles: h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--problems", default="data/problems.json")
    ap.add_argument("--out-dir", default="/workspace/phase3")
    ap.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    shadow_mamba()
    torch.manual_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    problems = json.load(open(args.problems))["problems"]
    sym, neg = contrast_sets(problems)
    print(f"contrast: {len(sym)} symbolic vs {len(neg)} stratified "
          f"({[p['problem_id'] for p in neg]})", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model

    sites, meta = resolve_sites(base, args.depths)
    print(f"{len(sites)} sites: { {n: m['layer'] for n, m in meta.items()} }",
          flush=True)
    tap = StateTap(base, sites)

    acc = {s: {"pos_mean": [], "pos_last": [], "neg_mean": [], "neg_last": []}
           for s in sites}
    with torch.inference_mode():
        for cls, plist in (("pos", sym), ("neg", neg)):
            for k, p in enumerate(plist):
                enc = tok(render(tok, p), return_tensors="pt").to("cuda")
                model(input_ids=enc["input_ids"])      # prefill only
                for s in sites:
                    mean_v, last_v = tap.buf[s]
                    acc[s][f"{cls}_mean"].append(mean_v)
                    acc[s][f"{cls}_last"].append(last_v)
                if (k + 1) % 6 == 0:
                    print(f"  {cls} {k+1}/{len(plist)}", flush=True)
    tap.detach()

    out = {}
    for s in sites:
        for kind in ("mean", "last"):
            v = (torch.stack(acc[s][f"pos_{kind}"]).mean(0)
                 - torch.stack(acc[s][f"neg_{kind}"]).mean(0)).numpy()
            out[f"v_{kind}_{s}"] = v.astype(np.float32)
        out[f"sigma_{s}"] = np.float32(self_sigma := tap.norm_sum[s] / tap.norm_n[s])
        print(f"site {s} (L{sites[s]}): |v_mean|={np.linalg.norm(out[f'v_mean_{s}']):.2f} "
              f"|v_last|={np.linalg.norm(out[f'v_last_{s}']):.2f} sigma={self_sigma:.1f}",
              flush=True)
    np.savez(os.path.join(args.out_dir, "vectors.npz"), **out)
    json.dump({"sites": meta, "seed": SEED,
               "pos_ids": [p["problem_id"] for p in sym],
               "neg_ids": [p["problem_id"] for p in neg],
               "model": args.model_path, "pooling_primary": "mean"},
              open(os.path.join(args.out_dir, "sites.json"), "w"), indent=2)
    print("vectors.npz + sites.json written", flush=True)


if __name__ == "__main__":
    main()
