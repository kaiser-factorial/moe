#!/usr/bin/env python3
"""Phase 3 Stage 0 pilot: steer with the symbolic-direction vector, measure
routing response (phase3_design.md §4 Stage 0).

Conditions: 2 sites near L17 (post-mamba + post-moe) x alpha in
{-8,-2,-0.5,+0.5,+2,+8}, plus alpha=0 run TWICE (base_a/base_b = noise
floor; greedy decoding so it should be ~0). 6 problems (3 symbolic +
1 factual + 1 computational + 1 creative), greedy, <=256 new tokens
(routing response only — no accuracy claims at this budget).

Injection: h <- h + alpha * sigma_L * v_hat at the site block's output
(every forward call, all positions). v = v_mean from vectors.npz.

Per condition: routing npz per problem (capture_routing format) + text +
mean token logprob (coherence proxy). Ends with a gate summary printed and
written to summary.json: mean decode-token JSD vs base_a (anchors: LoRA
effect 0.0537, between-problem null 0.110), symbolic-specialist occupancy
delta, mean logprob delta.

Usage:
  python phase3_pilot.py --model-path M --vectors /workspace/phase3/vectors.npz \
      [--problems data/problems.json] [--out-dir /workspace/phase3/pilot] \
      [--pilot-depth 17] [--alphas -8 -2 -0.5 0.5 2 8]
"""
import argparse, json, os, sys, time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_routing import (MOE_LAYERS, N_EXPERTS, TOP_K, SEED, ANSWER_SUFFIX,
                             GateRecorder, apply_nemotron_patches, shadow_mamba)

EPS = 1e-10
MAX_NEW = 256
# 25 surviving specialists (outputs/analysis/divergence/specialist_survival.csv)
SPECIALISTS = {  # (layer, expert): base_category
    (10, 8): "factual", (13, 20): "reasoning", (13, 21): "symbolic",
    (13, 52): "creative", (15, 118): "social_ethical", (17, 2): "creative",
    (17, 63): "creative", (17, 70): "social_ethical", (17, 84): "factual",
    (17, 103): "social_ethical", (17, 125): "creative", (20, 40): "factual",
    (20, 67): "reasoning", (20, 81): "social_ethical", (24, 20): "symbolic",
    (24, 26): "reasoning", (27, 8): "factual", (27, 17): "creative",
    (29, 34): "factual", (34, 30): "social_ethical", (34, 35): "computational",
    (36, 50): "creative", (36, 67): "factual", (40, 46): "factual",
    (43, 8): "social_ethical",
}
SYM_SPECIALISTS = [(L, e) for (L, e), c in SPECIALISTS.items() if c == "symbolic"]


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


def pick_problems(problems):
    by_cat = defaultdict(list)
    for p in problems: by_cat[p["category"]].append(p)
    for c in by_cat: by_cat[c].sort(key=lambda p: p["problem_id"])
    sel = by_cat["symbolic"][:3] + [by_cat["factual"][0],
                                    by_cat["computational"][0],
                                    by_cat["creative"][0]]
    return sel


class Steerer:
    """One hook per site; alpha=0 -> pass-through. Vector pre-scaled on GPU."""
    def __init__(self, base, site_layers, vectors):
        self.alpha = {s: 0.0 for s in site_layers}
        self.unit = {}
        self.sigma = {}
        self.handles = []
        for s, L in site_layers.items():
            v = torch.tensor(vectors[f"v_mean_{s}"], device="cuda",
                             dtype=torch.float32)
            self.unit[s] = (v / v.norm()).to(torch.bfloat16)
            self.sigma[s] = float(vectors[f"sigma_{s}"])
            self.handles.append(
                base.backbone.layers[L].register_forward_hook(self._mk(s)))
        print(f"steering hooks on {list(site_layers.items())}; "
              f"sigma={ {s: round(x,1) for s, x in self.sigma.items()} }", flush=True)

    def _mk(self, s):
        def hook(mod, inp, out):
            a = self.alpha[s]
            if a == 0.0: return out
            t = out[0] if isinstance(out, tuple) else out
            t = t + (a * self.sigma[s]) * self.unit[s].to(t.dtype)
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        return hook

    def set(self, site=None, alpha=0.0):
        for s in self.alpha: self.alpha[s] = 0.0
        if site is not None: self.alpha[site] = alpha


def problem_mass(arrs, layer, n_prefill):
    ids, w = arrs[f"ids_{layer}"], arrs[f"w_{layer}"].astype(np.float64)
    ids, w = ids[n_prefill:], w[n_prefill:]
    m = np.zeros(N_EXPERTS)
    np.add.at(m, ids.ravel(), w.ravel())
    t = m.sum()
    return m / t if t > 0 else m


def jsd(p, q):
    p, q = p + EPS, q + EPS
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda x, y: float((x * np.log(x / y)).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--problems", default="data/problems.json")
    ap.add_argument("--out-dir", default="/workspace/phase3/pilot")
    ap.add_argument("--pilot-depth", type=int, default=17)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[-8, -2, -0.5, 0.5, 2, 8])
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    shadow_mamba()
    torch.manual_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    vectors = np.load(args.vectors)
    sites_meta = json.load(open(os.path.join(
        os.path.dirname(args.vectors), "sites.json")))["sites"]
    d = args.pilot_depth
    moe_site = f"e{d}"
    mamba_site = max((n for n in sites_meta if n.startswith("m")
                      and sites_meta[n]["layer"] < d),
                     key=lambda n: sites_meta[n]["layer"])
    site_layers = {moe_site: sites_meta[moe_site]["layer"],
                   mamba_site: sites_meta[mamba_site]["layer"]}

    problems = pick_problems(json.load(open(args.problems))["problems"])
    print("pilot problems:", [p["problem_id"] for p in problems], flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()
    apply_nemotron_patches(model)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model

    steer = Steerer(base, site_layers, vectors)
    rec = GateRecorder(); rec.attach(model)

    conds = [("base_a", None, 0.0), ("base_b", None, 0.0)]
    conds += [(f"{s}_a{a:+g}", s, a) for s in (mamba_site, moe_site)
              for a in args.alphas]

    texts = [render(tok, p) for p in problems]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    B, T = enc["input_ids"].shape
    json.dump(dict(seed=SEED, decoding="greedy", max_new_tokens=MAX_NEW,
                   sites=site_layers, alphas=args.alphas,
                   problem_ids=[p["problem_id"] for p in problems],
                   conditions=[c[0] for c in conds]),
              open(os.path.join(args.out_dir, "run_config.json"), "w"), indent=2)

    store = {}   # cond -> {pid: (arrs, n_prefill)}
    coher = {}   # cond -> mean logprob
    for cond, site, alpha in conds:
        cdir = os.path.join(args.out_dir, cond)
        os.makedirs(cdir, exist_ok=True)
        steer.set(site, alpha)
        rec.reset()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id,
                                 return_dict_in_generate=True, output_scores=True)
        dt = time.time() - t0
        gen = out.sequences[:, T:]
        gen_lens = []
        for i in range(B):
            eos = (gen[i] == tok.eos_token_id).nonzero()
            gen_lens.append(int(eos[0, 0]) + 1 if len(eos) else gen[i].shape[0])
        # mean logprob of chosen tokens (coherence proxy)
        lp_sum = np.zeros(B); lp_n = np.zeros(B)
        for step, sc in enumerate(out.scores):
            logp = torch.log_softmax(sc.float(), dim=-1)
            for i in range(B):
                if step < gen_lens[i]:
                    lp_sum[i] += float(logp[i, gen[i, step]]); lp_n[i] += 1
        per_problem = rec.finalize(B, T, gen_lens)
        store[cond] = {}
        coher[cond] = float((lp_sum / np.maximum(lp_n, 1)).mean())
        with open(os.path.join(cdir, "results.jsonl"), "w") as f:
            for i, p in enumerate(problems):
                pid = p["problem_id"]
                arrs = dict(n_prefill=T, top_k=TOP_K)
                for L, (ids, w) in per_problem[i].items():
                    arrs[f"ids_{L}"] = ids; arrs[f"w_{L}"] = w
                np.savez_compressed(os.path.join(cdir, f"routing_{pid}.npz"), **arrs)
                store[cond][pid] = ({k: v for k, v in arrs.items()
                                     if k.startswith(("ids_", "w_"))}, T)
                text = tok.decode(gen[i][:gen_lens[i]], skip_special_tokens=True)
                f.write(json.dumps(dict(problem_id=pid, condition=cond,
                                        n_gen=gen_lens[i],
                                        mean_logprob=lp_sum[i] / max(lp_n[i], 1),
                                        generated=text)) + "\n")
        del out, gen
        torch.cuda.empty_cache()
        print(f"[{cond}] {sum(gen_lens)} tok in {dt:.0f}s "
              f"mean_logprob={coher[cond]:.3f}", flush=True)

    rec.detach()

    # ── gate summary: every condition vs base_a ─────────────────────────────
    pids = [p["problem_id"] for p in problems]
    sym_pids = pids[:3]
    base_mass = {(pid, L): problem_mass(store["base_a"][pid][0], L, T)
                 for pid in pids for L in MOE_LAYERS}

    def occupancy(cond, pid):
        return sum(problem_mass(store[cond][pid][0], L, T)[e]
                   for L, e in SYM_SPECIALISTS)

    summary = []
    for cond, _, _ in conds:
        if cond == "base_a": continue
        js = np.mean([jsd(problem_mass(store[cond][pid][0], L, T),
                          base_mass[(pid, L)])
                      for pid in pids for L in MOE_LAYERS])
        docc = np.mean([occupancy(cond, pid) - occupancy("base_a", pid)
                        for pid in sym_pids])
        summary.append(dict(condition=cond, mean_jsd_vs_base=round(float(js), 5),
                            d_sym_specialist_occ=round(float(docc), 5),
                            mean_logprob=round(coher[cond], 4),
                            d_logprob=round(coher[cond] - coher["base_a"], 4)))
    print("\n=== PILOT GATE SUMMARY (anchors: noise=base_b, LoRA=0.0537, "
          "null=0.110) ===", flush=True)
    for r in summary:
        print(f"  {r['condition']:>12}: JSD={r['mean_jsd_vs_base']:.4f}  "
              f"dOcc={r['d_sym_specialist_occ']:+.4f}  "
              f"dlogp={r['d_logprob']:+.3f}", flush=True)
    json.dump(dict(anchors=dict(lora_jsd=0.0537, between_problem_null=0.110),
                   base_logprob=coher["base_a"], rows=summary),
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print("summary.json written", flush=True)


if __name__ == "__main__":
    main()
