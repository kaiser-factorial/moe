#!/usr/bin/env python3
"""Phase 2: base vs LoRA routing divergence (SPEC §4 Phase 2).

For each problem and MoE layer, builds per-expert routing-mass distributions
(generation tokens only) for both models and computes:
  - Jaccard similarity of top-K expert sets (K=6 primary, K=16 sensitivity)
  - Jensen-Shannon divergence (bounded sym. KL, nats) + eps-smoothed KL(b||l)
  - entropy / top-1 concentration deltas (lora - base)

Null reference: same metrics between DIFFERENT problems of the same category
within the base model — "how far apart are two unrelated base runs" — so
LoRA-induced shifts have a scale.

Outputs (--out-dir):
  routing_divergence_symbolic_subset.csv   per layer, symbolic problems only
  routing_divergence_cross_domain.csv      per category x layer + summary rows
  pattern_correlation.csv                  per-layer specialization correlation
  divergence_null_reference.csv            within-category base-base null
  figures/divergence_by_layer.png          JSD over depth, symbolic vs rest
  figures/jaccard_by_category.png          top-6 Jaccard per category vs null
  figures/specialist_survival.png          base vs lora selectivity scatter
  figures/pattern_correlation_by_layer.png
"""
import argparse, glob, json, os
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

MOE_LAYERS = [1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51]
N_EXPERTS = 128
EPS = 1e-10

def problem_mass(npz, layer):
    """(128,) normalized routing-mass over generation tokens."""
    ids, w = npz[f"ids_{layer}"], npz[f"w_{layer}"].astype(np.float64)
    s = int(npz["n_prefill"])
    ids, w = ids[s:], w[s:]
    m = np.zeros(N_EXPERTS)
    np.add.at(m, ids.ravel(), w.ravel())
    t = m.sum()
    return m / t if t > 0 else m

def jaccard_topk(a, b, k):
    ta, tb = set(np.argsort(a)[-k:]), set(np.argsort(b)[-k:])
    return len(ta & tb) / len(ta | tb)

def jsd(p, q):
    p, q = p + EPS, q + EPS
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda x, y: float((x * np.log(x / y)).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def kl_smoothed(p, q):
    p, q = p + EPS, q + EPS
    p, q = p / p.sum(), q / q.sum()
    return float((p * np.log(p / q)).sum())

def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0

def load_all(log_dir):
    out = {}
    for f in glob.glob(os.path.join(log_dir, "routing_*.npz")):
        pid = os.path.basename(f)[len("routing_"):-len(".npz")]
        out[pid] = np.load(f)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="outputs/logs/base")
    ap.add_argument("--lora-dir", default="outputs/logs/lora")
    ap.add_argument("--problems", default="data/problems.json")
    ap.add_argument("--out-dir", default="outputs/analysis/divergence")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)

    meta = {p["problem_id"]: p for p in json.load(open(args.problems))["problems"]}
    base, lora = load_all(args.base_dir), load_all(args.lora_dir)
    pids = sorted(set(base) & set(lora))
    print(f"{len(pids)} paired problems")

    # ── per-problem, per-layer divergence ────────────────────────────────────
    rows = []
    mass_cache = {}  # (model, pid, layer) -> mass
    for pid in pids:
        cat = meta[pid]["category"]
        for L in MOE_LAYERS:
            mb = problem_mass(base[pid], L)
            ml = problem_mass(lora[pid], L)
            mass_cache[("b", pid, L)] = mb
            rows.append({
                "problem_id": pid, "category": cat, "layer": L,
                "jaccard_top6": jaccard_topk(mb, ml, 6),
                "jaccard_top16": jaccard_topk(mb, ml, 16),
                "jsd": jsd(mb, ml), "kl_base_lora": kl_smoothed(mb, ml),
                "d_entropy": entropy(ml) - entropy(mb),
                "d_concentration": float(ml.max() - mb.max()),
            })
    df = pd.DataFrame(rows)

    # ── null: base-vs-base, different problems, same category ────────────────
    nrows = []
    rng = np.random.default_rng(42)
    bycat = defaultdict(list)
    for pid in pids:
        bycat[meta[pid]["category"]].append(pid)
    for cat, plist in bycat.items():
        pairs = list(combinations(plist, 2))
        if len(pairs) > 60:
            pairs = [pairs[i] for i in rng.choice(len(pairs), 60, replace=False)]
        for p1, p2 in pairs:
            for L in MOE_LAYERS:
                m1, m2 = mass_cache[("b", p1, L)], mass_cache[("b", p2, L)]
                nrows.append({"category": cat, "layer": L,
                              "jaccard_top6": jaccard_topk(m1, m2, 6),
                              "jsd": jsd(m1, m2)})
    nul = pd.DataFrame(nrows)
    nul.groupby(["category", "layer"]).mean().reset_index() \
       .to_csv(os.path.join(args.out_dir, "divergence_null_reference.csv"), index=False)

    # ── outputs: symbolic subset + cross-domain ──────────────────────────────
    sym = df[df.category == "symbolic"].groupby("layer").agg(
        n=("problem_id", "count"), jaccard_top6=("jaccard_top6", "mean"),
        jaccard_top16=("jaccard_top16", "mean"), jsd=("jsd", "mean"),
        kl_base_lora=("kl_base_lora", "mean"), d_entropy=("d_entropy", "mean"),
        d_concentration=("d_concentration", "mean")).reset_index()
    sym.to_csv(os.path.join(args.out_dir, "routing_divergence_symbolic_subset.csv"), index=False)

    cross = df.groupby(["category", "layer"]).agg(
        jaccard_top6=("jaccard_top6", "mean"), jsd=("jsd", "mean"),
        d_entropy=("d_entropy", "mean"),
        d_concentration=("d_concentration", "mean")).reset_index()
    cross.to_csv(os.path.join(args.out_dir, "routing_divergence_cross_domain.csv"), index=False)

    # ── Q2c: specialization pattern correlation ──────────────────────────────
    cats = sorted(bycat)
    # category-normalized expert x category mass matrices per layer per model
    def spec_matrix(model_logs):
        M = {L: np.zeros((N_EXPERTS, len(cats))) for L in MOE_LAYERS}
        for pid in pids:
            ci = cats.index(meta[pid]["category"])
            for L in MOE_LAYERS:
                m = problem_mass(model_logs[pid], L)
                M[L][:, ci] += m
        for L in MOE_LAYERS:  # normalize each category column to unit mass
            colsum = M[L].sum(axis=0, keepdims=True)
            M[L] = M[L] / np.where(colsum > 0, colsum, 1)
        return M
    Mb, Ml = spec_matrix(base), spec_matrix(lora)
    prows = []
    for L in MOE_LAYERS:
        b, l = Mb[L].ravel(), Ml[L].ravel()
        r = float(np.corrcoef(b, l)[0, 1])
        # per-expert selectivity (share-normalized rows, max over cats)
        def selec(M):
            rs = M / np.where(M.sum(1, keepdims=True) > 0, M.sum(1, keepdims=True), 1)
            return rs.max(1)
        sb, sl = selec(Mb[L]), selec(Ml[L])
        prows.append({"layer": L, "pattern_pearson_r": r,
                      "selectivity_pearson_r": float(np.corrcoef(sb, sl)[0, 1]),
                      "n_specialists_base": int((sb > 0.8).sum()),
                      "n_specialists_lora": int((sl > 0.8).sum())})
    pc = pd.DataFrame(prows)
    pc.to_csv(os.path.join(args.out_dir, "pattern_correlation.csv"), index=False)

    # specialist survival detail (base selectivity > 0.8)
    surv = []
    for L in MOE_LAYERS:
        def shares(M):
            return M / np.where(M.sum(1, keepdims=True) > 0, M.sum(1, keepdims=True), 1)
        sb, sl = shares(Mb[L]), shares(Ml[L])
        for e in range(N_EXPERTS):
            if sb[e].max() > 0.8 and Mb[L][e].sum() > 1e-6:
                surv.append({"layer": L, "expert": e,
                             "base_selectivity": float(sb[e].max()),
                             "base_category": cats[int(sb[e].argmax())],
                             "lora_selectivity": float(sl[e].max()),
                             "lora_category": cats[int(sl[e].argmax())],
                             "same_category": cats[int(sb[e].argmax())] == cats[int(sl[e].argmax())]})
    sv = pd.DataFrame(surv)
    sv.to_csv(os.path.join(args.out_dir, "specialist_survival.csv"), index=False)

    # ── figures ──────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for cat, sub in df.groupby("category"):
        g = sub.groupby("layer")["jsd"].mean()
        ax.plot(g.index, g.values, marker="o", ms=3,
                lw=2.2 if cat == "symbolic" else 1.0,
                label=cat, alpha=1.0 if cat == "symbolic" else 0.65)
    nl = nul.groupby("layer")["jsd"].mean()
    ax.plot(nl.index, nl.values, "k--", lw=1, label="null (base, diff. problems)")
    ax.set_xlabel("MoE layer"); ax.set_ylabel("JSD base↔LoRA (nats)")
    ax.set_title("Routing divergence by depth"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "figures/divergence_by_layer.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    cat_j = df.groupby("category")["jaccard_top6"].mean().sort_values()
    nul_j = nul.groupby("category")["jaccard_top6"].mean()
    x = np.arange(len(cat_j))
    ax.bar(x - 0.2, cat_j.values, 0.4, label="base↔LoRA (same problem)")
    ax.bar(x + 0.2, [nul_j[c] for c in cat_j.index], 0.4, label="base↔base (diff. problems)")
    ax.set_xticks(x); ax.set_xticklabels(cat_j.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("top-6 Jaccard"); ax.set_title("Expert-set overlap per category")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "figures/jaccard_by_category.png"), dpi=150)

    if len(sv):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        col = sv.same_category.map({True: "tab:blue", False: "tab:red"})
        ax.scatter(sv.base_selectivity, sv.lora_selectivity, c=col, s=28)
        ax.plot([0.5, 1.0], [0.5, 1.0], "k--", lw=0.8)
        ax.set_xlabel("base selectivity"); ax.set_ylabel("LoRA selectivity")
        ax.set_title("Near-pure specialists (>0.8): survival under LoRA\n(red = flipped category)")
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "figures/specialist_survival.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pc.layer, pc.pattern_pearson_r, marker="o", label="expert×category pattern r")
    ax.plot(pc.layer, pc.selectivity_pearson_r, marker="s", ms=4, label="per-expert selectivity r")
    ax.set_xlabel("MoE layer"); ax.set_ylabel("Pearson r (base vs LoRA)")
    ax.set_ylim(0, 1.02); ax.set_title("Specialization pattern preservation")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "figures/pattern_correlation_by_layer.png"), dpi=150)

    # ── console summary ──────────────────────────────────────────────────────
    print("\n=== summary ===")
    for cat in cats:
        s = df[df.category == cat]
        n = nul[nul.category == cat]
        print(f"{cat:15} JSD {s.jsd.mean():.4f} (null {n.jsd.mean():.4f}) "
              f"J6 {s.jaccard_top6.mean():.3f} (null {n.jaccard_top6.mean():.3f}) "
              f"dConc {s.d_concentration.mean():+.4f}")
    print(f"\npattern r: mean {pc.pattern_pearson_r.mean():.3f} "
          f"min {pc.pattern_pearson_r.min():.3f} @L{int(pc.loc[pc.pattern_pearson_r.idxmin(),'layer'])}")
    print(f"specialists: base {pc.n_specialists_base.sum()} -> lora {pc.n_specialists_lora.sum()}; "
          f"survived same-cat {int(sv.same_category.sum())}/{len(sv)}")

if __name__ == "__main__":
    main()
