#!/usr/bin/env python3
"""Phase 1 analysis: expert specialization, difficulty effects, accuracy links.

Inputs:  --log-dir (routing_<pid>.npz + results.jsonl), --problems
Outputs (to --out-dir):
  base_routing_patterns.json     expert activation mass by category/difficulty
  base_expert_specialization.csv per-expert: category shares, selectivity, mass
  base_layer_analysis.csv        per-layer: mean unique experts, entropy, etc.
  per_problem_metrics.csv        per-problem per-layer routing stats
  figures/*.png                  heatmaps, difficulty plots, accuracy plots

Only generated-phase tokens are analyzed by default (prefill excluded) so that
routing reflects the model's own reasoning, not prompt encoding; use
--include-prefill to override.
"""
import argparse, glob, json, os, re
from collections import defaultdict

import numpy as np
import pandas as pd

N_EXPERTS = 128

def load_problem_meta(problems_path):
    probs = json.load(open(problems_path))["problems"]
    return {p["problem_id"]: p for p in probs}

def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--problems", default="data/problems.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--include-prefill", action="store_true")
    ap.add_argument("--prefix", default="base", help="output filename prefix")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)

    meta = load_problem_meta(args.problems)
    results = {}
    rpath = os.path.join(args.log_dir, "results.jsonl")
    # token budgets (to flag truncated generations): from run_config if present
    budgets = {}
    cpath = os.path.join(args.log_dir, "run_config.json")
    if os.path.exists(cpath):
        budgets = json.load(open(cpath)).get("max_tokens", {})
    if os.path.exists(rpath):
        for line in open(rpath):
            r = json.loads(line)
            cap = budgets.get(r.get("category"), None)
            # budget-capped + scored-wrong = ran out of tokens, not a real
            # miss; exclude from accuracy analysis (correct=None) but keep
            # the truncation flag.
            r["truncated"] = bool(cap and r.get("n_gen_tokens", 0) >= cap)
            if r["truncated"] and r.get("correct") is False:
                r["correct"] = None
            results[r["problem_id"]] = r

    # ── accumulate ───────────────────────────────────────────────────────────
    # mass[layer][expert] over whole corpus; cat_mass[layer][category][expert]
    mass = defaultdict(lambda: np.zeros(N_EXPERTS))
    cat_mass = defaultdict(lambda: defaultdict(lambda: np.zeros(N_EXPERTS)))
    per_problem_rows = []
    files = sorted(glob.glob(os.path.join(args.log_dir, "routing_*.npz")))
    if not files:
        raise SystemExit(f"no routing_*.npz in {args.log_dir}")

    for f in files:
        pid = re.sub(r"^routing_|\.npz$", "", os.path.basename(f))
        if pid not in meta:
            continue
        m = meta[pid]
        z = np.load(f)
        n_prefill = int(z["n_prefill"])
        layers = sorted({int(k.split("_")[1]) for k in z.files if k.startswith("ids_")})
        for L in layers:
            ids, w = z[f"ids_{L}"], z[f"w_{L}"].astype(np.float32)
            if not args.include_prefill and ids.shape[0] > n_prefill:
                ids, w = ids[n_prefill:], w[n_prefill:]
            # per-expert cumulative routing weight for this problem/layer
            vec = np.zeros(N_EXPERTS)
            np.add.at(vec, ids.reshape(-1), w.reshape(-1))
            mass[L] += vec
            cat_mass[L][m["category"]] += vec
            p = vec / vec.sum() if vec.sum() else vec
            top1 = int(vec.argmax())
            per_problem_rows.append(dict(
                problem_id=pid, category=m["category"], subtype=m["subtype"],
                difficulty=m["difficulty"], layer=L,
                n_tokens=int(ids.shape[0]),
                unique_experts=int((vec > 0).sum()),
                unique_per_token=float((vec > 0).sum() / max(ids.shape[0], 1)),
                entropy=entropy(p),
                concentration=float(p.max()) if p.sum() else 0.0,
                top1_expert=top1,
                top6_experts=",".join(map(str, np.argsort(vec)[::-1][:6])),
                correct=results.get(pid, {}).get("correct"),
                truncated=results.get(pid, {}).get("truncated"),
                n_gen_tokens=results.get(pid, {}).get("n_gen_tokens"),
            ))

    df = pd.DataFrame(per_problem_rows)
    df.to_csv(os.path.join(args.out_dir, "per_problem_metrics.csv"), index=False)
    layers = sorted(mass.keys())
    cats = sorted({r["category"] for r in per_problem_rows})

    # ── layer-level stats ─────────────────────────────────────────────────────
    layer_rows = []
    for L in layers:
        sub = df[df.layer == L]
        p = mass[L] / mass[L].sum()
        layer_rows.append(dict(
            layer=L, corpus_entropy=entropy(p),
            corpus_active_experts=int((mass[L] > 0).sum()),
            mean_unique_experts=sub.unique_experts.mean(),
            mean_entropy=sub.entropy.mean(),
            mean_concentration=sub.concentration.mean(),
            top1_share=float(p.max()),
        ))
    pd.DataFrame(layer_rows).to_csv(
        os.path.join(args.out_dir, f"{args.prefix}_layer_analysis.csv"), index=False)

    # ── Q1a: expert specialization ───────────────────────────────────────────
    spec_rows = []
    for L in layers:
        cmat = np.stack([cat_mass[L][c] for c in cats])      # (C, 128)
        col = cmat.sum(0)
        active = col > 0
        shares = np.divide(cmat, col, out=np.zeros_like(cmat), where=col > 0)
        for e in np.where(active)[0]:
            row = dict(layer=L, expert=int(e), total_mass=float(col[e]),
                       selectivity=float(shares[:, e].max()),
                       top_category=cats[int(shares[:, e].argmax())])
            row.update({f"share_{c}": float(shares[i, e]) for i, c in enumerate(cats)})
            spec_rows.append(row)
    spec = pd.DataFrame(spec_rows)
    spec.to_csv(os.path.join(args.out_dir, f"{args.prefix}_expert_specialization.csv"),
                index=False)

    # routing patterns json (per category/difficulty mass distributions)
    patterns = {
        "categories": cats, "layers": layers,
        "category_mass": {str(L): {c: cat_mass[L][c].tolist() for c in cats}
                          for L in layers},
    }
    json.dump(patterns, open(os.path.join(
        args.out_dir, f"{args.prefix}_routing_patterns.json"), "w"))

    # ── figures ──────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. specialization heatmap: for 4 representative layers, category x top experts
    rep = [layers[0], layers[len(layers)//3], layers[2*len(layers)//3], layers[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, L in zip(axes.flat, rep):
        cmat = np.stack([cat_mass[L][c] for c in cats])
        cmat = cmat / cmat.sum(axis=1, keepdims=True)        # row-normalize
        top = np.argsort(cmat.sum(0))[::-1][:30]
        im = ax.imshow(cmat[:, top], aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=8)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(top, fontsize=6, rotation=90)
        ax.set_title(f"Layer {L} — share of category mass on top-30 experts")
        plt.colorbar(im, ax=ax, fraction=0.025)
    fig.suptitle("Expert <-> problem-type routing (generated tokens)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "figures", f"{args.prefix}_heatmap_specialization.png"), dpi=150)
    plt.close(fig)

    # 2. selectivity distribution by layer
    fig, ax = plt.subplots(figsize=(12, 5))
    sel = [spec[spec.layer == L].selectivity for L in layers]
    ax.boxplot(sel, tick_labels=[str(L) for L in layers])
    ax.axhline(1/len(cats), color="r", ls="--", lw=1,
               label=f"uniform ({1/len(cats):.2f})")
    ax.set_xlabel("MoE layer"); ax.set_ylabel("expert selectivity (max category share)")
    ax.set_title("How category-specialized are experts, by depth?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "figures", f"{args.prefix}_selectivity_by_layer.png"), dpi=150)
    plt.close(fig)

    # 3. Q1b: difficulty effects (scored categories only)
    dmap = {"easy": 0, "medium": 1, "hard": 2}
    sub = df[df.difficulty.isin(dmap)].copy()
    sub["dnum"] = sub.difficulty.map(dmap)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, label in [
            (axes[0], "unique_per_token", "unique experts / token"),
            (axes[1], "entropy", "routing entropy"),
            (axes[2], "concentration", "top-1 concentration")]:
        g = sub.groupby(["category", "difficulty"])[metric].mean().unstack()
        g = g[[c for c in ["easy", "medium", "hard"] if c in g.columns]]
        g.plot(kind="bar", ax=ax, rot=20)
        ax.set_ylabel(label); ax.set_title(f"{label} by difficulty")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "figures", f"{args.prefix}_difficulty_effects.png"), dpi=150)
    plt.close(fig)

    # 4. Q1c: routing diversity vs accuracy
    scored = df[df.correct.notna()].copy()
    qc = {}
    if len(scored):
        agg = scored.groupby("problem_id").agg(
            entropy=("entropy", "mean"), unique=("unique_per_token", "mean"),
            conc=("concentration", "mean"), correct=("correct", "first"),
            category=("category", "first")).reset_index()
        agg["correct"] = agg.correct.astype(bool)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, m, lbl in [(axes[0], "entropy", "mean routing entropy"),
                           (axes[1], "unique", "mean unique experts/token"),
                           (axes[2], "conc", "mean concentration")]:
            for ok, marker, color in [(True, "o", "tab:green"), (False, "x", "tab:red")]:
                s = agg[agg.correct == ok]
                ax.scatter(s[m], np.random.RandomState(0).normal(
                    ok * 1.0, 0.05, len(s)), marker=marker, c=color, alpha=0.6,
                    label="correct" if ok else "incorrect")
            from scipy.stats import pointbiserialr
            r, pval = pointbiserialr(agg.correct, agg[m])
            qc[m] = dict(pointbiserial_r=float(r), p=float(pval))
            ax.set_xlabel(lbl); ax.set_yticks([0, 1])
            ax.set_yticklabels(["incorrect", "correct"])
            ax.set_title(f"{lbl}\nr={r:.2f}, p={pval:.3f}")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "figures", f"{args.prefix}_accuracy_vs_routing.png"), dpi=150)
        plt.close(fig)
        json.dump(qc, open(os.path.join(args.out_dir, f"{args.prefix}_accuracy_correlations.json"), "w"), indent=2)

    # summary to stdout
    print(json.dumps(dict(
        n_problems=df.problem_id.nunique(), n_layers=len(layers),
        mean_unique_experts=float(df.unique_experts.mean()),
        mean_entropy=float(df.entropy.mean()),
        accuracy=float(pd.Series([r.get("correct") for r in results.values()
                                  if r.get("correct") is not None]).mean())
        if results else None,
        accuracy_routing_correlations=qc), indent=2))

if __name__ == "__main__":
    main()
