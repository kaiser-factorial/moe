#!/usr/bin/env python3
"""Offline (CPU-only) extended analyses on existing Phase 1/2/3 logs.

Four independent analyses, no GPU / no RunPod required:
  A. subtype  — is routing semantic (splits by subtype) or surface/coarse?
  B. temporal — how routing concentration evolves across a generation
  C. coact    — within-layer expert co-activation "teams" + cross-layer pathways
  D. robust   — bootstrap CIs + permutation/FDR on specialist claims;
                geometry of the 16 banked Phase-3 steering vectors

Inputs (all local): outputs/logs/base/, data/problems.json,
                    outputs/analysis/steering/vectors.npz + sites.json
Outputs: outputs/analysis/extended/{figures,*.json,*.csv}

Usage: python scripts/extended_analysis.py [A B C D]   (default: all)
"""
import sys, os, json, glob, re, itertools
from collections import defaultdict
import numpy as np

N_EXPERTS = 128
MOE_LAYERS = [1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51]
MID = [13,15,17,20,22,24]          # mid-network band (Phase 1 selectivity peak)
LOG = "outputs/logs/base"
OUT = "outputs/analysis/extended"
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
rng = np.random.default_rng(42)

def jsd(p, q):
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0

# ── shared loader: per-problem per-layer mass vectors (generated tokens) ──────
def load(meta):
    """Return dict pid -> {layer -> mass vec (128,)} over generated tokens,
    plus dict pid -> per-token arrays for temporal analysis."""
    massvecs, tokstreams = {}, {}
    for f in sorted(glob.glob(os.path.join(LOG, "routing_*.npz"))):
        pid = re.sub(r"^routing_|\.npz$", "", os.path.basename(f))
        if pid not in meta:
            continue
        z = np.load(f)
        npre = int(z["n_prefill"])
        per_layer, per_layer_tok = {}, {}
        for L in MOE_LAYERS:
            ids, w = z[f"ids_{L}"], z[f"w_{L}"].astype(np.float32)
            if ids.shape[0] > npre:
                ids, w = ids[npre:], w[npre:]      # generated only
            vec = np.zeros(N_EXPERTS)
            np.add.at(vec, ids.reshape(-1), w.reshape(-1))
            per_layer[L] = vec
            per_layer_tok[L] = (ids, w)
        massvecs[pid] = per_layer
        tokstreams[pid] = per_layer_tok
    return massvecs, tokstreams

# ── A. subtype specialization ────────────────────────────────────────────────
def analysis_A(meta, massvecs):
    print("[A] subtype specialization")
    out = {}
    for cat in ["symbolic", "reasoning", "factual"]:
        pids = [p for p in meta if meta[p]["category"] == cat and p in massvecs]
        # mid-network concatenated, normalized routing distribution per problem
        def dist(pid):
            v = np.concatenate([massvecs[pid][L] for L in MID])
            return v / v.sum() if v.sum() else v
        D = {p: dist(p) for p in pids}
        sub = {p: meta[p]["subtype"] for p in pids}
        within, between = [], []
        for a, b in itertools.combinations(pids, 2):
            d = jsd(D[a], D[b])
            (within if sub[a] == sub[b] else between).append(d)
        # permutation test on the within-vs-between gap (shuffle subtype labels)
        obs = np.mean(between) - np.mean(within)
        labels = np.array([sub[p] for p in pids])
        pair_idx = list(itertools.combinations(range(len(pids)), 2))
        dmat = np.array([jsd(D[pids[i]], D[pids[j]]) for i, j in pair_idx])
        perm = []
        for _ in range(2000):
            sh = rng.permutation(labels)
            same = np.array([sh[i] == sh[j] for i, j in pair_idx])
            perm.append(dmat[~same].mean() - dmat[same].mean())
        perm = np.array(perm)
        pval = float((perm >= obs).mean())
        out[cat] = dict(n=len(pids), within_jsd=float(np.mean(within)),
                        between_jsd=float(np.mean(between)),
                        gap=float(obs), perm_p=pval,
                        ratio=float(np.mean(within) / np.mean(between)))
        print(f"  {cat}: within={out[cat]['within_jsd']:.4f} "
              f"between={out[cat]['between_jsd']:.4f} gap={obs:.4f} p={pval:.4f}")
    json.dump(out, open(os.path.join(OUT, "A_subtype.json"), "w"), indent=2)

    # figure: symbolic problem-problem JSD matrix ordered by subtype
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cat = "symbolic"
    pids = sorted([p for p in meta if meta[p]["category"] == cat and p in massvecs],
                  key=lambda p: meta[p]["subtype"])
    def dist(pid):
        v = np.concatenate([massvecs[pid][L] for L in MID]); return v/v.sum()
    M = np.array([[jsd(dist(a), dist(b)) for b in pids] for a in pids])
    subs = [meta[p]["subtype"] for p in pids]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(M, cmap="viridis_r")
    ax.set_xticks(range(len(pids))); ax.set_yticks(range(len(pids)))
    ax.set_xticklabels(subs, rotation=90, fontsize=6)
    ax.set_yticklabels(subs, fontsize=6)
    ax.set_title("Symbolic: pairwise routing JSD (mid-network)\n"
                 "dark blocks on diagonal = subtypes route alike")
    plt.colorbar(im, ax=ax, fraction=0.046, label="JSD (lower=more similar)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "A_symbolic_subtype_jsd.png"), dpi=150)
    plt.close(fig)
    return out

# ── B. temporal routing dynamics ─────────────────────────────────────────────
def analysis_B(meta, tokstreams, results):
    print("[B] temporal routing dynamics")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    NB = 10
    # per category: mean concentration & entropy & unique/tok by generation decile
    cats = sorted({meta[p]["category"] for p in tokstreams})
    conc = {c: np.zeros(NB) for c in cats}; ent = {c: np.zeros(NB) for c in cats}
    cnt = {c: np.zeros(NB) for c in cats}
    for pid, per_layer_tok in tokstreams.items():
        cat = meta[pid]["category"]
        # aggregate across mid-network layers, per token
        ids = np.concatenate([per_layer_tok[L][0] for L in MID], axis=1)  # (T, 6*|MID|)
        w = np.concatenate([per_layer_tok[L][1] for L in MID], axis=1)
        T = ids.shape[0]
        if T < NB: continue
        for t in range(T):
            vec = np.zeros(N_EXPERTS); np.add.at(vec, ids[t], w[t])
            p = vec / vec.sum() if vec.sum() else vec
            b = min(int(t / T * NB), NB - 1)
            conc[cat][b] += float(p.max()); ent[cat][b] += entropy(p); cnt[cat][b] += 1
    for c in cats:
        conc[c] /= np.maximum(cnt[c], 1); ent[c] /= np.maximum(cnt[c], 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    x = (np.arange(NB) + 0.5) / NB * 100
    for c in cats:
        a1.plot(x, conc[c], marker="o", ms=3, label=c)
        a2.plot(x, ent[c], marker="o", ms=3, label=c)
    a1.set_xlabel("position in generation (%)"); a1.set_ylabel("top-1 concentration")
    a1.set_title("Routing concentration over generation (mid-network)")
    a2.set_xlabel("position in generation (%)"); a2.set_ylabel("entropy (nats)")
    a2.set_title("Routing entropy over generation")
    a1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "B_temporal.png"), dpi=150)
    plt.close(fig)
    out = {c: dict(conc_first=float(conc[c][0]), conc_last=float(conc[c][-1]),
                   ent_first=float(ent[c][0]), ent_last=float(ent[c][-1])) for c in cats}
    json.dump(out, open(os.path.join(OUT, "B_temporal.json"), "w"), indent=2)
    for c in cats:
        print(f"  {c}: conc {out[c]['conc_first']:.3f}->{out[c]['conc_last']:.3f} "
              f"ent {out[c]['ent_first']:.2f}->{out[c]['ent_last']:.2f}")
    return out

# ── C. expert co-activation & cross-layer pathways ───────────────────────────
def analysis_C(meta, tokstreams):
    print("[C] co-activation & pathways")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # within-layer co-occurrence at a representative mid layer (17)
    L = 17
    C = np.zeros((N_EXPERTS, N_EXPERTS))
    for pid, plt_tok in tokstreams.items():
        ids = plt_tok[L][0]
        for row in ids:
            u = np.unique(row)
            for a, b in itertools.combinations(u, 2):
                C[a, b] += 1; C[b, a] += 1
    # greedy correlation clustering on the top-active experts
    active = np.where(C.sum(0) > 0)[0]
    sub = C[np.ix_(active, active)]
    # normalize to co-occurrence rate
    deg = sub.sum(1, keepdims=True); P = sub / np.maximum(deg, 1)
    # order by a 1-D spectral embedding (Fiedler vector) for a block-y picture
    Dg = np.diag(sub.sum(1)); Lap = Dg - sub
    try:
        evals, evecs = np.linalg.eigh(Lap)
        order = np.argsort(evecs[:, 1])
    except Exception:
        order = np.argsort(-sub.sum(1))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.log1p(sub[np.ix_(order, order)]), cmap="magma")
    ax.set_title(f"Layer {L}: expert co-activation (log counts, spectral order)\n"
                 "blocks = experts that fire together")
    ax.set_xlabel("expert (reordered)"); ax.set_ylabel("expert (reordered)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "C_coactivation_L17.png"), dpi=150)
    plt.close(fig)

    # cross-layer pathway: correlation of per-problem top1 expert identity
    # (do problems that pick specialist X at layer L also pick a consistent
    #  expert at the next MoE layer?) -> measure via normalized mutual info
    from math import log
    def nmi(a, b):
        a = np.asarray(a); b = np.asarray(b)
        ua, ub = np.unique(a), np.unique(b)
        n = len(a)
        Hxy = 0.0; Hx = 0.0; Hy = 0.0
        for x in ua:
            px = (a == x).mean(); Hx -= px * log(px)
        for y in ub:
            py = (b == y).mean(); Hy -= py * log(py)
        I = 0.0
        for x in ua:
            for y in ub:
                pxy = ((a == x) & (b == y)).mean()
                if pxy > 0:
                    I += pxy * log(pxy / ((a == x).mean() * (b == y).mean()))
        denom = (Hx + Hy) / 2
        return I / denom if denom > 0 else 0.0
    pids = [p for p in tokstreams]
    # top-1 expert per problem per layer
    top1 = {}
    for L in MOE_LAYERS:
        col = []
        for p in pids:
            vec = np.zeros(N_EXPERTS)
            np.add.at(vec, tokstreams[p][L][0].reshape(-1), tokstreams[p][L][1].reshape(-1))
            col.append(int(vec.argmax()))
        top1[L] = col
    nmis = []
    for i in range(len(MOE_LAYERS) - 1):
        L0, L1 = MOE_LAYERS[i], MOE_LAYERS[i+1]
        nmis.append((L0, L1, nmi(top1[L0], top1[L1])))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar([f"{a}→{b}" for a, b, _ in nmis], [v for _, _, v in nmis])
    ax.set_ylabel("NMI of top-1 expert identity")
    ax.set_title("Cross-layer routing coupling (adjacent MoE layers)\n"
                 "high = a problem's chosen expert predicts the next layer's choice")
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "C_crosslayer_nmi.png"), dpi=150)
    plt.close(fig)
    out = dict(mean_adjacent_nmi=float(np.mean([v for _, _, v in nmis])),
               coact_layer=L, n_active_experts_L=int(len(active)))
    json.dump(out, open(os.path.join(OUT, "C_coactivation.json"), "w"), indent=2)
    print(f"  mean adjacent-layer NMI={out['mean_adjacent_nmi']:.3f}; "
          f"L{L} active experts={out['n_active_experts_L']}")
    return out

# ── D. robustness + steering-vector geometry ─────────────────────────────────
def analysis_D(meta, massvecs):
    print("[D] robustness + vector geometry")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cats = sorted({meta[p]["category"] for p in massvecs})
    pids = list(massvecs)
    # per-layer category-mass matrix
    def cat_mass(L):
        m = np.zeros((len(cats), N_EXPERTS))
        for p in pids:
            m[cats.index(meta[p]["category"])] += massvecs[p][L]
        return m
    # observed specialists (sel>0.8) and an FDR-style permutation null
    specialists = []
    nulls = defaultdict(list)
    for L in MOE_LAYERS:
        m = cat_mass(L); m = m / np.maximum(m.sum(1, keepdims=True), 1e-9)
        tot = m.sum(0); active = tot > 0
        sel = np.where(active, (m / np.maximum(tot, 1e-9)).max(0), 0)
        for e in np.where(active)[0]:
            if sel[e] > 0.8:
                specialists.append((L, int(e), float(sel[e]), cats[int(m[:, e].argmax())]))
    # permutation null: shuffle category labels across problems, recount sel>0.8
    obs_n = len(specialists)
    perm_counts = []
    labels = np.array([cats.index(meta[p]["category"]) for p in pids])
    for _ in range(500):
        sh = rng.permutation(labels)
        cnt = 0
        for L in MOE_LAYERS:
            m = np.zeros((len(cats), N_EXPERTS))
            for k, p in enumerate(pids):
                m[sh[k]] += massvecs[p][L]
            m = m / np.maximum(m.sum(1, keepdims=True), 1e-9)
            tot = m.sum(0); active = tot > 0
            sel = np.where(active, (m / np.maximum(tot, 1e-9)).max(0), 0)
            cnt += int((sel[active] > 0.8).sum())
        perm_counts.append(cnt)
    perm_counts = np.array(perm_counts)
    p_spec = float((perm_counts >= obs_n).mean())
    # bootstrap CI on median mid-network selectivity
    midsel = []
    for L in MID:
        m = cat_mass(L); m = m / np.maximum(m.sum(1, keepdims=True), 1e-9)
        tot = m.sum(0); active = tot > 0
        midsel.append(np.where(active, (m / np.maximum(tot, 1e-9)).max(0), 0)[active])
    midsel = np.concatenate(midsel)
    boot = [np.median(rng.choice(midsel, len(midsel), replace=True)) for _ in range(2000)]
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    rob = dict(observed_specialists=obs_n, perm_null_mean=float(perm_counts.mean()),
               perm_null_max=int(perm_counts.max()), specialists_p=p_spec,
               mid_selectivity_median=float(np.median(midsel)), mid_sel_CI95=ci,
               uniform_baseline=1/len(cats))
    json.dump(rob, open(os.path.join(OUT, "D_robustness.json"), "w"), indent=2)
    print(f"  specialists obs={obs_n} null≈{perm_counts.mean():.1f} "
          f"(max {perm_counts.max()}) p={p_spec:.3f}; "
          f"mid sel median={np.median(midsel):.3f} CI{ci}")

    # steering-vector geometry
    vp = "outputs/analysis/steering/vectors.npz"
    geo = {}
    if os.path.exists(vp):
        z = np.load(vp); sites = json.load(open("outputs/analysis/steering/sites.json"))["sites"]
        keys = [k for k in z.files if k.startswith("v_mean_")]
        names = [k.replace("v_mean_", "") for k in keys]
        V = np.stack([z[k] / (np.linalg.norm(z[k]) + 1e-9) for k in keys])
        S = V @ V.T
        order = sorted(range(len(names)),
                       key=lambda i: (sites.get(names[i], {}).get("family", ""),
                                      sites.get(names[i], {}).get("layer", 0)))
        So = S[np.ix_(order, order)]; no = [names[i] for i in order]
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(So, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(no))); ax.set_yticks(range(len(no)))
        ax.set_xticklabels(no, rotation=90, fontsize=7); ax.set_yticklabels(no, fontsize=7)
        ax.set_title("Steering-vector cosine similarity (16 sites)\n"
                     "symbolic-vs-rest direction; m=post-Mamba e=post-MoE")
        plt.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures", "D_vector_cosine.png"), dpi=150)
        plt.close(fig)
        # mean within-family vs cross-family cosine
        fam = [sites.get(n, {}).get("family", "?") for n in names]
        wf, cf = [], []
        for i, j in itertools.combinations(range(len(names)), 2):
            (wf if fam[i] == fam[j] else cf).append(S[i, j])
        geo = dict(mean_cos_within_family=float(np.mean(wf)),
                   mean_cos_cross_family=float(np.mean(cf)),
                   n_sites=len(names))
        json.dump(geo, open(os.path.join(OUT, "D_vector_geometry.json"), "w"), indent=2)
        print(f"  vectors: within-family cos={geo['mean_cos_within_family']:.2f} "
              f"cross-family cos={geo['mean_cos_cross_family']:.2f}")
    return dict(robustness=rob, geometry=geo)

# ── driver ───────────────────────────────────────────────────────────────────
def main():
    which = [a.upper() for a in sys.argv[1:]] or ["A", "B", "C", "D"]
    meta = {p["problem_id"]: p for p in json.load(open("data/problems.json"))["problems"]}
    results = {json.loads(l)["problem_id"]: json.loads(l) for l in open(os.path.join(LOG, "results.jsonl"))}
    massvecs, tokstreams = load(meta)
    print(f"loaded {len(massvecs)} problems")
    if "A" in which: analysis_A(meta, massvecs)
    if "B" in which: analysis_B(meta, tokstreams, results)
    if "C" in which: analysis_C(meta, tokstreams)
    if "D" in which: analysis_D(meta, massvecs)
    print("done.")

if __name__ == "__main__":
    main()
