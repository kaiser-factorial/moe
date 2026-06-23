#!/usr/bin/env python3
"""Name the expert 'teams': community detection on the within-layer
co-activation graph, characterized by problem-category usage and specialists.

For each of several MoE layers:
  - build expert-expert co-occurrence (two experts co-fire on the same token)
  - normalize to a co-activation rate, drop weak edges, greedy-modularity
    communities
  - per community: size, total routing mass, category fingerprint (which
    problem categories drive it), and how many Phase-1 specialists it holds
Outputs: outputs/analysis/extended/E_communities.json + figures.
"""
import os, json, glob, re, itertools
from collections import defaultdict
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

N_EXPERTS = 128
LOG = "outputs/logs/base"
OUT = "outputs/analysis/extended"
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
LAYERS = [8, 17, 24, 38]          # spread across depth
meta = {p["problem_id"]: p for p in json.load(open("data/problems.json"))["problems"]}
cats = sorted({m["category"] for m in meta.values()})

# load per-token ids + per-problem category, and category-mass per expert
streams = {}
for f in sorted(glob.glob(os.path.join(LOG, "routing_*.npz"))):
    pid = re.sub(r"^routing_|\.npz$", "", os.path.basename(f))
    if pid not in meta: continue
    z = np.load(f); npre = int(z["n_prefill"])
    streams[pid] = (z, npre)

def cooccur_and_catmass(L):
    C = np.zeros((N_EXPERTS, N_EXPERTS))
    catmass = np.zeros((len(cats), N_EXPERTS))
    for pid, (z, npre) in streams.items():
        ids = z[f"ids_{L}"]; w = z[f"w_{L}"].astype(np.float32)
        if ids.shape[0] > npre: ids, w = ids[npre:], w[npre:]
        ci = cats.index(meta[pid]["category"])
        np.add.at(catmass[ci], ids.reshape(-1), w.reshape(-1))
        for row in ids:
            u = np.unique(row)
            for a, b in itertools.combinations(u, 2):
                C[a, b] += 1; C[b, a] += 1
    return C, catmass

# Phase-1 specialists (sel>0.8) for cross-reference
spec_by_layer = defaultdict(dict)
specf = "outputs/analysis/base/base_expert_specialization.csv"
if os.path.exists(specf):
    import csv
    for r in csv.DictReader(open(specf)):
        if float(r["selectivity"]) > 0.8:
            spec_by_layer[int(r["layer"])][int(r["expert"])] = r["top_category"]

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
results = {}
fig, axes = plt.subplots(1, len(LAYERS), figsize=(5*len(LAYERS), 5.2))
for ax, L in zip(axes, LAYERS):
    C, catmass = cooccur_and_catmass(L)
    deg = C.sum(1)
    active = np.where(deg > 0)[0]
    # co-activation rate: P(b in top6 | a in top6) symmetrized; threshold edges
    rate = C / np.maximum(deg[:, None], 1)
    rate = 0.5 * (rate + rate.T)
    G = nx.Graph()
    G.add_nodes_from(active.tolist())
    thr = np.percentile(rate[active][:, active][rate[active][:, active] > 0], 70)
    for i in active:
        for j in active:
            if i < j and rate[i, j] >= thr:
                G.add_edge(int(i), int(j), weight=float(rate[i, j]))
    comms = list(greedy_modularity_communities(G, weight="weight"))
    comms = [c for c in comms if len(c) >= 3]
    catmass_n = catmass / np.maximum(catmass.sum(1, keepdims=True), 1e-9)  # per-cat normalize
    layer_rec = []
    for k, com in enumerate(comms):
        members = sorted(com)
        mass = float(catmass[:, members].sum())
        # category fingerprint: share of this community's mass by category
        cmass = np.array([catmass[ci, members].sum() for ci in range(len(cats))])
        cshare = cmass / max(cmass.sum(), 1e-9)
        top = cats[int(cshare.argmax())]
        n_spec = sum(1 for e in members if e in spec_by_layer.get(L, {}))
        layer_rec.append(dict(size=len(members), mass=mass,
                              top_category=top, top_share=float(cshare.max()),
                              cat_shares={cats[i]: float(cshare[i]) for i in range(len(cats))},
                              n_specialists=n_spec,
                              example_experts=members[:8]))
    results[str(L)] = dict(n_communities=len(comms),
                           modularity=float(nx.algorithms.community.modularity(G, comms, weight="weight")),
                           n_active=int(len(active)), edge_threshold=float(thr),
                           communities=sorted(layer_rec, key=lambda r: -r["mass"]))
    # figure: stacked category fingerprint of the top communities
    top_comms = sorted(layer_rec, key=lambda r: -r["mass"])[:8]
    bottom = np.zeros(len(top_comms))
    xs = range(len(top_comms))
    for ci, c in enumerate(cats):
        vals = [tc["cat_shares"][c] for tc in top_comms]
        ax.bar(xs, vals, bottom=bottom, label=c)
        bottom += np.array(vals)
    ax.set_title(f"Layer {L}: {len(comms)} teams (Q={results[str(L)]['modularity']:.2f})")
    ax.set_xlabel("community (by mass)"); ax.set_ylim(0, 1)
    ax.set_xticks(list(xs)); ax.set_xticklabels(
        [f"{tc['size']}e\n{tc['top_category'][:4]}" for tc in top_comms], fontsize=7)
axes[0].set_ylabel("category share of community mass")
axes[-1].legend(fontsize=7, loc="upper right")
fig.suptitle("Expert co-activation communities, characterized by problem category")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "E_communities.png"), dpi=150)
json.dump(results, open(os.path.join(OUT, "E_communities.json"), "w"), indent=2)
for L in LAYERS:
    r = results[str(L)]
    print(f"L{L}: {r['n_communities']} teams, Q={r['modularity']:.2f}")
    for c in r["communities"][:5]:
        print(f"   {c['size']:3d} experts | {c['top_category']:14s} "
              f"{c['top_share']*100:4.0f}% | specialists={c['n_specialists']}")
print("done")
