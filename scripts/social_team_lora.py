#!/usr/bin/env python3
"""Track the base-model social/ethical expert team into the LoRA run.

1. Detect the social-enriched community at layers 17/24/38 in BASE.
2. Freeze that membership; in BOTH base and LoRA measure:
   - social/ethical enrichment of that exact expert set (preserved?)
   - membership overlap (Jaccard) vs LoRA's own most-social team
   - LEAKAGE: symbolic mass-share onto the social team, base vs LoRA
Outputs: outputs/analysis/extended/F_social_team_lora.json + figure.
"""
import os, json, glob, re, itertools
from collections import defaultdict
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

N_EXPERTS = 128
LAYERS = [17, 24, 38]
OUT = "outputs/analysis/extended"; os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
meta = {p["problem_id"]: p for p in json.load(open("data/problems.json"))["problems"]}
cats = sorted({m["category"] for m in meta.values()})
SOC = cats.index("social_ethical"); SYM = cats.index("symbolic")
# base rates
pat = json.load(open("outputs/analysis/base/base_routing_patterns.json"))
tot = {c: sum(sum(pat["category_mass"][L][c]) for L in pat["category_mass"]) for c in cats}
s = sum(tot.values()); base_rate = {c: tot[c]/s for c in cats}

def load(logdir):
    st = {}
    for f in sorted(glob.glob(os.path.join(logdir, "routing_*.npz"))):
        pid = re.sub(r"^routing_|\.npz$", "", os.path.basename(f))
        if pid in meta:
            z = np.load(f); st[pid] = (z, int(z["n_prefill"]))
    return st

def coactivation(st, L):
    C = np.zeros((N_EXPERTS, N_EXPERTS)); catmass = np.zeros((len(cats), N_EXPERTS))
    for pid, (z, npre) in st.items():
        ids = z[f"ids_{L}"]; w = z[f"w_{L}"].astype(np.float32)
        if ids.shape[0] > npre: ids, w = ids[npre:], w[npre:]
        ci = cats.index(meta[pid]["category"])
        np.add.at(catmass[ci], ids.reshape(-1), w.reshape(-1))
        for row in ids:
            u = np.unique(row)
            for a, b in itertools.combinations(u, 2):
                C[a, b] += 1; C[b, a] += 1
    return C, catmass

def communities(C):
    deg = C.sum(1); active = np.where(deg > 0)[0]
    rate = C / np.maximum(deg[:, None], 1); rate = 0.5*(rate+rate.T)
    sub = rate[active][:, active]
    thr = np.percentile(sub[sub > 0], 70)
    G = nx.Graph(); G.add_nodes_from(active.tolist())
    for i in active:
        for j in active:
            if i < j and rate[i, j] >= thr:
                G.add_edge(int(i), int(j), weight=float(rate[i, j]))
    return [set(c) for c in greedy_modularity_communities(G, weight="weight") if len(c) >= 3]

def soc_enrich(catmass, members):
    members = list(members)
    cmass = np.array([catmass[ci, members].sum() for ci in range(len(cats))])
    share = cmass / max(cmass.sum(), 1e-9)
    return {c: float(share[i] / base_rate[c]) for i, c in enumerate(cats)}

def team_set(catmass, comms, ci):
    """community whose mass is most enriched for category ci."""
    best, bestv = None, -1
    for com in comms:
        e = soc_enrich(catmass, com)[cats[ci]]
        if e > bestv: best, bestv = com, e
    return best, bestv

base = load("outputs/logs/base"); lora = load("outputs/logs/lora")
res = {}
for L in LAYERS:
    Cb, cmb = coactivation(base, L); Cl, cml = coactivation(lora, L)
    comms_b = communities(Cb); comms_l = communities(Cl)
    soc_b, eb = team_set(cmb, comms_b, SOC)      # base social team
    soc_l, el = team_set(cml, comms_l, SOC)      # lora's own social team
    # frozen base membership measured in both
    enr_b = soc_enrich(cmb, soc_b)               # base team, base data
    enr_l_same = soc_enrich(cml, soc_b)          # base team, LORA data
    jac = len(soc_b & soc_l) / len(soc_b | soc_l)
    res[str(L)] = dict(
        base_team_size=len(soc_b), lora_team_size=len(soc_l),
        base_social_enrich=round(enr_b["social_ethical"], 3),
        baseteam_in_lora_social_enrich=round(enr_l_same["social_ethical"], 3),
        baseteam_symbolic_enrich_base=round(enr_b["symbolic"], 3),
        baseteam_symbolic_enrich_lora=round(enr_l_same["symbolic"], 3),
        membership_jaccard=round(jac, 3),
        lora_own_social_team_enrich=round(el, 3))
    print(f"L{L}: base soc team {len(soc_b)}e enrich {enr_b['social_ethical']:.2f}x "
          f"-> same experts in LoRA {enr_l_same['social_ethical']:.2f}x | "
          f"membership Jaccard {jac:.2f} | "
          f"symbolic-leak {enr_b['symbolic']:.2f}->{enr_l_same['symbolic']:.2f}")
json.dump(res, open(os.path.join(OUT, "F_social_team_lora.json"), "w"), indent=2)

# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(LAYERS)); wdt = 0.35
a1.bar(x-wdt/2, [res[str(L)]["base_social_enrich"] for L in LAYERS], wdt, label="base data")
a1.bar(x+wdt/2, [res[str(L)]["baseteam_in_lora_social_enrich"] for L in LAYERS], wdt, label="LoRA data")
a1.axhline(1, color="k", lw=0.7, ls="--"); a1.set_xticks(x); a1.set_xticklabels([f"L{L}" for L in LAYERS])
a1.set_ylabel("social/ethical enrichment of the base team"); a1.legend()
a1.set_title("Does the base social team stay social under LoRA?")
a2.bar(x-wdt/2, [res[str(L)]["baseteam_symbolic_enrich_base"] for L in LAYERS], wdt, label="base data")
a2.bar(x+wdt/2, [res[str(L)]["baseteam_symbolic_enrich_lora"] for L in LAYERS], wdt, label="LoRA data")
a2.axhline(1, color="k", lw=0.7, ls="--"); a2.set_xticks(x); a2.set_xticklabels([f"L{L}" for L in LAYERS])
a2.set_ylabel("symbolic enrichment of the base social team"); a2.legend()
a2.set_title("Did symbolic traffic leak INTO the social team?")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "F_social_team_lora.png"), dpi=150)
print("saved figure")
