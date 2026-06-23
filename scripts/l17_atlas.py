#!/usr/bin/env python3
"""Deep-dive atlas of layer 17 — the recurring specialization hotspot.
Per-expert profile (mass, top category, enrichment, selectivity) at L17 in
base, plus base->LoRA mass change. CPU-only (reads *_routing_patterns.json).
Outputs: outputs/analysis/extended/I_l17_atlas.csv, .json, + figures."""
import os, json
import numpy as np
OUT="outputs/analysis/extended"; os.makedirs(os.path.join(OUT,"figures"),exist_ok=True)
L=17
bp=json.load(open("outputs/analysis/base/base_routing_patterns.json"))
lp=json.load(open("outputs/analysis/lora/lora_routing_patterns.json"))
cats=bp["categories"]
B=np.array([bp["category_mass"][str(L)][c] for c in cats])   # (6,128)
Lm=np.array([lp["category_mass"][str(L)][c] for c in cats])
# corpus base rates (all layers) for enrichment
tot={c:sum(sum(bp["category_mass"][k][c]) for k in bp["category_mass"]) for c in cats}
s=sum(tot.values()); rate=np.array([tot[c]/s for c in cats])

bmass=B.sum(0); lmass=Lm.sum(0)
bshare=B/np.maximum(B.sum(1,keepdims=True),1e-9)        # per-category share across experts
# per-expert category distribution (normalized within expert)
bexp=B/np.maximum(bmass,1e-9)                            # (6,128) share of expert's mass by cat
sel=bexp.max(0)                                          # selectivity
topcat=np.array(cats)[bexp.argmax(0)]
# enrichment of an expert for its top category = (expert's cat share) / base rate
enr=(bexp/rate[:,None]).max(0)
bn=bmass/bmass.sum(); ln=lmass/lmass.sum()
delta=ln-bn                                              # normalized mass change

import csv
rows=[]
for e in range(128):
    rows.append(dict(expert=e, mass=float(bmass[e]), mass_share=float(bn[e]),
                     top_category=topcat[e], selectivity=round(float(sel[e]),3),
                     enrichment=round(float(enr[e]),2),
                     lora_mass_share=float(ln[e]), delta=round(float(delta[e]),5)))
rows.sort(key=lambda r:-r["mass"])
with open(os.path.join(OUT,"I_l17_atlas.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

specialists=[r for r in rows if r["selectivity"]>0.8]
hubs=sorted(rows,key=lambda r:-r["mass"])[:8]
summary=dict(layer=L, n_specialists=len(specialists),
    specialists=[(r["expert"],r["top_category"],r["selectivity"]) for r in specialists],
    top_hubs=[(r["expert"],r["top_category"],round(r["mass_share"],3),r["selectivity"]) for r in hubs],
    biggest_lora_gainers=[(r["expert"],r["top_category"],r["delta"]) for r in sorted(rows,key=lambda r:-r["delta"])[:6]],
    biggest_lora_losers=[(r["expert"],r["top_category"],r["delta"]) for r in sorted(rows,key=lambda r:r["delta"])[:6]])
json.dump(summary,open(os.path.join(OUT,"I_l17_atlas.json"),"w"),indent=2)
print(json.dumps(summary,indent=2))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
colors={c:p for c,p in zip(cats,plt.cm.tab10.colors)}
# Fig 1: mass vs selectivity, colored by top category (the two-tier economy)
fig,ax=plt.subplots(figsize=(10,6.5))
for c in cats:
    idx=[i for i in range(128) if topcat[i]==c]
    ax.scatter(np.array(bn)[idx]*100, sel[idx], s=40, alpha=0.75, color=colors[c], label=c)
for r in specialists:
    ax.annotate(f"E{r['expert']}", (r['mass_share']*100, r['selectivity']),
                fontsize=7, xytext=(3,3), textcoords="offset points")
ax.axhline(1/len(cats),color="grey",ls="--",lw=0.8,label="uniform selectivity (1/6)")
ax.set_xlabel("share of layer-17 routing mass (%)"); ax.set_ylabel("selectivity (max category share)")
ax.set_title("Layer 17 expert atlas: the two-tier economy\n"
             "high-mass generalists (low selectivity) vs low-mass specialists (high)")
ax.legend(fontsize=7,ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"figures","I_l17_atlas.png"),dpi=150)
plt.close(fig)
# Fig 2: base->LoRA mass change per expert, colored by category
fig,ax=plt.subplots(figsize=(12,4.5))
order=np.argsort(delta)
ax.bar(range(128),delta[order]*100,
       color=[colors[topcat[i]] for i in order])
ax.set_xlabel("expert (sorted by Δ mass)"); ax.set_ylabel("Δ mass share base→LoRA (%)")
ax.set_title("Layer 17: which experts the symbolic LoRA recruited (+) or abandoned (−)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=colors[c],label=c) for c in cats],fontsize=7,ncol=3)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"figures","I_l17_lora_delta.png"),dpi=150)
print("saved figures")
