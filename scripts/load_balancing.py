#!/usr/bin/env python3
"""Does NemotronH's routing actually balance load across its 128 experts?

The architecture uses a DeepSeek-V3-style e_score_correction_bias whose job is
to prevent expert collapse. We test the outcome on real traffic: per MoE layer,
the corpus-aggregate routing-mass distribution over 128 experts.
Metrics: Gini, dead experts (zero mass), #experts carrying 50%/90% of mass,
top-1 share. Base vs LoRA. CPU-only (reads *_routing_patterns.json).
Outputs: outputs/analysis/extended/H_load_balancing.json + figure.
"""
import os, json
import numpy as np
OUT="outputs/analysis/extended"; os.makedirs(os.path.join(OUT,"figures"),exist_ok=True)

def gini(x):
    x=np.sort(np.asarray(x,float)); n=len(x); s=x.sum()
    if s==0: return 0.0
    return float((2*np.arange(1,n+1)-n-1).dot(x)/(n*s))

def per_layer_totals(path):
    d=json.load(open(path)); cats=d["categories"]
    out={}
    for L,cm in d["category_mass"].items():
        tot=np.zeros(128)
        for c in cats: tot+=np.array(cm[c])
        out[int(L)]=tot
    return out

base=per_layer_totals("outputs/analysis/base/base_routing_patterns.json")
lora=per_layer_totals("outputs/analysis/lora/lora_routing_patterns.json")
layers=sorted(base)

def stats(tot):
    p=tot/tot.sum() if tot.sum() else tot
    sp=np.sort(p)[::-1]; cum=np.cumsum(sp)
    return dict(gini=gini(tot), dead=int((tot==0).sum()),
               n50=int(np.searchsorted(cum,0.5)+1),
               n90=int(np.searchsorted(cum,0.9)+1),
               top1=float(sp[0]))

res={"base":{},"lora":{}}
for L in layers:
    res["base"][str(L)]=stats(base[L]); res["lora"][str(L)]=stats(lora[L])
json.dump(res,open(os.path.join(OUT,"H_load_balancing.json"),"w"),indent=2)

gb=[res["base"][str(L)]["gini"] for L in layers]; gl=[res["lora"][str(L)]["gini"] for L in layers]
db=[res["base"][str(L)]["dead"] for L in layers]; dl=[res["lora"][str(L)]["dead"] for L in layers]
n50b=[res["base"][str(L)]["n50"] for L in layers]; n50l=[res["lora"][str(L)]["n50"] for L in layers]
print(f"GINI    base mean {np.mean(gb):.3f} (range {min(gb):.3f}-{max(gb):.3f}) | lora {np.mean(gl):.3f}")
print(f"DEAD    base total {sum(db)} (max/layer {max(db)}) | lora total {sum(dl)}")
print(f"N@50%   base mean {np.mean(n50b):.1f}/128 | lora {np.mean(n50l):.1f}")
print("per-layer (L: gini_b/gini_l dead_b/dead_l n50_b/n50_l):")
for L in layers:
    b,l=res["base"][str(L)],res["lora"][str(L)]
    print(f"  {L:2d}: {b['gini']:.2f}/{l['gini']:.2f}  {b['dead']:3d}/{l['dead']:3d}  {b['n50']:3d}/{l['n50']:3d}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(16,4.6))
a1.plot(layers,gb,marker="o",ms=3,label="base"); a1.plot(layers,gl,marker="s",ms=3,label="lora")
a1.set_title("Gini of expert utilization by layer\n(0=perfectly balanced, 1=one expert)")
a1.set_xlabel("MoE layer"); a1.set_ylabel("Gini"); a1.legend(); a1.set_ylim(0,1)
a2.plot(layers,db,marker="o",ms=3,label="base"); a2.plot(layers,dl,marker="s",ms=3,label="lora")
a2.set_title("Dead experts (zero mass) by layer"); a2.set_xlabel("MoE layer"); a2.set_ylabel("# dead /128"); a2.legend()
a3.plot(layers,n50b,marker="o",ms=3,label="base"); a3.plot(layers,n50l,marker="s",ms=3,label="lora")
a3.axhline(64,color="k",ls="--",lw=0.7,label="half (uniform)")
a3.set_title("# experts carrying 50% of mass"); a3.set_xlabel("MoE layer"); a3.set_ylabel("# experts"); a3.legend()
fig.suptitle("Expert load balancing across 128 routed experts per layer")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"figures","H_load_balancing.png"),dpi=150)
print("saved figure")
