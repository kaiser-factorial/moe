#!/usr/bin/env python3
"""Subtype-specialization on base vs LoRA logs (CPU-only).
Did fine-tuning preserve, sharpen, or blur the subtype routing structure?
Outputs: outputs/analysis/extended/A2_lora_subtype.json + figure."""
import os, json, glob, re, itertools
from collections import defaultdict
import numpy as np
N_EXPERTS=128
MOE=[1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51]
MID=[13,15,17,20,22,24]
OUT="outputs/analysis/extended"; os.makedirs(os.path.join(OUT,"figures"),exist_ok=True)
rng=np.random.default_rng(42)
meta={p["problem_id"]:p for p in json.load(open("data/problems.json"))["problems"]}

def jsd(p,q):
    p=p/p.sum() if p.sum() else p; q=q/q.sum() if q.sum() else q; m=0.5*(p+q)
    kl=lambda a,b:(a[a>0]*np.log(a[a>0]/b[a>0])).sum()
    return float(0.5*kl(p,m)+0.5*kl(q,m))

def load(logdir):
    mv={}
    for f in sorted(glob.glob(os.path.join(logdir,"routing_*.npz"))):
        pid=re.sub(r"^routing_|\.npz$","",os.path.basename(f))
        if pid not in meta: continue
        z=np.load(f); npre=int(z["n_prefill"])
        d={}
        for L in MID:
            ids,w=z[f"ids_{L}"],z[f"w_{L}"].astype(np.float32)
            if ids.shape[0]>npre: ids,w=ids[npre:],w[npre:]
            v=np.zeros(N_EXPERTS); np.add.at(v,ids.reshape(-1),w.reshape(-1)); d[L]=v
        mv[pid]=np.concatenate([d[L] for L in MID])
    return mv

def subtype_gap(mv, cat):
    pids=[p for p in mv if meta[p]["category"]==cat]
    D={p:(mv[p]/mv[p].sum() if mv[p].sum() else mv[p]) for p in pids}
    sub={p:meta[p]["subtype"] for p in pids}
    pair=list(itertools.combinations(pids,2))
    dmat=np.array([jsd(D[a],D[b]) for a,b in pair])
    same=np.array([sub[a]==sub[b] for a,b in pair])
    within,between=dmat[same].mean(),dmat[~same].mean()
    # permutation
    labels=np.array([sub[p] for p in pids]); idx=list(itertools.combinations(range(len(pids)),2))
    obs=between-within
    perm=[]
    for _ in range(2000):
        sh=rng.permutation(labels); sm=np.array([sh[i]==sh[j] for i,j in idx])
        perm.append(dmat[~sm].mean()-dmat[sm].mean())
    p=float((np.array(perm)>=obs).mean())
    return dict(n=len(pids),within=float(within),between=float(between),
                gap=float(obs),ratio=float(within/between),perm_p=p)

base=load("outputs/logs/base"); lora=load("outputs/logs/lora")
res={}
for cat in ["symbolic","reasoning","factual"]:
    res[cat]={"base":subtype_gap(base,cat),"lora":subtype_gap(lora,cat)}
    b,l=res[cat]["base"],res[cat]["lora"]
    print(f"{cat:10s} base within/between={b['within']:.3f}/{b['between']:.3f} ratio={b['ratio']:.2f} | "
          f"lora {l['within']:.3f}/{l['between']:.3f} ratio={l['ratio']:.2f}")
json.dump(res,open(os.path.join(OUT,"A2_lora_subtype.json"),"w"),indent=2)

# figure: side-by-side symbolic JSD matrices base vs lora
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def matrix(mv):
    pids=sorted([p for p in mv if meta[p]["category"]=="symbolic"],key=lambda p:meta[p]["subtype"])
    D={p:(mv[p]/mv[p].sum()) for p in pids}
    return np.array([[jsd(D[a],D[b]) for b in pids] for a in pids]),[meta[p]["subtype"] for p in pids]
Mb,subs=matrix(base); Ml,_=matrix(lora)
vmax=max(Mb.max(),Ml.max())
fig,axes=plt.subplots(1,2,figsize=(16,7))
for ax,(M,t) in zip(axes,[(Mb,"BASE"),(Ml,"LoRA")]):
    im=ax.imshow(M,cmap="viridis_r",vmax=vmax)
    ax.set_xticks(range(len(subs))); ax.set_yticks(range(len(subs)))
    ax.set_xticklabels(subs,rotation=90,fontsize=6); ax.set_yticklabels(subs,fontsize=6)
    ax.set_title(f"{t}: symbolic subtype routing JSD")
    plt.colorbar(im,ax=ax,fraction=0.046)
fig.suptitle("Did the symbolic LoRA preserve subtype routing structure? (mid-network)")
fig.tight_layout()
fig.savefig(os.path.join(OUT,"figures","A2_base_vs_lora_symbolic.png"),dpi=150)
print("saved figure")
