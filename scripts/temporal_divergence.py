#!/usr/bin/env python3
"""Where in the generation does the LoRA's routing re-weighting happen?
For each category, aggregate per-decile routing mass (mid-network) separately
for base and LoRA, then JSD(base_decile, lora_decile) per decile.
Flat => uniform re-weighting; rising => localizes (e.g. at the answer)."""
import os, json, glob, re
import numpy as np
N_EXPERTS=128; MID=[13,15,17,20,22,24]; NB=10
OUT="outputs/analysis/extended"; os.makedirs(os.path.join(OUT,"figures"),exist_ok=True)
meta={p["problem_id"]:p for p in json.load(open("data/problems.json"))["problems"]}

def jsd(p,q):
    p=p/p.sum() if p.sum() else p; q=q/q.sum() if q.sum() else q; m=0.5*(p+q)
    kl=lambda a,b:(a[a>0]*np.log(a[a>0]/b[a>0])).sum()
    return float(0.5*kl(p,m)+0.5*kl(q,m))

def decile_mass(logdir):
    """cat -> (NB,128) summed routing mass by generation decile (mid-network)."""
    acc={}
    for f in sorted(glob.glob(os.path.join(logdir,"routing_*.npz"))):
        pid=re.sub(r"^routing_|\.npz$","",os.path.basename(f))
        if pid not in meta: continue
        cat=meta[pid]["category"]; acc.setdefault(cat,np.zeros((NB,N_EXPERTS)))
        z=np.load(f); npre=int(z["n_prefill"])
        ids=np.concatenate([z[f"ids_{L}"] for L in MID],axis=1)
        w=np.concatenate([z[f"w_{L}"].astype(np.float32) for L in MID],axis=1)
        if ids.shape[0]>npre: ids,w=ids[npre:],w[npre:]
        T=ids.shape[0]
        if T<NB: continue
        for t in range(T):
            b=min(int(t/T*NB),NB-1); np.add.at(acc[cat][b],ids[t],w[t])
    return acc

base=decile_mass("outputs/logs/base"); lora=decile_mass("outputs/logs/lora")
cats=sorted(set(base)&set(lora))
res={}
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(11,5)); x=(np.arange(NB)+0.5)/NB*100
for c in cats:
    dv=[jsd(base[c][b],lora[c][b]) for b in range(NB)]
    res[c]=dict(per_decile=dv,mean=float(np.mean(dv)),
                first=float(dv[0]),last=float(dv[-1]))
    ax.plot(x,dv,marker="o",ms=3,label=c)
ax.set_xlabel("position in generation (%)"); ax.set_ylabel("base↔LoRA routing JSD")
ax.set_title("Where does the symbolic LoRA re-weight routing? (mid-network, by category)")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(OUT,"figures","A3_temporal_divergence.png"),dpi=150)
json.dump(res,open(os.path.join(OUT,"A3_temporal_divergence.json"),"w"),indent=2)
for c in cats: print(f"{c:14s} mean JSD={res[c]['mean']:.4f}  first={res[c]['first']:.4f} last={res[c]['last']:.4f}")
