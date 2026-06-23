#!/usr/bin/env python3
"""Permutation null for the §C cross-layer NMI claim.
Observed adjacent-layer top-1 NMI vs a null that shuffles problem identity
between layers (breaks problem-specific coupling, preserves each layer's
marginal). Corrects for small-sample/many-label NMI bias."""
import json, glob, re, numpy as np
from math import log
N_EXPERTS=128
MOE=[1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51]
meta={p["problem_id"]:p for p in json.load(open("data/problems.json"))["problems"]}
rng=np.random.default_rng(42)
def nmi(a,b):
    a=np.asarray(a); b=np.asarray(b); n=len(a)
    Hx=Hy=I=0.0
    for x in np.unique(a):
        px=(a==x).mean(); Hx-=px*log(px)
    for y in np.unique(b):
        py=(b==y).mean(); Hy-=py*log(py)
    for x in np.unique(a):
        for y in np.unique(b):
            pxy=((a==x)&(b==y)).mean()
            if pxy>0: I+=pxy*log(pxy/((a==x).mean()*(b==y).mean()))
    d=(Hx+Hy)/2; return I/d if d>0 else 0.0
# top1 per problem per layer (generated tokens)
pids=[]; top1={L:[] for L in MOE}
for f in sorted(glob.glob("outputs/logs/base/routing_*.npz")):
    pid=re.sub(r"^routing_|\.npz$","",f.split("/")[-1])
    if pid not in meta: continue
    pids.append(pid); z=np.load(f); npre=int(z["n_prefill"])
    for L in MOE:
        ids=z[f"ids_{L}"]; w=z[f"w_{L}"].astype(np.float32)
        if ids.shape[0]>npre: ids,w=ids[npre:],w[npre:]
        v=np.zeros(N_EXPERTS); np.add.at(v,ids.reshape(-1),w.reshape(-1)); top1[L].append(int(v.argmax()))
obs=[]; nullmean=[]; excess=[]
for i in range(len(MOE)-1):
    a=np.array(top1[MOE[i]]); b=np.array(top1[MOE[i+1]])
    o=nmi(a,b)
    nd=[nmi(a,rng.permutation(b)) for _ in range(60)]
    obs.append(o); nullmean.append(np.mean(nd)); excess.append(o-np.mean(nd))
obs=np.array(obs); nullmean=np.array(nullmean); excess=np.array(excess)
print(f"observed NMI mean       {obs.mean():.3f}")
print(f"permutation-null mean   {nullmean.mean():.3f}  (small-sample bias floor)")
print(f"bias-corrected excess   {excess.mean():.3f}  (real problem-specific coupling)")
print(f"all pairs: observed > null?  {(obs>nullmean+3*0).all()} ; min excess {excess.min():.3f}")
import json as J
J.dump(dict(obs_mean=float(obs.mean()),null_mean=float(nullmean.mean()),
            excess_mean=float(excess.mean()),min_excess=float(excess.min()),
            n_pairs=len(obs)),open("outputs/analysis/extended/C_nmi_null.json","w"),indent=2)
