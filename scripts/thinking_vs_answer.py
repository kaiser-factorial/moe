#!/usr/bin/env python3
"""Does routing change at the think->answer boundary?

Splits each generation at </think> (99/111 base problems have it). Maps the
char position of </think> to a token row via char-fraction * n_gen_tokens
(token ids weren't stored, so this is an approximation; answer regions are
short so the boundary lands close). Compares mid-network per-token routing
concentration & entropy in the THINKING vs ANSWER region, paired across
problems (Wilcoxon), and relates answer-region concentration to correctness.
Outputs: outputs/analysis/extended/G_think_answer.json + figure.
"""
import os, json, glob, re
import numpy as np
from scipy.stats import wilcoxon, pointbiserialr
N_EXPERTS=128; MID=[13,15,17,20,22,24]
LOG="outputs/logs/base"; OUT="outputs/analysis/extended"
os.makedirs(os.path.join(OUT,"figures"),exist_ok=True)
meta={p["problem_id"]:p for p in json.load(open("data/problems.json"))["problems"]}
results={json.loads(l)["problem_id"]:json.loads(l) for l in open(os.path.join(LOG,"results.jsonl"))}

def ent(p):
    p=p[p>0]; return float(-(p*np.log(p)).sum()) if p.size else 0.0
def region_stats(ids,w,lo,hi):
    conc=[]; ent_=[]
    for t in range(lo,hi):
        v=np.zeros(N_EXPERTS); np.add.at(v,ids[t],w[t])
        p=v/v.sum() if v.sum() else v
        conc.append(float(p.max())); ent_.append(ent(p))
    return (np.mean(conc) if conc else np.nan, np.mean(ent_) if ent_ else np.nan)

rows=[]
for f in sorted(glob.glob(os.path.join(LOG,"routing_*.npz"))):
    pid=re.sub(r"^routing_|\.npz$","",os.path.basename(f))
    if pid not in meta: continue
    r=results.get(pid,{}); gen=r.get("generated",""); ng=r.get("n_gen_tokens",0)
    if "</think>" not in gen or ng<8: continue
    z=np.load(f); npre=int(z["n_prefill"])
    ids=np.concatenate([z[f"ids_{L}"] for L in MID],axis=1)
    w=np.concatenate([z[f"w_{L}"].astype(np.float32) for L in MID],axis=1)
    if ids.shape[0]>npre: ids,w=ids[npre:],w[npre:]
    T=ids.shape[0]
    frac=gen.index("</think>")/max(len(gen),1)
    b=int(round(frac*T)); b=max(3,min(b,T-3))   # ensure both regions >=3 tok
    if T-b<3: continue
    tc,te=region_stats(ids,w,0,b); ac,ae=region_stats(ids,w,b,T)
    rows.append(dict(pid=pid,cat=meta[pid]["category"],correct=r.get("correct"),
                     think_conc=tc,ans_conc=ac,think_ent=te,ans_ent=ae,
                     ans_frac=(T-b)/T))

import numpy as np
tc=np.array([r["think_conc"] for r in rows]); ac=np.array([r["ans_conc"] for r in rows])
te=np.array([r["think_ent"] for r in rows]); ae=np.array([r["ans_ent"] for r in rows])
wc=wilcoxon(ac,tc); we=wilcoxon(ae,te)
# correctness: answer-region vs thinking-region concentration as predictor
scored=[r for r in rows if r["correct"] is not None]
cor=np.array([bool(r["correct"]) for r in scored])
ra,pa=pointbiserialr(cor,[r["ans_conc"] for r in scored])
rt,pt=pointbiserialr(cor,[r["think_conc"] for r in scored])
# category-composition control: restrict to the two categories with real
# accuracy variance (factual, reasoning); symbolic/computational were ~100%
# and creative/social unscored, so the pooled correlation can be category-
# composition rather than a within-condition routing signal.
fr=[r for r in scored if r["cat"] in ("factual","reasoning")]
corfr=np.array([bool(r["correct"]) for r in fr])
if len(set(corfr.tolist()))>1:
    rt_fr,pt_fr=pointbiserialr(corfr,[r["think_conc"] for r in fr])
    ra_fr,pa_fr=pointbiserialr(corfr,[r["ans_conc"] for r in fr])
else:
    rt_fr=pt_fr=ra_fr=pa_fr=float("nan")
out=dict(n=len(rows),n_scored=len(scored),
    mean_think_conc=float(tc.mean()),mean_ans_conc=float(ac.mean()),
    conc_delta_ans_minus_think=float((ac-tc).mean()),
    wilcoxon_conc_p=float(wc.pvalue),
    mean_think_ent=float(te.mean()),mean_ans_ent=float(ae.mean()),
    ent_delta=float((ae-te).mean()),wilcoxon_ent_p=float(we.pvalue),
    corr_answerconc_correct=dict(r=float(ra),p=float(pa)),
    corr_thinkconc_correct=dict(r=float(rt),p=float(pt)),
    corr_thinkconc_correct_factual_reasoning=dict(r=float(rt_fr),p=float(pt_fr),n=len(fr)),
    corr_answerconc_correct_factual_reasoning=dict(r=float(ra_fr),p=float(pa_fr),n=len(fr)))
json.dump(out,open(os.path.join(OUT,"G_think_answer.json"),"w"),indent=2)
print(json.dumps(out,indent=2))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
a1.scatter(tc,ac,s=18,alpha=0.6)
lim=[min(tc.min(),ac.min()),max(tc.max(),ac.max())]
a1.plot(lim,lim,"k--",lw=0.8); a1.set_xlabel("thinking-region concentration")
a1.set_ylabel("answer-region concentration")
a1.set_title(f"Concentration: answer vs thinking\nΔ={ (ac-tc).mean():+.4f}, Wilcoxon p={wc.pvalue:.1e}")
# correctness panel
for ok,m,c in [(True,"o","tab:green"),(False,"x","tab:red")]:
    s=[r for r in scored if bool(r["correct"])==ok]
    a2.scatter([r["ans_conc"] for r in s],
               np.random.RandomState(0).normal(ok,0.05,len(s)),
               marker=m,c=c,alpha=0.6,label="correct" if ok else "incorrect")
a2.set_yticks([0,1]); a2.set_yticklabels(["incorrect","correct"])
a2.set_xlabel("answer-region concentration")
a2.set_title(f"Answer-region concentration vs correctness\nr={ra:.2f}, p={pa:.3f}")
a2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"figures","G_think_answer.png"),dpi=150)
print("saved figure; n=",len(rows))
