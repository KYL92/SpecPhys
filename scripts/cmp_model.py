"""CoTAF 대비 모델 비교: MMPD(조건별+Wilcoxon) + 13축. 사용: python cmp_model.py MODEL [REF=CoTAF]"""
import json,sys,numpy as np
from scipy.stats import wilcoxon
M=sys.argv[1]; REF=sys.argv[2] if len(sys.argv)>2 else "CoTAF"; R="_workspace/data/results"
def mm(m): return [r for r in json.load(open(f"{R}/agg_{m}_MMPD.json")) if r["subject"]>=7]
A,B=mm(REF),mm(M); key=lambda r:(r["subject"],r["light"],r["motion"],r["exercise"],r["skin"])
ka={key(r):r["err"] for r in A}; kb={key(r):r["err"] for r in B}; ks=sorted(set(ka)&set(kb)); x=np.array([ka[k] for k in ks]); y=np.array([kb[k] for k in ks])
print(f"MMPD n={len(ks)} {REF} {x.mean():.2f} -> {M} {y.mean():.2f} p={wilcoxon(x,y).pvalue:.4f} | hlv {sum(abs(r['ratio']-0.5)<0.1 for r in B)} dbl {sum(abs(r['ratio']-2)<0.2 for r in B)}")
for k in ("light","motion","skin"):
    print(k, "  ".join(f"{v}: {np.mean([r['err'] for r in A if r[k]==v]):.2f}->{np.mean([r['err'] for r in B if r[k]==v]):.2f}" for v in sorted(set(r[k] for r in A))))
a=json.load(open(f"{R}/agg_{REF}.json")); b=json.load(open(f"{R}/agg_{M}.json"))
print("  ".join(f"{ax} {np.mean([r['err'] for r in a[ax]]):.2f}->{np.mean([r['err'] for r in b[ax]]):.2f}" for ax in b if ax in a and isinstance(b[ax],list) and b[ax] and "err" in b[ax][0]))
