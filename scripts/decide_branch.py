"""모델 vs 기준(CoTAF_L) MMPD held-out Wilcoxon → IMPROVED/NOT 출력 (평균 낮고 p<0.1)"""
import json,sys,numpy as np
from scipy.stats import wilcoxon
M=sys.argv[1]; REF=sys.argv[2] if len(sys.argv)>2 else "CoTAF_L"; R="/SPECPHYS_ROOT/_workspace/data/results"
def mm(m): return {(r["subject"],r["light"],r["motion"],r["exercise"],r["skin"]):r["err"] for r in json.load(open(f"{R}/agg_{m}_MMPD.json")) if r["subject"]>=7}
a,b=mm(REF),mm(M); k=sorted(set(a)&set(b)); x=np.array([a[i] for i in k]); y=np.array([b[i] for i in k]); p=wilcoxon(x,y).pvalue
print("IMPROVED" if (y.mean()<x.mean() and p<0.1) else "NOT", f"{x.mean():.2f}->{y.mean():.2f} p={p:.4f}")
