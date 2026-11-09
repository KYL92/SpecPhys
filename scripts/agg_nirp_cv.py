"""3-fold 피험자 분리 CV 결합: 각 fold의 held-out 피험자 예측만 모아 가상 모델 CoTANcv_L(19명 전원 held-out)로 저장 + 다른 축은 fold 평균 보고"""
import json, os, numpy as np
R="/SPECPHYS_ROOT/_workspace/data/results"
FOLDS={1:[1,2,3,4,5,6,7],2:[8,9,10,11,12,13],3:[14,15,16,17,18,19]}
comb={}; other={}
for k,test in FOLDS.items():
    d=json.load(open(f"{R}/agg_CoTANcv{k}_L.json"))
    for ax in ("NIRP_STILL","NIRP_SMALL","NIRP_LARGE"): comb.setdefault(ax,[]).extend([r for r in d.get(ax,[]) if int(r["vid"]) in test])
    for ax,rows in d.items():
        if not ax.startswith("NIRP"): other.setdefault(ax,[]).append(np.mean([r["err"] for r in rows]))
json.dump(comb,open(f"{R}/agg_CoTANcv_L.json","w"),indent=1)
for ax,rows in comb.items():
    e=np.array([r["err"] for r in rows]); print(f"{ax:11s} n={len(e)} MAE={e.mean():.2f} dbl={sum(abs(r['ratio']-2)<0.2 for r in rows)} hlv={sum(abs(r['ratio']-0.5)<0.1 for r in rows)} | fold MAEs:",[round(float(np.mean([r['err'] for r in rows if int(r['vid']) in FOLDS[k]])),2) for k in FOLDS])
print("other axes (fold mean):",{ax:round(float(np.mean(v)),2) for ax,v in other.items()})
