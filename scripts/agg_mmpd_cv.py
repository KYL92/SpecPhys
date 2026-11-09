"""MMPD 3-fold CV 집계: 각 폴드 모델의 held 폴드 피험자만 모아 agg_CoTAF_MCV_L_MMPD.json 생성"""
import json; R="/SPECPHYS_ROOT/_workspace/data/results"; FOLDS={1:range(1,12),2:range(12,23),3:range(23,34)}; rows=[]
for k,subs in FOLDS.items():
    d=json.load(open(f"{R}/agg_CoTAF_MCV{k}_L_MMPD.json")); rows+=[r for r in d if r["subject"] in set(subs)]
json.dump(rows,open(f"{R}/agg_CoTAF_MCV_L_MMPD.json","w"),indent=0); import numpy as np
print("CV rows",len(rows),"all-subject MAE",round(np.mean([r["err"] for r in rows]),2),"held-out(7~33) MAE",round(np.mean([r["err"] for r in rows if r["subject"]>=7]),2), "skin", {s:round(np.mean([r["err"] for r in rows if r["skin"]==s]),2) for s in (3,4,5,6)})
