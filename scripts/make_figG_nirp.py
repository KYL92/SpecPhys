"""MR-NIRP 조건별 비교(MMPD 그림과 동형): 전체 19명 / held-out 7~19, 모델 행 = base, RhythmFormer, FactorizePhys, POS, CoTAF(zero-shot), CoTAF-L, CoTAN-L(NIRP in-domain+label-fix)"""
import json, os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
plt.rcParams["font.family"]="Noto Sans CJK JP"
R="/SPECPHYS_ROOT/_workspace/data/results"; S="/SPECPHYS_ROOT/_workspace/slides"
def load(m):
    d={}
    for suf in ("","_newsets"):
        p=f"{R}/agg_{m}{suf}.json"
        if os.path.exists(p): d.update(json.load(open(p)))
    return d
AX=[("NIRP_STILL","정지"),("NIRP_SMALL","소동작"),("NIRP_LARGE","대동작")]
def errs(m,ax):
    if m=="POS":
        p=f"{R}/pos_pervideo_{ax}.json"; return {int(k):v for k,v in json.load(open(p)).items()} if os.path.exists(p) else {}
    return {int(r["vid"]):r["err"] for r in load(m).get(ax,[])}
def dh(m,ax,ho):
    rows=[r for r in load(m).get(ax,[]) if (int(r["vid"])>=7 or not ho)]
    return sum(abs(r["ratio"]-2)<0.2 for r in rows),sum(abs(r["ratio"]-0.5)<0.1 for r in rows)
models=[("base","PhysFormer — UBFC만"),("RhythmFormer_arch","RhythmFormer — UBFC만"),("FactorizePhys_arch","FactorizePhys — UBFC만"),("POS","POS (학습 없음)"),("CoTAF","CoTAF (제안, zero-shot)"),("CoTAF_L","CoTAF-L (zero-shot, 라벨 수정)"),("CoTAN_L","CoTAN-L (MR-NIRP 1~6 in-domain + 라벨 수정)"),("CoTANcv_L","CoTAN-CV-L (3-fold, 12~13명 학습, 전원 held-out)")]
models=[(m,l) for m,l in models if any(errs(m,a) for a,_ in AX)]
cols=[(a,f"{l}\n전체 19명") for a,l in AX]+[(a,f"{l}\nheld-out 7~19") for a,l in AX]
M=[]
for m,_ in models:
    row=[]
    for a,_ in AX: e=errs(m,a); row.append(np.mean(list(e.values())) if e else np.nan)
    for a,_ in AX: e={k:v for k,v in errs(m,a).items() if (k>=7 or m=='CoTANcv_L')}; row.append(np.mean(list(e.values())) if e else np.nan)
    M.append(row)
M=np.array(M)
fig,ax=plt.subplots(figsize=(12.5,4.6)); cm=matplotlib.colormaps["RdYlGn_r"].copy(); cm.set_bad("#d9d6d2"); im=ax.imshow(np.ma.masked_invalid(M),cmap=cm,vmin=3,vmax=20,aspect="auto")
ax.set_xticks(range(len(cols))); ax.set_xticklabels([c for _,c in cols],fontsize=10); ax.set_yticks(range(len(models))); ax.set_yticklabels([l for _,l in models],fontsize=10)
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j,i,"–" if np.isnan(M[i,j]) else f"{M[i,j]:.2f}",ha="center",va="center",fontsize=10)
ax.axvline(2.5,color="white",lw=4); ax.set_title("MR-NIRP Car(실차 운전) 조건별 HR MAE — 전체 19명(zero-shot 모델) vs held-out 7~19(공통 비교) · 정본 프로토콜",fontsize=12); plt.colorbar(im,ax=ax,fraction=0.02,pad=0.01,label="MAE")
fig.text(0.01,0.01,"CoTAN-L: 피험자 1~6 학습 → held-out 7~19. CoTAN-CV-L: 3-fold 피험자 분리 교차검증(fold당 12~13명 학습) → 19명 전원이 held-out이므로 두 열 모두 held-out 값. 다른 모델은 MR-NIRP zero-shot.",fontsize=9,color="gray")
fig.tight_layout(rect=(0,0.03,1,1)); fig.savefig(f"{S}/figG_nirp.png",dpi=200); plt.close()
print("| 모델 | 정지 (19 / HO) | 소동작 (19 / HO) | 대동작 (19 / HO) | dbl/hlv HO |"); print("|---|---|---|---|---|")
for (m,l),row in zip(models,M):
    d=[dh(m,a,True) for a,_ in AX]; print(f"| {l} | {row[0]:.2f} / {row[3]:.2f} | {row[1]:.2f} / {row[4]:.2f} | {row[2]:.2f} / {row[5]:.2f} | "+" · ".join(f"{x}/{y}" for x,y in d)+" |")
out=[]
for a,l in AX:
    for m1,m2 in (("POS","CoTAN_L"),("CoTAF","CoTAN_L"),("POS","CoTAF"),("FactorizePhys_arch","CoTAN_L"),("POS","CoTANcv_L"),("CoTAF","CoTANcv_L"),("FactorizePhys_arch","CoTANcv_L")):
        A,B=errs(m1,a),errs(m2,a); ks=[k for k in sorted(set(A)&set(B)) if (k>=7 or m2=='CoTANcv_L')]
        if len(ks)<5: continue
        x=np.array([A[k] for k in ks]); y=np.array([B[k] for k in ks])
        if np.all(x==y): continue
        W,p=wilcoxon(x,y); out.append(dict(label=f"{a} held-out: {m1} vs {m2}",n=len(ks),mae_a=float(x.mean()),mae_b=float(y.mean()),p=float(p))); print(f"{l:4s} {m1:18s}->{m2:8s} n={len(ks)} {x.mean():6.2f}->{y.mean():6.2f} p={p:.3g}")
json.dump(out,open(f"{R}/stats_nirp.json","w"),indent=1)
