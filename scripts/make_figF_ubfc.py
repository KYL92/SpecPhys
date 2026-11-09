import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"]="Noto Sans CJK JP"
R="/SPECPHYS_ROOT/_workspace/data/results"; S="/SPECPHYS_ROOT/_workspace/slides"
def load(m):
    d={}
    for suf in ("","_newsets"):
        p=f"{R}/agg_{m}{suf}.json"
        if os.path.exists(p): d.update(json.load(open(p)))
    return d
def cell(m,ax):
    rows=load(m).get(ax)
    return np.nan if not rows else np.mean([r["err"] for r in rows])
# (모델, 라벨, UBFC42 사용가능?)  in-domain 표시 †
rows=[("PFpure","PhysFormer — PURE만 (zero-shot)",True),("base","PhysFormer — UBFC만 (in-domain †)",True),("CoTAF2","PhysFormer + 동일 코퍼스 (CoTAF2, held-out)",False),
      ("RFpure","RhythmFormer — PURE만 (zero-shot)",True),("RhythmFormer","RhythmFormer — UBFC만 (in-domain †)",True),("RFA","RhythmFormer + 동일 코퍼스 (RFA, held-out)",False),
      ("FPpure","FactorizePhys — PURE만 (zero-shot)",True),("FactorizePhys","FactorizePhys — UBFC만 (in-domain †)",True),("FFA","FactorizePhys + 동일 코퍼스 (FFA, held-out)",False),
      ("T2L_s100","PhysFormer — 합성만 (T2L, zero-shot)",True)]
cols=[("UBFC","UBFC 전체 42명\n(clean)"),("REALTEST","held-out 42–49\n(clean)"),("RTNI","held-out 42–49\n+ flicker"),("RTNC","held-out 42–49\n+ colour-temp")]
M=np.array([[cell(m,a) if (a!="UBFC" or ok) else np.nan for a,_ in cols] for m,_,ok in rows])
fig,ax=plt.subplots(figsize=(12.5,5.2)); cm=matplotlib.colormaps["RdYlGn_r"].copy(); cm.set_bad("#d9d6d2"); im=ax.imshow(np.ma.masked_invalid(M),cmap=cm,vmin=0,vmax=25,aspect="auto")
ax.set_xticks(range(len(cols))); ax.set_xticklabels([c for _,c in cols],fontsize=10); ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _,l,_ in rows],fontsize=9.5)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v=M[i,j]; ax.text(j,i,"n/a (학습 포함)" if np.isnan(v) else f"{v:.2f}",ha="center",va="center",fontsize=9.5,color="gray" if np.isnan(v) else "black")
for y in (2.5,5.5,8.5): ax.axhline(y,color="white",lw=3)
ax.set_title("UBFC-rPPG 조건별 HR MAE — 백본별 'PURE만 / UBFC만' vs '+ 동일 코퍼스(ALLF2)' · 정본 프로토콜",fontsize=12); plt.colorbar(im,ax=ax,fraction=0.02,pad=0.01,label="MAE")
fig.text(0.01,0.01,"† UBFC 42명 전부를 학습에 사용한 공식 가중치 — 모든 열이 in-domain. 코퍼스 모델은 UBFC 1~41 학습 → 42~49가 held-out. PURE만·합성만 모델은 UBFC를 본 적 없음(zero-shot).",fontsize=8.5,color="gray")
fig.tight_layout(rect=(0,0.03,1,1)); fig.savefig(f"{S}/figF_ubfc_fair.png",dpi=200); plt.close()
print("| 모델 | "+" | ".join(c.replace("\n"," ") for _,c in cols)+" |"); print("|---|"+"---|"*len(cols))
for (m,l,ok),row in zip(rows,M): print(f"| {l} | "+" | ".join("n/a" if np.isnan(v) else f"{v:.2f}" for v in row)+" |")
