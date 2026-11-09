"""도메인 특화 합성 결과 그림: MMPD·MR-NIRP에서 zero-shot(CoTAF) / 도메인 특화 합성(CoTAF-DM-L, CoTAF-DN-L) / 실 라벨 in-domain(CoTAM-L, CoTAN-CV-L) / FUSE 비교 + Wilcoxon"""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
plt.rcParams["font.family"]="Noto Sans CJK JP"
R="/SPECPHYS_ROOT/_workspace/data/results"; S="/SPECPHYS_ROOT/_workspace/slides"
def load(m):
    d={}
    for suf in ("","_newsets"):
        p=f"{R}/agg_{m}{suf}.json"
        if os.path.exists(p): d.update(json.load(open(p)))
    p=f"{R}/agg_{m}_MMPD.json"
    if os.path.exists(p): d["MMPD"]=[r for r in json.load(open(p)) if r["subject"]>=7]
    return d
def errs(d,ax,sub=None):
    rows=d.get(ax,[]); rows=[r for r in rows if sub(r)] if sub else rows
    return {str(r["vid"]):r["err"] for r in rows}
LIGHT={1:"LED-low",2:"LED-high",3:"Incand.",4:"Nature"}; MOTION={1:"Stat.",2:"Rot.",3:"Talk",4:"Walk"}; SKIN={3:"S3",4:"S4",5:"S5",6:"S6"}
rowsM=[("base","PhysFormer — UBFC만 (zero-shot)"),("CoTAF","CoTAF (zero-shot, legacy 라벨)"),("CoTAF_L","CoTAF-L (zero-shot, cumsum 라벨 = 기준)"),("CoTAF_DM_L","CoTAF-DM-L (MMPD 특화 합성: 실 정지 프레임 + 측정 교란)"),("CoTAF_DM2A_L","CoTAF-DM2A-L (2요인 A: 실 정지 프레임만)"),("CoTAF_DM2B_L","CoTAF-DM2B-L (2요인 B: 가상얼굴 + 색 통계 + 측정 교란)"),("CoTAF_S300F_L","S300F-L (가상얼굴 203, 고정 주입)"),("CoTAF_S300P_L","S300P-L (203, 피부톤 의존 주입)"),("CoTAF_S300R_L","S300R-L (203, 실 움직임)"),("CoTAF_S300RB_L","S300RB-L (실 움직임 + 실 ×2)"),("CoTAF_S300RA_L","S300RA-L (실 움직임 + 해부학 A(x,y))"),("CoTAF_A_L","CoTAF-A-L (코퍼스 42클립 A(x,y) 재렌더)"),("CoTAF_T_L","CoTAF-T-L (디지털 트윈 v1: 파형+정반사+코덱)"),("CoTAF_TW_L","CoTAF-TW-L (실 PPG 파형 뱅크만) ★채택"),("CoTAF_TS_L","CoTAF-TS-L (정반사 분리만)"),("CoTAF_TC_L","CoTAF-TC-L (코덱 스윕만)"),("CoTAF_TWM_L","CoTAF-TWM-L (파형 뱅크 + 실 구동 32종)"),("CoTAF_N_L","CoTAF-N-L (MR-NIRP 트윈)"),("CoTAF_L_DYN","동적 얼굴 검출 전처리 (CoTAF-L 가중치)"),("CoTAF_MCV_L","MMPD 실 라벨 3-fold CV (상한)"),("CoTAM_L","CoTAM-L (MMPD 실 라벨 in-domain)"),("CoTAF_S300RM_L","S300RM-L (실 움직임 + MMPD 실)"),("CoTAF_S300RAM_L","S300RAM-L (+A(x,y) + MMPD 실)"),("FUSE","FUSE (CoTAF ⊗ FactorizePhys)"),("FUSE_DM","FUSE-DM (CoTAF-DM-L ⊗ FactorizePhys)")]
rowsM=[(m,l) for m,l in rowsM if load(m).get("MMPD")]
cols=[("all","전체")]+[(("light",v),n) for v,n in LIGHT.items()]+[(("motion",v),n) for v,n in MOTION.items()]+[(("skin",v),n) for v,n in SKIN.items()]
M=[]
for m,_ in rowsM:
    rows=load(m)["MMPD"]; M.append([np.mean([r["err"] for r in rows if c=="all" or r[c[0]]==c[1]]) for c,_ in cols])
M=np.array(M); fig,ax=plt.subplots(figsize=(12.5,0.36*len(rowsM)+1.2)); im=ax.imshow(M,cmap="RdYlGn_r",vmin=3,vmax=25,aspect="auto")
ax.set_xticks(range(len(cols))); ax.set_xticklabels([n for _,n in cols],fontsize=10); ax.set_yticks(range(len(rowsM))); ax.set_yticklabels([l for _,l in rowsM],fontsize=10)
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j,i,f"{M[i,j]:.1f}",ha="center",va="center",fontsize=9)
ax.set_title("MMPD held-out(7~33, 540클립) HR MAE — zero-shot vs 도메인 특화 합성(비지도) vs 실 라벨 in-domain vs 융합",fontsize=12); plt.colorbar(im,ax=ax,fraction=0.02,pad=0.01)
fig.tight_layout(); fig.savefig(f"{S}/figH_domain_mmpd.png",dpi=200); plt.close()
print("MMPD held-out:"); 
for (m,l),row in zip(rowsM,M): print(f"  {l:45s} {row[0]:.2f}")
A=errs(load("CoTAF_L"),"MMPD")
for m,_ in rowsM:
    if m=="CoTAF": continue
    B=errs(load(m),"MMPD"); ks=sorted(set(A)&set(B)); x=np.array([A[k] for k in ks]); y=np.array([B[k] for k in ks])
    if len(ks)>5 and not np.all(x==y): print(f"  Wilcoxon CoTAF-L→{m}: n={len(ks)} {x.mean():.2f}→{y.mean():.2f} p={wilcoxon(x,y)[1]:.3g}")
# NIRP
rowsN=[("CoTAF","CoTAF (zero-shot)"),("CoTAF_DN_L","CoTAF-DN-L (MR-NIRP 특화 합성, 비지도 적응)"),("CoTANcv_L","CoTAN-CV-L (실 라벨 3-fold in-domain)"),("FUSE","FUSE"),("FUSE_DN","FUSE-DN")]
print("MR-NIRP (전체 19명 / held-out 7~19) 정지·소·대동작:")
for m,l in rowsN:
    d=load(m)
    if not d.get("NIRP_STILL"): continue
    vals=[]
    for ax_ in ("NIRP_STILL","NIRP_SMALL","NIRP_LARGE"):
        e=errs(d,ax_); ho={k:v for k,v in e.items() if int(k)>=7}; vals.append(f"{np.mean(list(e.values())):.2f}/{np.mean(list(ho.values())):.2f}")
    print(f"  {l:45s} "+"  ".join(vals))
