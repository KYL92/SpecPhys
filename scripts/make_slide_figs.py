import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"]="Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"]=False
R="/SPECPHYS_ROOT/_workspace/data/results"; S="/SPECPHYS_ROOT/_workspace/slides"
C_OURS="#b3312e"; C_BASE="#8a8a8a"; C_POS="#2e7d4f"; C_OTHER="#4a6fa5"
def load(m):
    d={}
    for suf in ("","_newsets"):
        p=f"{R}/agg_{m}{suf}.json"
        if os.path.exists(p): d.update(json.load(open(p)))
    return d
def mae(m,ax,sub=None):
    rows=load(m).get(ax,[]); rows=[r for r in rows if sub(r)] if sub else rows
    return np.mean([r["err"] for r in rows]) if rows else np.nan
axes=[("PURE","PURE\n(07–10)",lambda r:r["vid"]>=700),("MCD_B","MCD\n안정",None),("MCD_A","MCD\n운동후",None),("DLCN","DLCN",None),("NIRP_STILL","운전\n정지",None),("NIRP_SMALL","운전\n소동작",None),("NIRP_LARGE","운전\n대동작",None)]
pos={"PURE":2.74,"MCD_B":4.75,"MCD_A":12.17,"DLCN":2.55,"NIRP_STILL":15.08,"NIRP_SMALL":13.04,"NIRP_LARGE":9.48}
mm=lambda m: np.mean([r["err"] for r in json.load(open(f"{R}/agg_{m}_MMPD.json")) if r["subject"]>=7])
labels=[a[1] for a in axes]+["MMPD\n(7–33)"]
b=[mae("base",a,s) for a,_,s in axes]+[mm("base")]; o=[mae("CoTAF",a,s) for a,_,s in axes]+[mm("CoTAF")]; p=[pos[a] for a,_,_ in axes]+[mm("POS")]; rf=[mae("RhythmFormer_arch",a,s) for a,_,s in axes]+[mm("RhythmFormer")]
x=np.arange(len(labels)); w=0.2
fig,ax=plt.subplots(figsize=(12,4.6))
ax.bar(x-1.5*w,b,w,color=C_BASE,label="PhysFormer (UBFC 사전학습)"); ax.bar(x-0.5*w,rf,w,color=C_OTHER,label="RhythmFormer (UBFC)"); ax.bar(x+0.5*w,p,w,color=C_POS,label="POS (고전)"); ax.bar(x+1.5*w,o,w,color=C_OURS,label="PhysFormer + 가상얼굴 코퍼스 (CoTAF)")
for i,v in enumerate(o): ax.text(x[i]+1.5*w,v+0.4,f"{v:.1f}",ha="center",fontsize=9,color=C_OURS,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=10); ax.set_ylabel("HR MAE (BPM)"); ax.set_title("6개 실데이터셋 교차 평가 — 같은 백본, 학습 분포만 바꿈",fontsize=13)
ax.legend(fontsize=9,ncol=2,frameon=False); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.3)
fig.tight_layout(); fig.savefig(f"{S}/figA_cross.png",dpi=200); plt.close()
LIGHT={1:"LED-low",2:"LED-high",3:"Incand.",4:"Nature"}; MOTION={1:"Stationary",2:"Rotation",3:"Talking",4:"Walking"}; SKIN={3:"Skin 3",4:"Skin 4",5:"Skin 5",6:"Skin 6"}
cols=[("light",LIGHT),("motion",MOTION),("skin",SKIN)]
models=[("base","PhysFormer(UBFC)"),("RhythmFormer","RhythmFormer(UBFC)"),("FactorizePhys","FactorizePhys(UBFC)"),("POS","POS"),("CoTAF","CoTAF (zero-shot)"),("CoTAM_L","CoTAM-L (MMPD 1–6 + label-fix)")]
names=[nm for _,d in cols for nm in d.values()]; M=[]
for m,_ in models:
    rows=[r for r in json.load(open(f"{R}/agg_{m}_MMPD.json")) if r["subject"]>=7]
    M.append([np.mean([r["err"] for r in rows if r[key]==v]) for key,d in cols for v in d])
M=np.array(M)
fig,ax=plt.subplots(figsize=(12,3.8)); im=ax.imshow(M,cmap="RdYlGn_r",vmin=3,vmax=25,aspect="auto")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=30,ha="right",fontsize=9); ax.set_yticks(range(len(models))); ax.set_yticklabels([n for _,n in models],fontsize=10)
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j,i,f"{M[i,j]:.1f}",ha="center",va="center",fontsize=8.5,color="black")
ax.set_title("MMPD held-out(피험자 7–33, 540클립) 조건별 HR MAE — 조명 · 동작 · 피부톤",fontsize=12); plt.colorbar(im,ax=ax,fraction=0.02,pad=0.01,label="MAE")
fig.tight_layout(); fig.savefig(f"{S}/figB_mmpd_heat.png",dpi=200); plt.close()
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.2),gridspec_kw=dict(width_ratios=[1,1.3]))
ds=["MMPD","PURE","MCD","UBFC","합성"]; leg=[51.7,19.2,19.7,10.4,0.0]; fix=[0.5,0.0,0.0,0.0,0.0]; x=np.arange(5)
a1.bar(x-0.2,leg,0.4,color=C_OURS,label="legacy (Welch on diff)"); a1.bar(x+0.2,fix,0.4,color=C_POS,label="수정 (cumsum→detrend→bandpass)")
for i,v in enumerate(leg): a1.text(x[i]-0.2,v+1,f"{v:.1f}%",ha="center",fontsize=9)
a1.set_xticks(x); a1.set_xticklabels(ds); a1.set_ylabel("2f로 잡힌 학습 라벨 청크 비율 (%)"); a1.set_title("학습 HR 라벨의 harmonic doubling 감사"); a1.legend(frameon=False,fontsize=9); a1.spines[["top","right"]].set_visible(False)
cases=["CoTN\nMCD 안정","CoTAM\nMMPD 학습분할","CoTAM\n합성 C3","CoTB\n합성 C3","CoTAM\nMMPD held-out"]; before=[42.63,22.31,33.46,17.58,13.95]; after=[18.41,1.78,0.12,0.12,9.98]; x=np.arange(5)
a2.bar(x-0.2,before,0.4,color=C_OURS,label="legacy 라벨"); a2.bar(x+0.2,after,0.4,color=C_POS,label="라벨 수정 (-L)")
for i in range(5): a2.text(x[i]-0.2,before[i]+0.6,f"{before[i]:.1f}",ha="center",fontsize=9); a2.text(x[i]+0.2,after[i]+0.6,f"{after[i]:.2f}",ha="center",fontsize=9,color=C_POS,fontweight="bold")
a2.set_xticks(x); a2.set_xticklabels(cases,fontsize=9); a2.set_ylabel("HR MAE (BPM)"); a2.set_title("같은 데이터·설정, 라벨만 수정"); a2.legend(frameon=False,fontsize=9); a2.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{S}/figC_label.png",dpi=200); plt.close()
lin=[("base","base"),("CoTN","+노이즈\nco-train"),("CoTNH","+harmonic\n다양화"),("CoTNHM","+고심박\n+MCD"),("CoTA","+PURE"),("CoTAF","+저심박\n(CoTAF)")]
fig,ax=plt.subplots(figsize=(12,4.2))
for axn,lab,col,sub in [("MCD_B","MCD 안정",C_OURS,None),("MCD_A","MCD 운동후","#e08a2e",None),("PURE","PURE held-out",C_OTHER,lambda r:r["vid"]>=700),("DLCN","DLCN (zero-shot)",C_POS,None)]:
    ys=[mae(m,axn,sub) for m,_ in lin]; ax.plot(range(6),ys,marker="o",lw=2,color=col,label=lab)
    for i,v in enumerate(ys):
        if not np.isnan(v): ax.text(i,v+1,f"{v:.1f}",ha="center",fontsize=8,color=col)
ax.set_xticks(range(6)); ax.set_xticklabels([l for _,l in lin],fontsize=10); ax.set_ylabel("HR MAE (BPM)"); ax.set_title("코퍼스를 한 축씩 채울 때의 궤적 (동일 PhysFormer 백본)",fontsize=13); ax.legend(frameon=False,fontsize=9); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.3)
fig.tight_layout(); fig.savefig(f"{S}/figD_lineage.png",dpi=200); plt.close()
print("figs ok")
