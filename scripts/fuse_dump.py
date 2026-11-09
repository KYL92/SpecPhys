"""상보 백본 스펙트럼 융합(FUSE = CoTAF^α · FactorizePhys^(1-α), α=0.6)의 per-video 결과를 agg 포맷으로 저장 + CoTAF 대비 Wilcoxon"""
import glob, json, pickle, re, sys, numpy as np, scipy.signal as ss
from scipy.stats import wilcoxon
sys.path.insert(0,"/SPECPHYS_ROOT/scripts"); from agg_model2 import cat_chunks, fast_detrend, PF
sys.path.insert(0,"/SPECPHYS_ROOT/repos/rPPG-Toolbox"); from evaluation.post_process import _next_power_of_2
fs=30; AL=float(sys.argv[1]) if len(sys.argv)>1 else 0.6; A=sys.argv[2] if len(sys.argv)>2 else "CoTAF"; B="FactorizePhys"; TAG=sys.argv[3] if len(sys.argv)>3 else "FUSE"; R=f"{PF}/../results"
def prep(sig):
    x=fast_detrend(np.cumsum(sig),100); b,a=ss.butter(1,[0.6/fs*2,3.3/fs*2],btype="bandpass"); return ss.filtfilt(b,a,np.double(x))
def psd(x,N):
    f,p=ss.periodogram(x,fs=fs,nfft=N,detrend=False); m=(f>=0.6)&(f<=3.3); return f[m],p[m]/p[m].sum()
def load(m,ax):
    pks=sorted(glob.glob(f"{PF}/runs/ev_{m}_MMPD_B*/*/saved_test_outputs/*.pickle")) if ax=="MMPD" else glob.glob(f"{PF}/runs/ev_{m}_{ax}/*/saved_test_outputs/*.pickle"); out={}
    for pk in pks:
        d=pickle.load(open(pk,"rb"))
        for vid in d["predictions"]: out[str(vid)]=(cat_chunks(d["predictions"][vid]),cat_chunks(d["labels"][vid]))
    return out
AXES=["MCD_B","MCD_A","PURE","REALTEST","RTNI","RTNC","DLCN","NIRP_STILL","NIRP_SMALL","NIRP_LARGE","MMPD"]
out={}; mm=[]; stats=[]
for ax in AXES:
    PA=load(A,ax); PB=load(B,ax); rows=[]; ea=[]; ef=[]
    for k in sorted(set(PA)&set(PB)):
        pa,la=PA[k]; pb,_=PB[k]; n=min(len(pa),len(pb)); N=_next_power_of_2(n); xa=prep(pa[:n]); xb=prep(pb[:n]); xl=prep(la[:n])
        f,ga=psd(xa,N); _,gb=psd(xb,N); _,gl=psd(xl,N); gt=f[gl.argmax()]*60; hf=f[(ga**AL*gb**(1-AL)).argmax()]*60; ha=f[ga.argmax()]*60
        if ax=="MMPD":
            m=re.match(r"subject(\d+)_L(\d)_MO(\d)_E(\d)_S(\d)",k); mm.append(dict(vid=k,subject=int(m.group(1)),light=int(m.group(2)),motion=int(m.group(3)),exercise=int(m.group(4)),skin=int(m.group(5)),gt=gt,pred=hf,err=abs(hf-gt),ratio=hf/gt))
            if int(m.group(1))<7: continue
        else:
            vn=int(re.search(r"(\d+)$",k).group(1)) if re.search(r"(\d+)$",k) else -1
            rows.append(dict(vid=vn,gt=gt,pred=hf,err=abs(hf-gt),gt_h=None,pred_h=None,err_h=None,ratio=hf/gt))
        ea.append(abs(ha-gt)); ef.append(abs(hf-gt))
    if ax!="MMPD": out[ax]=rows
    x=np.array(ea); y=np.array(ef); p=wilcoxon(x,y)[1] if len(x)>=5 and not np.all(x==y) else float("nan")
    stats.append(dict(label=f"{ax}: {A} vs {TAG}",n=len(x),mae_a=float(x.mean()),mae_b=float(y.mean()),p=float(p))); print(f"{ax:11s} n={len(x):3d} {A}={x.mean():6.2f} {TAG}={y.mean():6.2f} p={p:.3g}")
json.dump(out,open(f"{R}/agg_{TAG}.json","w"),indent=1); json.dump(mm,open(f"{R}/agg_{TAG}_MMPD.json","w"),indent=0); json.dump(stats,open(f"{R}/stats_{TAG.lower()}.json","w"),indent=1)
