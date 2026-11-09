"""N-way 스펙트럼 융합: python fuseN_dump.py TAG modelA:wA modelB:wB [modelC:wC ...]  (가중 기하평균 P=∏P_i^w_i, argmax). 기준 비교: 첫 모델"""
import glob, json, pickle, re, sys, numpy as np, scipy.signal as ss
from scipy.stats import wilcoxon
sys.path.insert(0,"/SPECPHYS_ROOT/scripts"); from agg_model2 import cat_chunks, fast_detrend, PF
sys.path.insert(0,"/SPECPHYS_ROOT/repos/rPPG-Toolbox"); from evaluation.post_process import _next_power_of_2
fs=30; TAG=sys.argv[1]; MS=[(a.split(":")[0],float(a.split(":")[1])) for a in sys.argv[2:]]; R=f"{PF}/../results"
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
    P=[load(m,ax) for m,_ in MS]; keys=sorted(set.intersection(*[set(p) for p in P])); rows=[]; ea=[]; ef=[]
    for k in keys:
        n=min(len(p[k][0]) for p in P); N=_next_power_of_2(n); la=P[0][k][1]; xl=prep(la[:n]); f,gl=psd(xl,N); gt=f[gl.argmax()]*60
        G=None
        for (m,w),p in zip(MS,P):
            _,g=psd(prep(p[k][0][:n]),N); G=g**w if G is None else G*g**w
            if m==MS[0][0]: ha=f[g.argmax()]*60
        hf=f[G.argmax()]*60
        if ax=="MMPD":
            mo=re.match(r"subject(\d+)_L(\d)_MO(\d)_E(\d)_S(\d)",k); mm.append(dict(vid=k,subject=int(mo.group(1)),light=int(mo.group(2)),motion=int(mo.group(3)),exercise=int(mo.group(4)),skin=int(mo.group(5)),gt=gt,pred=hf,err=abs(hf-gt),ratio=hf/gt))
            if int(mo.group(1))<7: continue
        else:
            vn=int(re.search(r"(\d+)$",k).group(1)) if re.search(r"(\d+)$",k) else -1; rows.append(dict(vid=vn,gt=gt,pred=hf,err=abs(hf-gt),gt_h=None,pred_h=None,err_h=None,ratio=hf/gt))
        ea.append(abs(ha-gt)); ef.append(abs(hf-gt))
    if ax!="MMPD": out[ax]=rows
    x=np.array(ea); y=np.array(ef); p=wilcoxon(x,y)[1] if len(x)>=5 and not np.all(x==y) else float("nan")
    stats.append(dict(label=f"{ax}: {MS[0][0]} vs {TAG}",n=len(x),mae_a=float(x.mean()) if len(x) else None,mae_b=float(y.mean()) if len(y) else None,p=float(p))); print(f"{ax:11s} n={len(x):3d} {MS[0][0]}={x.mean() if len(x) else float('nan'):6.2f} {TAG}={y.mean() if len(y) else float('nan'):6.2f} p={p:.3g}")
json.dump(out,open(f"{R}/agg_{TAG}.json","w"),indent=1); json.dump(mm,open(f"{R}/agg_{TAG}_MMPD.json","w"),indent=0); json.dump(stats,open(f"{R}/stats_{TAG.lower()}.json","w"),indent=1)
