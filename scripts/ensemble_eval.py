"""모델 측 개선 후보 2: 스펙트럼 융합 앙상블(재학습 없음). 두 모델의 정규화 파워 스펙트럼을 합쳐 argmax → HR.
PhysFormer(CoTAF)의 커버리지와 FactorizePhys의 강건성이 상보적인지 검사."""
import glob, json, pickle, re, sys, numpy as np, scipy.signal as ss
sys.path.insert(0,"/SPECPHYS_ROOT/scripts"); from agg_model2 import cat_chunks, fast_detrend, PF
sys.path.insert(0,"/SPECPHYS_ROOT/repos/rPPG-Toolbox"); from evaluation.post_process import _next_power_of_2
fs=30
def prep(sig):
    x=fast_detrend(np.cumsum(sig),100); b,a=ss.butter(1,[0.6/fs*2,3.3/fs*2],btype="bandpass"); return ss.filtfilt(b,a,np.double(x))
def psd(x,N):
    f,p=ss.periodogram(x,fs=fs,nfft=N,detrend=False); m=(f>=0.6)&(f<=3.3); return f[m],p[m]/p[m].sum()
def load_preds(m,ax):
    pks=sorted(glob.glob(f"{PF}/runs/ev_{m}_MMPD_B*/*/saved_test_outputs/*.pickle")) if ax=="MMPD" else glob.glob(f"{PF}/runs/ev_{m}_{ax}/*/saved_test_outputs/*.pickle"); out={}
    for pk in pks:
        d=pickle.load(open(pk,"rb"))
        for vid in d["predictions"]:
            if ax=="MMPD":
                mm=re.match(r"subject(\d+)_",str(vid)); 
                if not mm or int(mm.group(1))<7: continue
            if ax=="PURE":
                mm=re.search(r"(\d+)$",str(vid)); 
                if not mm or int(mm.group(1))<700: continue
            out[str(vid)]=(prep(cat_chunks(d["predictions"][vid])),prep(cat_chunks(d["labels"][vid])))
    return out
AXES=["MCD_B","MCD_A","PURE","REALTEST","RTNI","RTNC","DLCN","NIRP_STILL","NIRP_SMALL","NIRP_LARGE","MMPD"]
pairs=[("CoTAF","FactorizePhys")]
res={}
for a,b in pairs:
    key=f"{a}+{b}"; res[key]={}
    for ax in AXES:
        A=load_preds(a,ax); B=load_preds(b,ax); ks=sorted(set(A)&set(B))
        if not ks: continue
        ea=[];eb=[];ee=[];ep=[];eg=[]; ew={al:[] for al in (0.6,0.7,0.8,0.9)}
        for k in ks:
            pa,la=A[k]; pb,lb=B[k]; n=min(len(pa),len(pb)); N=_next_power_of_2(n)
            f,ga=psd(pa[:n],N); _,gb=psd(pb[:n],N); _,gl=psd(la[:n],N); gt=f[gl.argmax()]*60
            ea.append(abs(f[ga.argmax()]*60-gt)); eb.append(abs(f[gb.argmax()]*60-gt)); ee.append(abs(f[(ga+gb).argmax()]*60-gt)); ep.append(abs(f[(ga*gb).argmax()]*60-gt)); eg.append(abs(f[(np.sqrt(ga)+np.sqrt(gb)).argmax()]*60-gt))
            for al in ew: ew[al].append(abs(f[(ga**al*gb**(1-al)).argmax()]*60-gt))
        res[key][ax]=dict(n=len(ks),a=float(np.mean(ea)),b=float(np.mean(eb)),ens_sum=float(np.mean(ee)),ens_prod=float(np.mean(ep)),ens_sqrt=float(np.mean(eg)),**{f'w{al}':float(np.mean(v)) for al,v in ew.items()})
        print(f"{key:22s} {ax:11s} n={len(ks):3d} {a}={np.mean(ea):6.2f} {b}={np.mean(eb):6.2f} sum={np.mean(ee):6.2f} prod={np.mean(ep):6.2f} sqrt={np.mean(eg):6.2f} "+" ".join(f"w{al}={np.mean(v):5.2f}" for al,v in ew.items()))
json.dump(res,open(f"{PF}/../results/ensemble_eval.json","w"),indent=1)
