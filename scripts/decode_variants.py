"""모델 측 개선(재학습 없음): 저장된 예측 파형에 대해 HR 디코딩 변형을 비교.
 canonical : cumsum→detrend→butter 0.6–3.3→periodogram argmax (정본)
 harmguard : canonical 피크 f에 대해 f/2 대역 파워가 τ·peak 이상이면 f/2 채택 (harmonic doubling 방어; halving 방어는 2f 검사)
 winmed    : 30초 창(stride 10초)마다 canonical HR → 중앙값 (국소 붕괴에 강건)
 combo     : winmed + harmguard
출력: results/decode_variants.json + 표"""
import glob, json, pickle, re, sys, numpy as np, scipy.signal as ss
sys.path.insert(0,"/SPECPHYS_ROOT/scripts"); from agg_model2 import cat_chunks, fast_detrend, PF
sys.path.insert(0,"/SPECPHYS_ROOT/repos/rPPG-Toolbox"); from evaluation.post_process import _next_power_of_2
fs=30
def prep(sig):
    x=fast_detrend(np.cumsum(sig),100); b,a=ss.butter(1,[0.6/fs*2,3.3/fs*2],btype="bandpass"); return ss.filtfilt(b,a,np.double(x))
def spec(x):
    N=_next_power_of_2(len(x)); f,p=ss.periodogram(x,fs=fs,nfft=N,detrend=False); m=(f>=0.6)&(f<=3.3); return f[m],p[m]
def hr_canon(x): f,p=spec(x); return f[p.argmax()]*60
def hr_guard(x,tau=0.3,tau_hi=0.5):
    f,p=spec(x); i=p.argmax(); f0=f[i]; pk=p[i]
    def band(fc): w=(f>fc*0.92)&(f<fc*1.08); return (f[w][p[w].argmax()],p[w].max()) if w.any() else (None,0)
    fh,ph=band(f0/2)
    if fh is not None and ph>=tau*pk: return fh*60          # 2f였을 가능성 → 기본파
    return f0*60
def hr_win(x,fn,win=30,stride=10):
    n=len(x); W=win*fs; S=stride*fs
    if n<=W: return fn(x)
    return float(np.median([fn(x[s:s+W]) for s in range(0,n-W+1,S)]))
AXES=["MCD_B","MCD_A","PURE","REALTEST","RTNI","RTNC","DLCN","NIRP_STILL","NIRP_SMALL","NIRP_LARGE"]
MODELS=sys.argv[1:] or ["base","CoTAF","CoTAM_L","RhythmFormer","FactorizePhys","RFA","FFA"]
res={}
for m in MODELS:
    res[m]={}
    axes=AXES+["MMPD"]
    for ax in axes:
        pks=sorted(glob.glob(f"{PF}/runs/ev_{m}_MMPD_B*/*/saved_test_outputs/*.pickle")) if ax=="MMPD" else glob.glob(f"{PF}/runs/ev_{m}_{ax}/*/saved_test_outputs/*.pickle")
        errs={k:[] for k in ("canonical","harmguard","winmed","combo")}
        for pk in pks:
            d=pickle.load(open(pk,"rb"))
            for vid in d["predictions"]:
                if ax=="MMPD":
                    mm=re.match(r"subject(\d+)_",str(vid)); 
                    if not mm or int(mm.group(1))<7: continue
                if ax=="PURE":
                    mm=re.search(r"(\d+)$",str(vid)); 
                    if not mm or int(mm.group(1))<700: continue
                p=prep(cat_chunks(d["predictions"][vid])); l=prep(cat_chunks(d["labels"][vid])); gt=hr_canon(l)
                errs["canonical"].append(abs(hr_canon(p)-gt)); errs["harmguard"].append(abs(hr_guard(p)-gt)); errs["winmed"].append(abs(hr_win(p,hr_canon)-gt)); errs["combo"].append(abs(hr_win(p,hr_guard)-gt))
        if errs["canonical"]: res[m][ax]={k:float(np.mean(v)) for k,v in errs.items()}; res[m][ax]["n"]=len(errs["canonical"])
json.dump(res,open(f"{PF}/../results/decode_variants.json","w"),indent=1)
for m in MODELS:
    print(f"--- {m}"); 
    for ax,v in res[m].items(): print(f"  {ax:11s} n={v['n']:3d} canonical={v['canonical']:6.2f} harmguard={v['harmguard']:6.2f} winmed={v['winmed']:6.2f} combo={v['combo']:6.2f}")
