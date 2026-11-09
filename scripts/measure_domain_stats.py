"""대상 데이터셋(MMPD 학습 분할 1~6)의 라벨-프리 도메인 통계 + GT 기반 심박/harmonic 통계 측정 → results/domain_stats_MMPD.json"""
import glob, json, sys, numpy as np, cv2, scipy.io as sio
from scipy.signal import welch, butter, filtfilt
sys.path.insert(0,"/SPECPHYS_ROOT/scripts"); from agg_model2 import fast_detrend
FPS=30; out={"hr":[], "a2":[], "a3":[], "flicker_hz":[], "flicker_amp":[], "drift":[], "motion_px":[], "motion_hz":[], "sensor_std":[], "skin_rgb":[], "n":0}
cas=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
def hr_harm(ppg):
    x=fast_detrend(ppg-ppg.mean(),100); b,a=butter(1,[0.6/15,3.3/15],btype="band"); x=filtfilt(b,a,x)
    fr=np.fft.rfftfreq(len(x)*8,1/FPS); pw=np.abs(np.fft.rfft(x,n=len(x)*8))**2; band=(fr>=0.6)&(fr<=3.3); f0=fr[band][pw[band].argmax()]
    raw=ppg-ppg.mean(); fr2=np.fft.rfftfreq(len(raw)*8,1/FPS); pw2=np.abs(np.fft.rfft(raw,n=len(raw)*8))
    def pk(f): w=(fr2>f*0.93)&(fr2<f*1.07); return pw2[w].max() if w.any() else 0
    return f0*60, pk(2*f0)/max(pk(f0),1e-9), pk(3*f0)/max(pk(f0),1e-9)
for mp in sorted(glob.glob("/SPECPHYS_RAW/MMPD/subject[1-6]/p*_*.mat")):
    m=sio.loadmat(mp); v=(np.asarray(m["video"],np.float32)*255).astype(np.uint8); ppg=np.asarray(m["GT_ppg"],float).reshape(-1)[:len(v)]
    hr,a2,a3=hr_harm(ppg); out["hr"].append(hr); out["a2"].append(a2); out["a3"].append(a3)
    # 조명: 배경/전체 프레임 평균 밝기 시계열 (얼굴 밖 영역 = 프레임 가장자리 띠)
    g=v.mean(axis=3); edge=np.concatenate([g[:,:20,:].reshape(len(v),-1),g[:,-20:,:].reshape(len(v),-1)],1).mean(1); rel=edge/edge.mean()
    f,p=welch(rel-rel.mean(),FPS,nperseg=512); band=(f>0.5)&(f<8); fpk=f[band][p[band].argmax()]; amp=np.sqrt(p[band].max()*(f[1]-f[0]))*np.sqrt(2)
    out["flicker_hz"].append(float(fpk)); out["flicker_amp"].append(float(amp)); out["drift"].append(float(rel.max()-rel.min()))
    # 움직임: 30프레임마다 Haar 얼굴 상자 중심 궤적
    cx=[]; 
    for i in range(0,len(v),15):
        fc=cas.detectMultiScale(cv2.cvtColor(v[i],cv2.COLOR_RGB2GRAY),1.1,4)
        if len(fc): x,y,w,h=max(fc,key=lambda r:r[2]*r[3]); cx.append((x+w/2,y+h/2,w))
    if len(cx)>10:
        c=np.array(cx); disp=np.sqrt(((c[:,:2]-c[:,:2].mean(0))**2).sum(1)); out["motion_px"].append(float(np.percentile(disp,90)*256/c[:,2].mean()))  # 얼굴폭 256px 기준 정규화
        d=c[:,1]-c[:,1].mean(); f2,p2=welch(d,2,nperseg=min(64,len(d))); out["motion_hz"].append(float(f2[1:][p2[1:].argmax()]))
    # 센서 노이즈: 평탄 배경 패치의 시간 표준편차(프레임 간 차분)
    patch=v[:,5:25,5:25,:].astype(np.float32).reshape(len(v),-1); out["sensor_std"].append(float(np.diff(patch,axis=0).std()/np.sqrt(2)))
    # 피부톤: 얼굴 상자 내부 중앙 평균 RGB
    if cx: x,y,w=c[0]; out["skin_rgb"].append(v[0,int(y-w*0.1):int(y+w*0.1),int(x-w*0.15):int(x+w*0.15)].reshape(-1,3).mean(0).tolist())
    out["n"]+=1
summ={k:(dict(p10=float(np.percentile(v,10)),p50=float(np.percentile(v,50)),p90=float(np.percentile(v,90)),min=float(np.min(v)),max=float(np.max(v))) if k not in("skin_rgb","n") and len(v) else None) for k,v in out.items()}
summ["skin_rgb_mean"]=np.mean(out["skin_rgb"],0).tolist(); summ["n"]=out["n"]; summ["resolution"]="320x240 (H×W), mobile"
json.dump(dict(summary=summ,raw=out),open("/SPECPHYS_ROOT/_workspace/data/results/domain_stats_MMPD.json","w"),indent=1)
for k,v in summ.items():
    if isinstance(v,dict): print(f"{k:12s} p10={v['p10']:.3f} p50={v['p50']:.3f} p90={v['p90']:.3f} (min {v['min']:.3f}, max {v['max']:.3f})")
print("skin_rgb_mean",[round(x) for x in summ["skin_rgb_mean"]],"n",summ["n"])
