"""실 PPG 파형 뱅크: UBFC 1~41 + PURE 01~06 학습 분할 GT에서 30초 세그먼트 추출 → 밴드패스·정규화·기본 주파수 f0 기록.
held-out(UBFC 42~49, PURE 07~10)은 사용하지 않음. 출력: physformer/ppg_bank.npz (segments[K,900]@30Hz, f0[K])"""
import glob, numpy as np
from scipy.signal import butter, filtfilt, periodogram, resample
ROOT="/SPECPHYS_ROOT"; FS=30; L=900; segs=[]; f0s=[]; srcs=[]
def add(sig, fs, src):
    if fs!=FS: sig=resample(sig, int(len(sig)*FS/fs))
    b,a=butter(2,[0.7/(FS/2),4.0/(FS/2)],btype="band")
    if len(sig)<L+60: return
    sig=filtfilt(b,a,sig-np.mean(sig))
    for st in range(0,len(sig)-L,L):
        s=sig[st:st+L]; f,p=periodogram(s,FS); band=(f>=0.7)&(f<=4.0)
        f0=f[band][p[band].argmax()]
        if not (0.75<=f0<=3.5): continue
        # 기본파가 지배적인 세그먼트만 (2f 지배 세그먼트 제외 — 라벨 안전)
        p2=p[np.argmin(np.abs(f-2*f0))]; 
        if p2>0.9*p[band].max(): continue
        s=(s-s.mean())/(np.abs(s).max()+1e-9); segs.append(s.astype(np.float32)); f0s.append(f0); srcs.append(src)
for g in sorted(glob.glob(f"{ROOT}/UBFC_dataset/DATASET_2/subject*/ground_truth.txt")):
    sid=int(g.split("subject")[-1].split("/")[0])
    if sid>41: continue
    add(np.array(open(g).read().split("\n")[0].split(),dtype=float),30,f"ubfc{sid}")
for g in sorted(glob.glob(f"{ROOT}/_workspace/data/physformer/PURE_TRAIN/subject*/ground_truth.txt")):
    add(np.array(open(g).read().split("\n")[0].split(),dtype=float),30,g.split("subject")[-1].split("/")[0])
np.savez(f"{ROOT}/_workspace/data/physformer/ppg_bank.npz",segments=np.array(segs),f0=np.array(f0s),src=np.array(srcs))
print(f"bank {len(segs)} segments, f0 {np.percentile(f0s,[10,50,90]).round(2)} Hz, sources {len(set(srcs))}")
