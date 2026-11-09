#!/usr/bin/env python3
"""19단계(체인 36) ①: 실 PPG 파형 뱅크 v2 — v1(UBFC/PURE 학습분할 131) + PPG-only 공개 데이터셋 확장.

추가 소스 (영상 불필요, GT PPG만 사용):
  BIDMC (PhysioNet, ICU fingertip PLETH 125Hz + Numerics HR): 형태 다양성(고령·임상), HR 라벨 일치 게이트
  PPG-DaLiA (UCI, 손목 BVP 64Hz + ECG 기반 HR 라벨): 운동 구간의 실 고심박 형태, HR 라벨 일치 게이트

v1과 동일한 품질 게이트(기본파 지배, 2f 지배 제외) + |f0·60 − HR라벨| ≤ 8 BPM.
소스별 상한·HR 층화로 특정 소스 지배 방지. 출력: physformer/ppg_bank_v2.npz (segments[K,900]@30Hz, f0, src)
"""
import glob, io, os, pickle, zipfile
import numpy as np
from scipy.signal import butter, filtfilt, periodogram, resample

ROOT = "/SPECPHYS_ROOT"
PF = f"{ROOT}/_workspace/data/physformer"
EXT = f"{ROOT}/_workspace/data/ppg_ext"
FS, L = 30, 900
segs, f0s, srcs = [], [], []

def gate_and_add(sig, fs, src, hr_label=None, max_seg=999):
    """v1과 동일 게이트로 30초 세그먼트 추출. hr_label(BPM)이 있으면 f0 일치 게이트 추가."""
    if fs != FS: sig = resample(sig, int(len(sig) * FS / fs))
    if len(sig) < L + 60: return 0
    b, a = butter(2, [0.7 / (FS / 2), 4.0 / (FS / 2)], btype="band")
    sig = filtfilt(b, a, sig - np.mean(sig))
    n = 0
    for st in range(0, len(sig) - L, L):
        if n >= max_seg: break
        s = sig[st:st + L]
        f, p = periodogram(s, FS); band = (f >= 0.7) & (f <= 4.0)
        if not band.any() or p[band].max() <= 0: continue
        f0 = f[band][p[band].argmax()]
        if not (0.6 <= f0 <= 3.5): continue
        p2 = p[np.argmin(np.abs(f - 2 * f0))]
        if p2 > 0.9 * p[band].max(): continue
        if hr_label is not None:
            hl = hr_label(st / FS, (st + L) / FS)
            if hl is None or abs(f0 * 60 - hl) > 8: continue
        s = (s - s.mean()) / (np.abs(s).max() + 1e-9)
        segs.append(s.astype(np.float32)); f0s.append(float(f0)); srcs.append(src); n += 1
    return n

# ---- v1 그대로 (UBFC/PURE 학습분할) ----
v1 = np.load(f"{PF}/ppg_bank.npz")
for s, f, sr in zip(v1["segments"], v1["f0"], v1["src"]):
    segs.append(np.asarray(s, np.float32)); f0s.append(float(f)); srcs.append(str(sr))
print(f"v1 carried: {len(segs)}")

# ---- BIDMC (csv 125Hz PLETH + Numerics HR 1Hz) ----
bz = f"{EXT}/bidmc.zip"
if os.path.exists(bz):
    n0 = len(segs)
    with zipfile.ZipFile(bz) as z:
        sig_files = sorted(p for p in z.namelist() if p.endswith("_Signals.csv"))
        for p in sig_files:
            rid = p.split("bidmc_")[-1].split("_")[0]
            with z.open(p) as fh:
                head = fh.readline().decode().strip().split(",")
                head = [h.strip() for h in head]
                if "PLETH" not in head: continue
                col = head.index("PLETH")
                data = np.loadtxt(io.TextIOWrapper(fh), delimiter=",", usecols=col)
            nump = p.replace("_Signals.csv", "_Numerics.csv")
            hr_t = hr_v = None
            if nump in z.namelist():
                with z.open(nump) as fh:
                    h2 = [h.strip() for h in fh.readline().decode().strip().split(",")]
                    if "HR" in h2:
                        arr = np.loadtxt(io.TextIOWrapper(fh), delimiter=",",
                                         usecols=(h2.index("Time [s]") if "Time [s]" in h2 else 0, h2.index("HR")))
                        hr_t, hr_v = arr[:, 0], arr[:, 1]
            def hl(t0, t1):
                if hr_v is None: return None
                m = (hr_t >= t0) & (hr_t <= t1)
                v = hr_v[m]; v = v[~np.isnan(v)]
                return float(v.mean()) if len(v) else None
            gate_and_add(data, 125, f"bidmc{rid}", hr_label=hl, max_seg=8)
    print(f"BIDMC added: {len(segs) - n0}")
else:
    print("BIDMC zip 없음 — 건너뜀")

# ---- PPG-DaLiA (S*.pkl, wrist BVP 64Hz + label HR(8s창, 2s shift)) ----
dz = f"{EXT}/dalia.zip"
if os.path.exists(dz):
    n0 = len(segs)
    with zipfile.ZipFile(dz) as z:
        pkls = sorted(p for p in z.namelist() if p.endswith(".pkl") and "/S" in p)
        for p in pkls:
            sid = os.path.basename(p).split(".")[0]
            try:
                d = pickle.loads(z.read(p), encoding="latin1")
            except Exception as e:
                print(f"  {sid} load fail: {e}"); continue
            bvp = np.asarray(d["signal"]["wrist"]["BVP"], float).ravel()
            hr = np.asarray(d["label"], float).ravel()  # 8초 창, 2초 간격
            def hl(t0, t1):
                i0, i1 = int(max(0, (t0 - 4) / 2)), int((t1 - 4) / 2) + 1
                v = hr[max(0, i0):min(len(hr), i1)]
                return float(v.mean()) if len(v) else None
            # HR 층화: 세그먼트 게이트는 gate_and_add에 위임, 상한만 소스별로
            gate_and_add(bvp, 64, f"dalia_{sid}", hr_label=hl, max_seg=60)
    print(f"DaLiA added: {len(segs) - n0}")
else:
    print("DaLiA zip 없음 — 건너뜀")

F = np.array(f0s)
np.savez(f"{PF}/ppg_bank_v2.npz", segments=np.array(segs), f0=F, src=np.array(srcs))
bins = [0.6, 1.0, 1.33, 1.67, 2.0, 2.5, 3.5]
hist = np.histogram(F, bins)[0]
print(f"BANKV2 total={len(segs)} sources={len(set(srcs))} "
      f"f0[bpm] p10/50/90={np.percentile(F * 60, [10, 50, 90]).round(0)} "
      f"BPM분포 {[f'{int(bins[i]*60)}-{int(bins[i+1]*60)}:{hist[i]}' for i in range(len(hist))]}")
