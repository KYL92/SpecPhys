#!/usr/bin/env python3
"""[체인 45 / 트랙 A] MMPD 원본(.mat)에서 **관측 가능한** 라우터 특징 추출 (GT 메타 라벨 불사용, CPU).
- Haar(HC) 얼굴 추적(STEP 프레임 간격): bbox 중심 변위 평균·최대, 중심 위치 std, 크기 변화율, 검출률, 얼굴 크기
- 프레임 차분 에너지(전체/얼굴 상자, 회색조), 얼굴 상자 휘도 평균·시간 std·저주파 드리프트·맥박대역 전력비
- POS(피부 마스크 평균, eval_classical_mmpd.py와 동일) 파형 → 정본 전처리 → HR·자기 피크 SNR
키 = (subject, light, motion, exercise, skin) — 메타는 **조인 키로만** 보관(특징 아님). 이어쓰기 지원.
출력: _workspace/data/results/mmpd_obs_features.json
사용: python mmpd_obs_features.py [N_WORKERS=2]
"""
import glob, json, os, sys
import numpy as np, cv2, scipy.io as sio, scipy.signal as ss
from multiprocessing import Pool
if not hasattr(np, "mat"): np.mat = np.asmatrix
sys.path.insert(0, "/SPECPHYS_ROOT/repos/rPPG-Toolbox")
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from unsupervised_methods.methods.POS_WANG import POS_WANG
from evaluation.post_process import _calculate_fft_hr
from agg_model2 import fast_detrend

SRC = "/SPECPHYS_RAW/MMPD"
OUT = "/SPECPHYS_ROOT/_workspace/data/results/mmpd_obs_features.json"
FPS, STEP = 30, 5
LIGHT = {"LED-low": 1, "LED-high": 2, "Incandescent": 3, "Nature": 4}
MOTION = {"Stationary": 1, "Stationary (after exercise)": 1, "Rotation": 2, "Talking": 3, "Walking": 4, "Watching Videos": 4}
CASC = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def tb_filter(x):
    x = fast_detrend(np.asarray(x, float) - np.mean(x), 100)
    b, a = ss.butter(1, [0.6 / FPS * 2, 3.3 / FPS * 2], btype="bandpass")
    return ss.filtfilt(b, a, x)

def hr_snr(x):
    """정본 전처리 후 HR(periodogram argmax) + 자기 피크 기준 SNR(toolbox _calculate_SNR 정의: ±6bpm 1·2고조파 / 나머지 0.6–3.3Hz)."""
    hr = float(_calculate_fft_hr(x, fs=FPS, low_pass=0.6, high_pass=3.3))
    N = 1 << int(np.ceil(np.log2(len(x))))
    f, p = ss.periodogram(x, fs=FPS, nfft=N, detrend=False)
    f0, dv = hr / 60, 0.1
    h1 = (f >= f0 - dv) & (f <= f0 + dv); h2 = (f >= 2 * f0 - dv) & (f <= 2 * f0 + dv)
    band = (f >= 0.6) & (f <= 3.3); rest = band & ~h1 & ~h2
    snr = 10 * np.log10((p[h1].sum() + p[h2].sum()) / (p[rest].sum() + 1e-12))
    return hr, float(snr)

def skin_series(frames):  # uint8 RGB (T,H,W,3) — eval_classical_mmpd.series_from_rgb와 동일
    acc = np.zeros(frames.shape[1:3], np.float32); sub = frames[::40]
    for fr in sub:
        y = cv2.cvtColor(fr, cv2.COLOR_RGB2YCrCb)
        acc += ((y[:, :, 1] >= 133) & (y[:, :, 1] <= 173) & (y[:, :, 2] >= 77) & (y[:, :, 2] <= 127)).astype(np.float32)
    m = (acc / len(sub)) > 0.5
    if m.sum() < 500:
        h, w = frames.shape[1:3]; m = np.zeros((h, w), bool); m[h//4:3*h//4, w//4:3*w//4] = True
    return [fr.astype(np.float32)[m].mean(0).reshape(1, 1, 3) for fr in frames], float(m.mean())

def features(mp):
    subj = int(mp.split("subject")[1].split("/")[0]); trial = int(mp.split("_")[-1][:-4])
    m = sio.loadmat(mp)
    v = (np.round(np.asarray(m["video"], np.float32) * 255)).astype(np.uint8); del m["video"]
    T, H, W, _ = v.shape
    gray = np.stack([cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY) for fr in v]).astype(np.float32)
    # --- Haar 추적 ---
    boxes, det = [], []
    last = None
    for t in range(0, T, STEP):
        fs = CASC.detectMultiScale(v[t, :, :, ::-1], 1.1, 4, minSize=(40, 40))
        if len(fs):
            x, y, w, h = max(fs, key=lambda b: b[2] * b[3]); last = (x + w / 2, y + h / 2, w, h); det.append(1)
        else:
            det.append(0)
        boxes.append(last)
    if all(b is None for b in boxes):
        boxes = [(W / 2, H / 2, W / 2, H / 2)] * len(boxes)
    first = next(b for b in boxes if b is not None); boxes = [b if b is not None else first for b in boxes]
    B = np.array(boxes, float)  # (n, 4) cx, cy, w, h
    disp = np.linalg.norm(np.diff(B[:, :2], axis=0), axis=1) / STEP  # px/frame
    size = np.log(B[:, 2] * B[:, 3])
    cx, cy, bw, bh = np.median(B, 0)
    x0, x1 = int(max(0, cx - bw / 2)), int(min(W, cx + bw / 2)); y0, y1 = int(max(0, cy - bh / 2)), int(min(H, cy + bh / 2))
    # --- 프레임 차분·휘도 ---
    d_all = np.abs(np.diff(gray, axis=0)).mean((1, 2))
    face_g = gray[:, y0:y1, x0:x1]
    d_face = np.abs(np.diff(face_g, axis=0)).mean((1, 2))
    lum = face_g.mean((1, 2))
    lum_ma = np.convolve(lum, np.ones(FPS) / FPS, mode="valid")
    f, p = ss.periodogram(lum - lum.mean(), fs=FPS)
    band = (f >= 0.6) & (f <= 3.3)
    # --- POS ---
    ser, skin_frac = skin_series(v)
    bvp = np.asarray(POS_WANG(ser, FPS)).squeeze()
    pos_hr, pos_snr = hr_snr(tb_filter(bvp))
    return dict(subject=subj, trial=trial, light=LIGHT[str(m["light"][0])], motion=MOTION[str(m["motion"][0])],
                exercise=1 if str(m["exercise"][0]) == "True" else 2, skin=int(m["skin_color"][0][0]), n_frames=int(T),
                # 관측 특징
                f_disp_mean=float(disp.mean()), f_disp_max=float(disp.max()), f_disp_p90=float(np.percentile(disp, 90)),
                f_center_std=float(B[:, :2].std(0).mean()), f_size_std=float(size.std()), f_det_rate=float(np.mean(det)),
                f_face_frac=float(bw * bh / (W * H)),
                f_diff_all=float(d_all.mean()), f_diff_face=float(d_face.mean()), f_diff_face_std=float(d_face.std()),
                f_lum_mean=float(lum.mean()), f_lum_std=float(lum.std()), f_lum_drift=float(lum_ma.std()),
                f_lum_band_ratio=float(p[band].sum() / (p[1:].sum() + 1e-12)), f_skin_frac=skin_frac,
                # POS 출력 내재 특징
                f_pos_hr=pos_hr, f_pos_snr=pos_snr)

def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["subject"], r["trial"]) for r in rows}
    mats = sorted(glob.glob(f"{SRC}/subject*/p*_*.mat"),
                  key=lambda p: (int(p.split("subject")[1].split("/")[0]), int(p.split("_")[-1][:-4])))
    todo = [mp for mp in mats if (int(mp.split("subject")[1].split("/")[0]), int(mp.split("_")[-1][:-4])) not in done]
    print(f"[obs] total {len(mats)} done {len(done)} todo {len(todo)} workers {nw}", flush=True)
    with Pool(nw) as pool:
        for i, r in enumerate(pool.imap_unordered(features, todo, chunksize=1)):
            rows.append(r)
            if i % 10 == 0 or i == len(todo) - 1:
                json.dump(rows, open(OUT, "w"))
            print(f"[obs] {i+1}/{len(todo)} subject{r['subject']} p{r['subject']}_{r['trial']} disp={r['f_disp_mean']:.2f} "
                  f"pos_hr={r['f_pos_hr']:.1f} snr={r['f_pos_snr']:.1f}", flush=True)
    json.dump(rows, open(OUT, "w"))
    print("OBS_FEATURES_DONE")

if __name__ == "__main__":
    main()
