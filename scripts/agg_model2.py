#!/usr/bin/env python3
"""정본 프로토콜 집계 (동료 세션과 합의된 정의):
- 실데이터 축(MCD_B/MCD_A/PURE/REALTEST/RTNI/RTNC): rPPG-Toolbox 프로토콜
  cumsum → detrend(λ=100) → butter1 0.6–3.3Hz → periodogram (_calculate_fft_hr)
- 합성 축(C3/C6/E3): phase2 고해상 FFT (cumsum→butter3 0.7–4.0→8x zero-pad rfft)
- 보조 열로 고해상 hi-res 병기, 배가/반감은 정본 기준 |ratio−2|<0.2 / |ratio−0.5|<0.1
사용: conda run -n vface-rppg python agg_model2.py MODEL [MODEL...]
"""
import glob, json, pickle, re, sys, os
import numpy as np, torch
import scipy.signal as ss
from scipy.signal import butter, filtfilt

sys.path.insert(0, "/SPECPHYS_ROOT/repos/rPPG-Toolbox")
from evaluation.post_process import _calculate_fft_hr, _detrend  # noqa: E402

fs = 30
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
REAL_AXES = ["MCD_B", "MCD_A", "PURE", "REALTEST", "RTNI", "RTNC", "UBFC", "DLCN", "NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE"]
SYN_AXES = ["C3", "C6", "E3"]


def cat_chunks(chunks):
    return np.concatenate([chunks[k].detach().cpu().float().numpy().ravel() for k in sorted(chunks)])


def fast_detrend(x, lam=100):
    """toolbox _detrend와 수치 동치 (밀집 역행렬 대신 희소 해법 — O(N)):
    filtered = x - solve(I + λ²D'D, x)"""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    n = len(x)
    ones = np.ones(n)
    D = sparse.spdiags([ones, -2 * ones, ones], [0, 1, 2], n - 2, n)
    A = sparse.identity(n) + lam ** 2 * (D.T @ D)
    return x - spsolve(sparse.csc_matrix(A), x)


def toolbox_hr(sig, lo=0.6, hi=3.3):
    x = fast_detrend(np.cumsum(sig), 100)
    b, a = ss.butter(1, [lo / fs * 2, hi / fs * 2], btype="bandpass")
    x = ss.filtfilt(b, a, np.double(x))
    return float(_calculate_fft_hr(x, fs=fs, low_pass=lo, high_pass=hi))


def toolbox_hr_nodiff(sig, lo=0.6, hi=3.3):
    """[2026-10-17 정정] 표준화(Standardized) 라벨 모델(RhythmFormer·FactorizePhys 등): 툴박스 post_process 는 diff_flag=False 로 누적합 없이
    detrend→대역통과→periodogram. 이전 집계는 전 모델에 누적합을 적용해 이 모델들의 읽기가 툴박스 규칙과 어긋났다(33 문서 §5)."""
    x = fast_detrend(np.asarray(sig, float), 100)
    b, a = ss.butter(1, [lo / fs * 2, hi / fs * 2], btype="bandpass")
    x = ss.filtfilt(b, a, np.double(x))
    return float(_calculate_fft_hr(x, fs=fs, low_pass=lo, high_pass=hi))


def hr_for(sig, label_type):
    """pickle 의 label_type(툴박스 저장) 에 따라 읽기 선택: DiffNormalized → 누적합(정본), Standardized/Raw → 누적합 없음. RPPG_AGG_FORCE_CUMSUM=1 이면 구 방식 재현."""
    import os as _os
    if _os.environ.get("RPPG_AGG_FORCE_CUMSUM", "0") == "1" or label_type in (None, "DiffNormalized"):
        return toolbox_hr(sig)
    return toolbox_hr_nodiff(sig)


def hires_hr(sig):
    sig = np.cumsum(sig - sig.mean())
    b, a = butter(3, [0.7 / (fs / 2), 4.0 / (fs / 2)], btype="band")
    f = filtfilt(b, a, sig)
    fr = np.fft.rfftfreq(len(f) * 8, 1 / fs)
    pw = np.abs(np.fft.rfft(f - f.mean(), n=len(f) * 8)) ** 2
    band = (fr >= 0.7) & (fr <= 4.0)
    return float(fr[band][pw[band].argmax()] * 60)


def vidnum(v):
    m = re.search(r"(\d+)$", str(v))
    return int(m.group(1)) if m else -1


def agg(model, dump=None):
    out = {}
    for ax in REAL_AXES + SYN_AXES:
        pk = glob.glob(f"{PF}/runs/ev_{model}_{ax}/*/saved_test_outputs/*.pickle")
        if not pk:
            continue
        d = pickle.load(open(pk[0], "rb")); lt = d.get("label_type")
        rows = []
        for vid in d["predictions"]:
            p = cat_chunks(d["predictions"][vid]); l = cat_chunks(d["labels"][vid])
            if ax in SYN_AXES:
                hp, hl = hires_hr(p), hires_hr(l)
                hp2 = hl2 = None
            else:
                hp, hl = hr_for(p, lt), hr_for(l, lt)
                hp2, hl2 = hires_hr(p), hires_hr(l)
            rows.append(dict(vid=vidnum(vid), gt=hl, pred=hp, err=abs(hp - hl),
                             gt_h=hl2, pred_h=hp2,
                             err_h=(abs(hp2 - hl2) if hp2 is not None else None),
                             ratio=hp / hl))
        errs = np.array([r["err"] for r in rows])
        dbl = sum(1 for r in rows if abs(r["ratio"] - 2) < 0.2)
        hlv = sum(1 for r in rows if abs(r["ratio"] - 0.5) < 0.1)
        eh = [r["err_h"] for r in rows if r["err_h"] is not None]
        line = (f"  {ax:9s} n={len(errs):3d} MAE={errs.mean():6.2f} med={np.median(errs):5.2f} "
                f"배가={dbl} 반감={hlv}"
                + (f" | hi-res MAE={np.mean(eh):.2f}" if eh else ""))
        if ax == "PURE":
            ho = np.array([r["err"] for r in rows if r["vid"] >= 700])
            dh = sum(1 for r in rows if r["vid"] >= 700 and abs(r["ratio"] - 2) < 0.2)
            if len(ho):
                line += f" | held-out(07-10) MAE={ho.mean():.2f} 배가={dh}"
        print(line)
        out[ax] = rows
    if dump:
        json.dump(out, open(dump, "w"), indent=1)
    return out


if __name__ == "__main__":
    for m in sys.argv[1:]:
        print(f"--- {m} (정본: 실=toolbox 0.6-3.3 [읽기 = pickle label_type 기준], 합성=hi-res)")
        agg(m, dump=f"{PF}/../results/agg_{m}.json")
