#!/usr/bin/env python3
"""파형 고조파 구조 분석 — "학습 파형 분포가 실제 PPG 파형 분포를 덮는가?"

실데이터(MCD-rPPG 100 Hz 원본 + 30 fps 동기본, UBFC-rPPG) 정답 PPG와
주입 파형(neurokit ppg_simulate, 해석적 a2 그리드 TRAIN20H)의
기본파 대비 2·3차 고조파 파워비(H2/H1, H3/H1)와 등가 a2/φ2/a3를 동일 지표로 계산하고,
TRAIN20H 영상의 피부 ROI 픽셀 스펙트럼에서 주입 무결성을 확인한다.

인자 없이 실행: python scripts/analyze_waveform_harmonics.py
출력: _workspace/data/results/waveform_harmonics.{json,png}
      _workspace/data/physformer/configs/train_h2_proposal.json (실행하지 않는 제안 config)

지표 정의
- 스펙트럼: Welch(20 s Hann, 50% 중첩, 제로패딩 0.01 Hz 분해능). 기본파 f0는 0.6–3.3 Hz 구간의
  고조파 합산(P(f)+P(2f)+P(3f)) 최대점 → 배가 오검출 방지. Hk = f0·k 주변 ±k·0.12 Hz 파워 적분.
- 시간영역 적합: 기본파 대역(0.7·f0~1.3·f0) Hilbert 위상 θ(t)로
  x ≈ Σ_k [c_k sin(kθ) + s_k cos(kθ)] 최소제곱 → a_k = |(c_k,s_k)|/|(c_1,s_1)|, φ_k = 상대 위상.
  (주입식 p = sin θ + a2 sin(2θ+φ2) + a3 sin(3θ+φ3) 과 동일한 파라미터화; 극성 반전에 a2·φ2 불변)
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PF = os.path.join(ROOT, "_workspace", "data", "physformer")
RES = os.path.join(ROOT, "_workspace", "data", "results")
MCD_RAW = "/SPECPHYS_RAW/mcd_rppg"
UBFC2 = os.path.join(ROOT, "UBFC_dataset", "DATASET_2")
sys.path.insert(0, os.path.join(ROOT, ".claude", "skills", "pixel-pulse-injection", "scripts"))

F_LO, F_HI = 0.6, 3.3      # 기본파 탐색 대역 (36–198 BPM)
BW = 0.12                  # 기본파 적분 반폭(Hz); k차는 k·BW


# ---------------------------------------------------------------- 지표
def _welch(x, fs):
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    x = signal.detrend(x)
    nper = int(min(len(x), 20 * fs))
    nfft = int(2 ** np.ceil(np.log2(fs / 0.01)))
    f, p = signal.welch(x, fs=fs, window="hann", nperseg=nper,
                        noverlap=nper // 2, nfft=nfft, detrend="linear")
    return f, p


def _bandpower(f, p, fc, hw):
    m = (f >= fc - hw) & (f <= fc + hw)
    return float(p[m].sum()) if m.any() else 0.0


def spectral_harmonics(x, fs):
    """f0(고조파 합산), H2/H1, H3/H1, 스펙트럼 argmax(단순 피크)가 f0인지."""
    f, p = _welch(x, fs)
    band = (f >= F_LO) & (f <= F_HI)
    fb = f[band]
    pi = lambda fq: np.interp(fq, f, p)
    hps = pi(fb) + pi(2 * fb) + pi(3 * fb)
    f0 = float(fb[np.argmax(hps)])
    h1 = _bandpower(f, p, f0, BW)
    h2 = _bandpower(f, p, 2 * f0, 2 * BW)
    h3 = _bandpower(f, p, 3 * f0, 3 * BW)
    # 단순 스펙트럼 최대점(모델이 보는 "지배 주파수")
    wide = (f >= F_LO) & (f <= 3 * F_HI)
    f_peak = float(f[wide][np.argmax(p[wide])])
    return dict(f0=f0, bpm=60 * f0, H2_H1=h2 / max(h1, 1e-12), H3_H1=h3 / max(h1, 1e-12),
                f_peak=f_peak, peak_is_f0=bool(abs(f_peak - f0) < BW),
                peak_ratio=f_peak / f0)


def fit_harmonics(x, fs, f0, nh=3):
    """Hilbert 위상 기반 최소제곱 고조파 적합 → a_k, φ_k (k=2,3)."""
    x = signal.detrend(np.asarray(x, float) - np.mean(x))
    lo, hi = 0.7 * f0, min(1.3 * f0, 0.49 * fs)
    sos = signal.butter(3, [lo, hi], btype="band", fs=fs, output="sos")
    fund = signal.sosfiltfilt(sos, x)
    theta = np.unwrap(np.angle(signal.hilbert(fund))) + np.pi / 2   # fund ≈ A·sin θ
    # 양끝 필터 과도 구간 제외
    m = int(2 * fs / f0)
    sl = slice(m, len(x) - m)
    cols = []
    for k in range(1, nh + 1):
        cols += [np.sin(k * theta[sl]), np.cos(k * theta[sl])]
    A = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(A, x[sl], rcond=None)
    out = {}
    c1, s1 = coef[0], coef[1]
    amp1 = np.hypot(c1, s1)
    ph1 = np.arctan2(s1, c1)          # sin(θ+ph1)
    for k in range(2, nh + 1):
        ck, sk = coef[2 * (k - 1)], coef[2 * (k - 1) + 1]
        ak = np.hypot(ck, sk) / max(amp1, 1e-12)
        phk = np.arctan2(sk, ck) - k * ph1   # 기본파를 sin θ로 정렬한 상대 위상
        out[f"a{k}"] = float(ak)
        out[f"phi{k}"] = float((phk + np.pi) % (2 * np.pi) - np.pi)
    resid = x[sl] - A @ coef
    out["fit_r2"] = float(1 - resid.var() / max(x[sl].var(), 1e-12))
    return out


def canonical_polarity(x, fs, f0):
    """PPG 형태 기준 극성 정규화: 수축기 상승(빠름) < 이완기 하강(느림)이 되도록.
    극성 반전은 φ2를 π만큼 이동시키므로(φ3는 불변) φ2 비교 전에 필수. 반환: (x_canon, flipped)."""
    sos = signal.butter(3, [0.4 * f0, min(4.5 * f0, 0.49 * fs)], btype="band", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, np.asarray(x, float) - np.mean(x))
    dist = int(0.5 * fs / f0)
    pk, _ = signal.find_peaks(y, distance=dist); tr, _ = signal.find_peaks(-y, distance=dist)
    if len(pk) < 3 or len(tr) < 3:
        return np.asarray(x, float), False
    rise, fall = [], []
    for t0 in tr:
        nxt = pk[pk > t0]
        if len(nxt):
            rise.append(nxt[0] - t0)
    for p0 in pk:
        nxt = tr[tr > p0]
        if len(nxt):
            fall.append(nxt[0] - p0)
    flipped = bool(np.median(rise) > np.median(fall))
    return (-np.asarray(x, float) if flipped else np.asarray(x, float)), flipped


def analyze(x, fs, min_sec=15):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < min_sec * fs or np.std(x) < 1e-9:
        return None
    sp = spectral_harmonics(x, fs)
    if sp["H2_H1"] > 50:   # 기본파 사실상 부재 → 잘못된 f0 (버림)
        return None
    x, flipped = canonical_polarity(x, fs, sp["f0"])
    sp["polarity_flipped"] = flipped
    ft = fit_harmonics(x, fs, sp["f0"])
    sp.update(ft)
    sp["a2_spec"] = float(np.sqrt(sp["H2_H1"]))
    sp["a3_spec"] = float(np.sqrt(sp["H3_H1"]))
    return sp


# ---------------------------------------------------------------- 로더
def load_gt_txt(path):
    with open(path) as f:
        return np.array([float(v) for v in f.readline().split()])


def mcd_pw(path):
    vals, ts = [], []
    with open(path, errors="ignore") as f:
        for line in f:
            s = line.split()
            if len(s) >= 3:
                try:
                    vals.append(float(s[0]))
                    ts.append(s[2])
                except ValueError:
                    pass
    v = np.array(vals)
    # 샘플링 주파수: 타임스탬프 간격 중앙값
    def sec(t):
        h, m, s_ = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s_)
    try:
        tt = np.array([sec(t) for t in ts[:2000]])
        fs = 1.0 / np.median(np.diff(tt))
    except Exception:
        fs = 100.0
    return v, float(fs)


def collect_real():
    corpora = {}
    # (1) MCD 원본 100 Hz 전체 코퍼스 (600명 × 2상태)
    for state in ("before", "after"):
        rows = []
        for p in sorted(glob.glob(f"{MCD_RAW}/ppg/*_{state}.PW")):
            v, fs = mcd_pw(p)
            if not (80 <= fs <= 120):
                fs = 100.0
            r = analyze(v[: int(90 * fs)], fs)
            if r:
                r["id"] = os.path.basename(p).split("_")[0]
                rows.append(r)
        corpora[f"MCD_raw100Hz_{state}"] = rows
    # (2) MCD 30 fps 동기본(평가에 쓴 20명)
    for tag, state in (("MCD_B", "before"), ("MCD_A", "after")):
        rows = []
        for d in sorted(glob.glob(f"{PF}/{tag}/subject*")):
            r = analyze(load_gt_txt(f"{d}/ground_truth.txt"), 30.0)
            if r:
                r["id"] = os.path.basename(d)
                rows.append(r)
        corpora[f"MCD_sync30_{state}"] = rows
    # (3) UBFC-rPPG DATASET_2 (42명, ≈30 fps; 3행 = 시간)
    rows = []
    for d in sorted(glob.glob(f"{UBFC2}/subject*")):
        gp = f"{d}/ground_truth.txt"
        if not os.path.exists(gp):
            continue
        lines = open(gp).read().splitlines()
        v = np.array([float(a) for a in lines[0].split()])
        fs = 30.0
        if len(lines) >= 3:
            t = np.array([float(a) for a in lines[2].split()])
            if len(t) > 10 and t[-1] > t[0]:
                fs = float((len(t) - 1) / (t[-1] - t[0]))
        r = analyze(v, fs)
        if r:
            r["id"] = os.path.basename(d)
            rows.append(r)
    corpora["UBFC_DATASET_2"] = rows
    return corpora


def collect_synthetic():
    corpora = {}
    # (a) neurokit 파형이 들어간 기존 학습셋 GT
    for tag, pat in (("TRAIN", "subject*"), ("TRAIN20", "subject*"), ("COTRAIN_synth", "subject2[0-9][0-9]")):
        src = "COTRAIN" if tag.startswith("COTRAIN") else tag
        rows = []
        for d in sorted(glob.glob(f"{PF}/{src}/{pat}")):
            r = analyze(load_gt_txt(f"{d}/ground_truth.txt"), 30.0)
            if r:
                r["id"] = os.path.basename(d)
                mp = f"{d}/meta.json"
                if os.path.exists(mp):
                    m = json.load(open(mp))
                    r["bpm_target"] = m.get("bpm")
                rows.append(r)
        corpora[tag] = rows
    # (b) neurokit 직접 생성 — BPM 스윕(파형 생성기의 고유 고조파 구조)
    try:
        from inject_pulse import make_waveform
        rows = []
        for bpm in range(50, 125, 5):
            for seed in (0, 1, 2):
                p, _ = make_waveform(1800, 30, bpm, use_neurokit=True, seed=seed)
                r = analyze(p, 30.0)
                if r:
                    r["id"] = f"nk_bpm{bpm}_s{seed}"
                    r["bpm_target"] = bpm
                    rows.append(r)
        corpora["neurokit_direct"] = rows
        # 지표 자체의 교정: 알려진 a2로 만든 해석 파형 → 추정 a2 회복 확인
        cal = []
        for a2 in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75):
            for phi in (0.0, 1.4, 2.8):
                p, _ = make_waveform(1008, 30, 72, seed=0, wave_a2=a2, wave_phi2=phi, wave_a3=0.2)
                r = analyze(p, 30.0)
                r.update(dict(true_a2=a2, true_phi2=phi, true_a3=0.2))
                cal.append(r)
        corpora["analytic_calibration"] = cal
    except Exception as e:  # noqa
        print("neurokit direct generation skipped:", e)
    # (c) TRAIN20H 해석적 a2 그리드 GT
    rows = []
    for d in sorted(glob.glob(f"{PF}/TRAIN20H/subject*")):
        m = json.load(open(f"{d}/meta.json"))
        r = analyze(load_gt_txt(f"{d}/ground_truth.txt"), 30.0)
        if r:
            r.update(id=os.path.basename(d), bpm_target=m["bpm"], true_a2=m["wave_a2"],
                     true_phi2=m["wave_phi2"], true_a3=m.get("wave_a3", 0.0))
            rows.append(r)
    corpora["TRAIN20H_gt"] = rows
    return corpora


def _pos(rgb, fs, win_sec=1.6):
    """POS (Wang et al. 2017): 무채색 조명 플리커를 상쇄하는 색도 투영."""
    n = len(rgb); L = int(win_sec * fs); h = np.zeros(n)
    for i in range(n - L + 1):
        c = rgb[i:i + L]; cn = c / (c.mean(0) + 1e-9)
        s1 = cn[:, 1] - cn[:, 2]; s2 = -2 * cn[:, 0] + cn[:, 1] + cn[:, 2]
        hh = s1 + (s1.std() / (s2.std() + 1e-9)) * s2
        h[i:i + L] += hh - hh.mean()
    return h


def _lockin(x, gt, fs, nh=3):
    """GT 위상 θ(t)(정답 파형의 기본파 Hilbert)를 알고 있을 때의 고조파 최소제곱 → 픽셀 신호의 a2/φ2/a3."""
    sp = spectral_harmonics(gt, fs); f0 = sp["f0"]
    sos = signal.butter(3, [0.7 * f0, min(1.3 * f0, 0.49 * fs)], btype="band", fs=fs, output="sos")
    theta = np.unwrap(np.angle(signal.hilbert(signal.sosfiltfilt(sos, gt - gt.mean())))) + np.pi / 2
    m = int(2 * fs / f0); sl = slice(m, len(x) - m)
    A = np.stack(sum([[np.sin(k * theta[sl]), np.cos(k * theta[sl])] for k in range(1, nh + 1)], []), 1)
    coef, *_ = np.linalg.lstsq(A, x[sl], rcond=None)
    amp1 = np.hypot(coef[0], coef[1]); ph1 = np.arctan2(coef[1], coef[0])
    out = {}
    for k in range(2, nh + 1):
        ck, sk = coef[2 * (k - 1)], coef[2 * (k - 1) + 1]
        out[f"a{k}"] = float(np.hypot(ck, sk) / max(amp1, 1e-12))
        out[f"phi{k}"] = float(((np.arctan2(sk, ck) - k * ph1) + np.pi) % (2 * np.pi) - np.pi)
    resid = x[sl] - A @ coef
    out["explained_var"] = float(1 - resid.var() / max(x[sl].var(), 1e-12))
    out["amp1_rel"] = float(amp1)   # 기본파 절대 변조 진폭(정규화 픽셀 단위)
    return out


def pixel_integrity():
    """TRAIN20H 영상: 주입 마스크 내 RGB 평균 → (i) 녹색 원신호, (ii) POS 투영 스펙트럼,
    (iii) GT 위상 lock-in 회귀로 픽셀 레벨 a2/φ2/a3 복원. 노이즈(플리커·움직임·드리프트)는
    의도된 교란이므로 (i)는 지배 주파수가 노이즈일 수 있고, (ii)(iii)가 주입 무결성의 판정 기준."""
    import cv2
    rows = []
    sos = signal.butter(3, [0.5, 5.0], btype="band", fs=30, output="sos")
    for d in sorted(glob.glob(f"{PF}/TRAIN20H/subject*"),
                    key=lambda s: int(os.path.basename(s)[7:])):
        m = json.load(open(f"{d}/meta.json"))
        amap = cv2.imread(f"{d}/amplitude_map.png", cv2.IMREAD_GRAYSCALE)
        mask = amap > 127
        cap = cv2.VideoCapture(f"{d}/vid.avi"); rgb = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            rgb.append(fr[..., ::-1][mask].mean(0))   # BGR→RGB
        cap.release()
        rgb = np.array(rgb, float)
        gt = load_gt_txt(f"{d}/ground_truth.txt")
        n = min(len(rgb), len(gt)); rgb, gt = rgb[:n], gt[:n]
        g = signal.sosfiltfilt(sos, rgb[:, 1] / np.median(rgb[:, 1]) - 1)
        pos = signal.sosfiltfilt(sos, _pos(rgb, 30.0))
        rg_raw = spectral_harmonics(g, 30.0)
        rp = analyze(pos, 30.0) or {}
        lk_g = _lockin(g, gt, 30.0); lk_p = _lockin(pos, gt, 30.0)
        gb = signal.sosfiltfilt(sos, gt - gt.mean())
        row = dict(id=os.path.basename(d), bpm_target=m["bpm"], alpha=m["alpha"],
                   true_a2=m["wave_a2"], true_phi2=m["wave_phi2"], true_a3=m.get("wave_a3", 0.0),
                   spatial=m["spatial"], noise=m.get("noise", {}), mask_frac=float(mask.mean()), n_frames=int(n),
                   green_peak_hz=rg_raw["f_peak"], green_peak_is_f0=bool(abs(rg_raw["f_peak"] - m["bpm"] / 60) < BW),
                   green_H2_H1=rg_raw["H2_H1"], green_corr_gt=float(np.corrcoef(g, gb)[0, 1]),
                   pos_bpm=rp.get("bpm"), pos_peak_is_f0=rp.get("peak_is_f0"), pos_peak_ratio=rp.get("peak_ratio"),
                   pos_H2_H1=rp.get("H2_H1"), pos_a2_fit=rp.get("a2"), pos_corr_gt=float(np.corrcoef(pos, gb)[0, 1]),
                   lockin_green=lk_g, lockin_pos=lk_p)
        rows.append(row)
        print(f"  {row['id']:>10} a2={m['wave_a2']:.2f} | green: peak@{rg_raw['f_peak']:.2f}Hz(f0={m['bpm']/60:.2f}) "
              f"lock-in a2={lk_g['a2']:.3f} | POS: peak=f0 {rp.get('peak_is_f0')} H2/H1={rp.get('H2_H1', float('nan')):.3f} "
              f"lock-in a2={lk_p['a2']:.3f} phi2={lk_p['phi2']:+.2f}(true {m['wave_phi2']:+.2f}) corr={row['pos_corr_gt']:.2f}")
    return rows


# ---------------------------------------------------------------- 요약/그림
def q(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    pc = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    return dict(n=int(len(v)), min=pc[0], p5=pc[1], q1=pc[2], median=pc[3], q3=pc[4], p95=pc[5],
                max=pc[6], mean=float(v.mean()))


def summarize(rows, keys=("H2_H1", "H3_H1", "a2_spec", "a2", "phi2", "a3", "bpm")):
    return {k: q([r.get(k) for r in rows]) for k in keys}


def circ_stats(phis):
    v = np.asarray([p for p in phis if p is not None], float)
    if len(v) == 0:
        return None
    z = np.exp(1j * v).mean()
    return dict(n=int(len(v)), circ_mean=float(np.angle(z)), R=float(abs(z)),
                circ_std=float(np.sqrt(-2 * np.log(max(abs(z), 1e-9)))))


def coverage(real_rows, train_rows, key="a2"):
    rv = np.array([r[key] for r in real_rows])
    tv = np.array([r[key] for r in train_rows])
    lo, hi = tv.min(), tv.max()
    return dict(train_min=float(lo), train_max=float(hi),
                real_frac_inside=float(((rv >= lo) & (rv <= hi)).mean()),
                real_frac_below=float((rv < lo).mean()), real_frac_above=float((rv > hi).mean()),
                real_median_percentile_in_train=float((tv <= np.median(rv)).mean() * 100))


def make_figure(real, synth, pix, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [
        ("MCD rest\n100Hz n=%d", real["MCD_raw100Hz_before"]),
        ("MCD post-ex\n100Hz n=%d", real["MCD_raw100Hz_after"]),
        ("MCD 30fps\nrest n=%d", real["MCD_sync30_before"]),
        ("MCD 30fps\npost-ex n=%d", real["MCD_sync30_after"]),
        ("UBFC\nn=%d", real["UBFC_DATASET_2"]),
        ("neurokit\nTRAIN20 n=%d", synth["TRAIN20"]),
        ("neurokit\ndirect n=%d", synth.get("neurokit_direct", [])),
        ("analytic\nTRAIN20H n=%d", synth["TRAIN20H_gt"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    colors = ["#4C72B0"] * 5 + ["#DD8452"] * 2 + ["#55A868"]
    for ax, key, lab in ((axes[0], "H2_H1", "H2/H1 power ratio"), (axes[1], "a2", "equivalent a2 (fit)")):
        data = [[r[key] for r in g] for _, g in groups]
        bp = ax.boxplot(data, showfliers=True, flierprops=dict(marker=".", ms=3, alpha=.4), patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(.55)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([n % len(g) for n, g in groups], fontsize=7)
        if key == "H2_H1":
            ax.set_yscale("log"); ax.set_ylim(5e-3, 30)
        ax.set_ylabel(lab); ax.grid(alpha=.3, axis="y")
        if key == "a2":
            for a2 in (0, .15, .3, .45, .6, .75):
                ax.axhline(a2, color="#55A868", lw=.6, ls=":")
    axes[0].set_title("2nd-harmonic power: real vs injected waveforms")
    axes[1].set_title("Equivalent a2 (dotted = TRAIN20H grid)")
    # φ2 vs a2 산점
    ax = axes[2]
    for name, g, c, mk in (("MCD rest", real["MCD_raw100Hz_before"], "#4C72B0", "."),
                           ("MCD post-ex", real["MCD_raw100Hz_after"], "#8172B3", "."),
                           ("UBFC", real["UBFC_DATASET_2"], "#64B5CD", "^"),
                           ("neurokit", synth["TRAIN20"] + synth.get("neurokit_direct", []), "#DD8452", "s"),
                           ("TRAIN20H", synth["TRAIN20H_gt"], "#55A868", "D")):
        ax.scatter([r["phi2"] for r in g], [r["a2"] for r in g], s=10 if mk == "." else 28,
                   c=c, marker=mk, alpha=.5, label=name)
    ax.set_xlabel("phi2 (rad, fundamental aligned to sin)"); ax.set_ylabel("a2 (fit)")
    ax.set_xlim(-np.pi, np.pi); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("2nd-harmonic amplitude vs phase")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("figure ->", path)


def make_proposal(real_rows, path):
    """실데이터 분포(MCD rest+post-ex, UBFC 합산)에 맞춘 다음 학습셋 config 초안."""
    a2 = np.array([r["a2"] for r in real_rows]); phi2 = np.array([r["phi2"] for r in real_rows])
    a3 = np.array([r["a3"] for r in real_rows]); phi3 = np.array([r["phi3"] for r in real_rows])
    pa2 = np.percentile(a2, [5, 25, 50, 75, 95]); pa3 = np.percentile(a3, [5, 50, 95])
    cs2 = circ_stats(phi2); cs3 = circ_stats(phi3)
    base = json.load(open(f"{PF}/configs/train20h.json"))
    rng = np.random.default_rng(2026)
    subjects = []
    # 실데이터 5–95 백분위 + 상단 여유(배가 방어용 tail)를 균등 분위 샘플링
    a2_grid = np.round(np.percentile(a2, np.linspace(5, 95, 16)), 3).tolist() + [0.6, 0.7, 0.8, 0.85]
    for i, s in enumerate(base["subjects"]):
        s = dict(s)
        s["wave_a2"] = float(a2_grid[i % len(a2_grid)])
        s["wave_phi2"] = float(np.round(rng.normal(cs2["circ_mean"], cs2["circ_std"]), 3))
        s["wave_a3"] = float(np.round(np.clip(rng.normal(np.median(a3), (pa3[2] - pa3[0]) / 3.3), 0, 0.5), 3))
        s["wave_phi3"] = float(np.round(rng.normal(cs3["circ_mean"], cs3["circ_std"]), 3))
        subjects.append(s)
    out = dict(base)
    out["out"] = "_workspace/data/physformer/TRAIN20H2"
    out["_proposal_note"] = {
        "source": "scripts/analyze_waveform_harmonics.py — real PPG (MCD 100 Hz both states + UBFC) fitted a2/phi2/a3/phi3",
        "a2_real_percentiles_5_25_50_75_95": [float(x) for x in pa2],
        "a2_sampling": "16 clips: uniform quantiles of real a2 (p5..p95); 4 clips: upper tail {0.6, 0.7, 0.8, 0.85} (0.85 = make_waveform cap; real a2>0.85 in ~5%)",
        "phi2_sampling": f"N(circ_mean={cs2['circ_mean']:.3f}, circ_std={cs2['circ_std']:.3f}) rad",
        "a3_sampling": f"N(median={np.median(a3):.3f}, sd={(pa3[2]-pa3[0])/3.3:.3f}) clipped [0,0.5]",
        "phi3_sampling": f"N(circ_mean={cs3['circ_mean']:.3f}, circ_std={cs3['circ_std']:.3f}) rad",
        "not_executed": True,
    }
    out["subjects"] = subjects
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print("proposal config ->", path)
    return out["_proposal_note"]


def main():
    os.makedirs(RES, exist_ok=True)
    print("[1] real PPG corpora ...")
    real = collect_real()
    for k, v in real.items():
        print(f"  {k}: n={len(v)}")
    print("[2] injected waveforms ...")
    synth = collect_synthetic()
    for k, v in synth.items():
        print(f"  {k}: n={len(v)}")
    print("[4] TRAIN20H pixel-level integrity ...")
    pix = pixel_integrity()

    summary = {k: summarize(v) for k, v in {**real, **synth}.items()}
    for k, v in {**real, **synth}.items():
        summary[k]["phi2_circ"] = circ_stats([r["phi2"] for r in v])
        summary[k]["phi3_circ"] = circ_stats([r["phi3"] for r in v])
        summary[k]["peak_is_f0_frac"] = float(np.mean([r["peak_is_f0"] for r in v])) if v else None
        summary[k]["polarity_flipped_frac"] = float(np.mean([r["polarity_flipped"] for r in v])) if v else None

    real_all = (real["MCD_raw100Hz_before"] + real["MCD_raw100Hz_after"] + real["UBFC_DATASET_2"])
    nk_all = synth["TRAIN"] + synth["TRAIN20"] + synth["COTRAIN_synth"] + synth.get("neurokit_direct", [])
    cov = {
        "neurokit_vs_real": {
            "a2_fit": coverage(real_all, nk_all, "a2"), "H2_H1": coverage(real_all, nk_all, "H2_H1"),
            "nk_median_a2_percentile_in_real": float((np.array([r["a2"] for r in real_all])
                                                       <= np.median([r["a2"] for r in nk_all])).mean() * 100),
            "nk_median_H2H1_percentile_in_real": float((np.array([r["H2_H1"] for r in real_all])
                                                         <= np.median([r["H2_H1"] for r in nk_all])).mean() * 100),
        },
        "TRAIN20H_vs_real": {"a2_fit": coverage(real_all, synth["TRAIN20H_gt"], "a2"),
                             "H2_H1": coverage(real_all, synth["TRAIN20H_gt"], "H2_H1"),
                             "phi2_real_circ": circ_stats([r["phi2"] for r in real_all]),
                             "phi2_train20h_circ": circ_stats([r["phi2"] for r in synth["TRAIN20H_gt"]])},
        "per_corpus_a2_inside_neurokit_range": {
            k: coverage(v, nk_all, "a2")["real_frac_inside"] for k, v in real.items()},
        "per_corpus_a2_inside_TRAIN20H_range": {
            k: coverage(v, synth["TRAIN20H_gt"], "a2")["real_frac_inside"] for k, v in real.items()},
    }
    proposal = make_proposal(real_all, f"{PF}/configs/train_h2_proposal.json")
    out = dict(metric_definition=__doc__, summary=summary, coverage=cov,
               pixel_integrity=pix, proposal=proposal,
               per_video={k: v for k, v in {**real, **synth}.items()})
    jp = f"{RES}/waveform_harmonics.json"
    json.dump(out, open(jp, "w"), indent=1, default=float)
    print("json ->", jp)
    make_figure(real, synth, pix, f"{RES}/waveform_harmonics.png")

    print("\n=== summary (median [q1–q3]) ===")
    for k, s in summary.items():
        h, a, p = s["H2_H1"], s["a2"], s["phi2_circ"]
        print(f"{k:24s} n={h['n']:4d} H2/H1={h['median']:.3f} [{h['q1']:.3f}–{h['q3']:.3f}] "
              f"a2_fit={a['median']:.3f} [{a['q1']:.3f}–{a['q3']:.3f}] phi2={p['circ_mean']:+.2f}±{p['circ_std']:.2f} "
              f"peak=f0 {s['peak_is_f0_frac']:.2f}")
    print(json.dumps(cov, indent=1))
    print("\n=== metric calibration (analytic, known a2/phi2/a3=0.2) ===")
    for r in synth.get("analytic_calibration", []):
        print(f"true a2={r['true_a2']:.2f} phi2={r['true_phi2']:+.2f} | fit a2={r['a2']:.3f} phi2={r['phi2']:+.2f} "
              f"a3={r['a3']:.3f} spec a2={r['a2_spec']:.3f} flipped={r['polarity_flipped']}")
    print("\n=== polarity flipped fraction ===")
    for k, s_ in summary.items():
        print(f"{k:24s} {s_['polarity_flipped_frac']:.2f}")


if __name__ == "__main__":
    main()
