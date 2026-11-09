#!/usr/bin/env python3
"""[61 문서 P1, 학습 없음] 같은 u8 얼굴 크롭에서 POS 로 신호 품질을 비교: (a) 크롭 전체 (b) YCrCb 피부 (c) parser 피부(p_skin≥0.5) (d) (c) ÷ 안정 배경 조명(채널별, 시간 평균 1) (e) (d) 에서 잡음 참조(배경 RGB·얼굴 내 비피부·LK 이동)를 1 s 고역 통과 후 최소제곱으로 회귀 제거.
심박 = 정본 readout(detrend λ=100 → butter1 0.6–3.3 Hz → periodogram). 정답 = 같은 영상의 정본 집계 gt.
  python scripts/pos_parse_check.py DATASET(MMPD|VIPL) [--max N] → results/pos_parse_check_{DATASET}.json"""
import glob, json, os, re, sys
import numpy as np, cv2, pandas as pd
if not hasattr(np, "mat"): np.mat = np.asmatrix
sys.path.insert(0, "/SPECPHYS_ROOT/repos/rPPG-Toolbox"); sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from unsupervised_methods.methods.POS_WANG import POS_WANG
from evaluation.post_process import _calculate_fft_hr
from agg_model2 import fast_detrend
import scipy.signal as ss
R = "/SPECPHYS_ROOT/_workspace/data/results"; L = "/SPECPHYS_ROOT/_workspace/data/physformer/cache/DataFileLists"; FS = 30
def tb_hr(x):
    x = fast_detrend(np.asarray(x, float) - np.mean(x), 100); b, a = ss.butter(1, [0.6 / FS * 2, 3.3 / FS * 2], btype="bandpass")
    return float(_calculate_fft_hr(ss.filtfilt(b, a, x), fs=FS, low_pass=0.6, high_pass=3.3))
def hp(x, k=31):
    k = min(k, len(x) // 2 * 2 - 1); return x - np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
def lk(g):   # g (T,H,W) luminance → global translation velocity (vx, vy) (T,)
    Ix = 0.5 * (g[:, 1:-1, 2:] - g[:, 1:-1, :-2]); Iy = 0.5 * (g[:, 2:, 1:-1] - g[:, :-2, 1:-1]); It = np.diff(g, axis=0)[:, 1:-1, 1:-1]
    Ix, Iy = Ix[:-1], Iy[:-1]; a = (Ix * Ix).sum((1, 2)); b = (Ix * Iy).sum((1, 2)); c = (Iy * Iy).sum((1, 2)); sx = (Ix * It).sum((1, 2)); sy = (Iy * It).sum((1, 2))
    det = a * c - b * b + 1e-6; vx = -(c * sx - b * sy) / det; vy = -(a * sy - b * sx) / det
    return np.r_[vx, vx[-1:]], np.r_[vy, vy[-1:]]
ds = sys.argv[1]; mx = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 10 ** 9
csv = {"MMPD": f"{L}/mmpd_zs_7_33.csv", "VIPL": f"{L}/vipl_all.csv"}[ds]
gtf = {"MMPD": f"{R}/agg_NF_REP_D_TA2_L_s600_MMPD.json", "VIPL": f"{R}/agg_NF_REP_D_TA2_L_s600_VIPL.json"}[ds]
GT = {r["vid"]: r for r in json.load(open(gtf))}
files = list(pd.read_csv(csv)["input_files"]); vids = {}
for f in files:
    b = os.path.basename(f); i = b.rindex("_input"); vids.setdefault(b[:i], []).append((int(b[i + 6:].split(".")[0]), f))
out = []; keys = sorted(vids)[:mx]
for n, v in enumerate(keys):
    if v not in GT: continue
    ser = {k: [] for k in "abcde"}; refs = []
    for cid, f in sorted(vids[v]):
        x = np.load(f)[:160].astype(np.float32); side = f.replace("_input", "_parse")
        if not os.path.exists(side): ser = None; break
        P = np.load(side).astype(np.float32) / 255.0; H, W = x.shape[1:3]
        sk = cv2.resize(P[0], (W, H)) >= 0.5; bg = cv2.resize(P[1], (W, H)) >= 0.875; ns = cv2.resize(P[2], (W, H)) >= 0.5
        ycc = cv2.cvtColor(x.mean(0).astype(np.uint8), cv2.COLOR_RGB2YCrCb); yk = (ycc[..., 1] >= 133) & (ycc[..., 1] <= 173) & (ycc[..., 2] >= 77) & (ycc[..., 2] <= 127)
        if yk.sum() < 200: yk = np.ones((H, W), bool)
        if sk.sum() < 50: sk = yk
        ser["a"].append(x.mean((1, 2))); ser["b"].append(x[:, yk].mean(1)); ser["c"].append(x[:, sk].mean(1))
        bgt = x[:, bg].mean(1) if bg.mean() >= 0.03 else np.ones((len(x), 3), np.float32) * x.mean((0, 1, 2))
        ser["d"].append(x[:, sk].mean(1) / (bgt / bgt.mean(0, keepdims=True)))
        nst = x[:, ns].mean((1, 2)) if ns.sum() >= 20 else np.zeros(len(x), np.float32)
        vx, vy = lk(x.mean(-1)); refs.append(np.c_[bgt, nst, np.cumsum(vx), np.cumsum(vy)])
    if ser is None: continue
    S = {k: np.concatenate(s, 0) for k, s in ser.items() if k != "e"}; Rf = np.concatenate(refs, 0)
    X = np.stack([hp(Rf[:, j]) for j in range(Rf.shape[1])], 1); X = X / (X.std(0, keepdims=True) + 1e-6)
    e = S["d"].copy()
    for c in range(3):
        y = hp(e[:, c]); beta, *_ = np.linalg.lstsq(X, y, rcond=None); e[:, c] = e[:, c] - X @ beta
    S["e"] = e
    g = GT[v]["gt_lit"] if ds == "VIPL" else GT[v]["gt"]; row = dict(vid=v, gt=g, gt_wave=GT[v]["gt"])
    for k, s in S.items():
        bvp = np.asarray(POS_WANG([s[t].reshape(1, 1, 3) for t in range(len(s))], FS)).squeeze(); row[k] = tb_hr(bvp)
    if ds == "MMPD": row.update({kk: GT[v][kk] for kk in ("subject", "light", "motion", "exercise", "skin")})
    out.append(row)
    if (n + 1) % 100 == 0: print(f"[pos] {n + 1}/{len(keys)}", flush=True)
json.dump(out, open(f"{R}/pos_parse_check_{ds}.json", "w"))
g = np.array([r["gt"] for r in out]); names = {"a": "크롭 전체", "b": "YCrCb 피부", "c": "parser 피부", "d": "parser 피부 ÷ 배경 조명", "e": "+ 잡음 참조 회귀 제거"}
bands = [("전체", g > 0), ("<70", g < 70), ("70-110", (g >= 70) & (g < 110)), ("≥110", g >= 110)]
print(f"[pos] {ds} n={len(out)}")
for k, nm in names.items():
    p = np.array([r[k] for r in out]); print(f"  {nm:22s} " + " | ".join(f"{b} MAE {np.abs(p - g)[m].mean():6.2f} (5bpm {np.mean(np.abs(p - g)[m] <= 5):.2f}, n={m.sum()})" for b, m in bands))
