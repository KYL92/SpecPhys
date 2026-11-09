#!/usr/bin/env python3
"""[61 문서 §13] 두 셀 출력의 스펙트럼 곱 융합(학습 없음): 영상마다 A·B 예측 파형을 정본 판독과 같은 전처리(detrend λ=100 → Butterworth 1차 0.6–3.3 Hz → periodogram, nfft=다음 2의 거듭제곱)로
정규화 주기도 P_A·P_B 를 만들고 기하 평균 sqrt(P_A·P_B) 의 최대 봉우리를 심박으로 읽는다(독립 우도의 곱). 정답·영상 속성은 A 의 정본 집계 행을 그대로 쓰고 pred·err 만 바꿔
agg_{ENS}_s{S}_{VIPL,MMPD}.json 을 쓴다(정본 정답 동일).
  python scripts/spec_fusion.py ENS A B SEED [SEED ...]   (A·B 는 시드 접미사 없는 셀 이름)"""
import glob, json, os, pickle, sys
import numpy as np, scipy.signal as ss
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from agg_model2 import cat_chunks, fast_detrend
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"; R = "/SPECPHYS_ROOT/_workspace/data/results"; FS = 30
ENS, A, B = sys.argv[1:4]; SEEDS = [int(s) for s in sys.argv[4:]]
def spec(x):
    x = fast_detrend(np.asarray(x, float), 100); b, a = ss.butter(1, [0.6 / FS * 2, 3.3 / FS * 2], btype="bandpass"); x = ss.filtfilt(b, a, np.double(x))
    N = 1 << (len(x) - 1).bit_length(); f, P = ss.periodogram(x, fs=FS, nfft=N, detrend=False); m = (f >= 0.6) & (f <= 3.3); return f[m], P[m] / (P[m].sum() + 1e-12)
def pk(m, ax):
    g = glob.glob(f"{PF}/runs/finetune_{m}/vipl_u8/saved_test_outputs/*.pickle") if ax == "MMPD" else glob.glob(f"{PF}/runs/ev_{m}_VIPL/vipl_u8/saved_test_outputs/*.pickle")
    return g[0] if g else None
for S in SEEDS:
    a, b = f"{A}_s{S}", f"{B}_s{S}"
    for ax in ("VIPL", "MMPD"):
        pa, pb = pk(a, ax), pk(b, ax); rows = json.load(open(f"{R}/agg_{a}_{ax}.json"))
        da, db = pickle.load(open(pa, "rb")), pickle.load(open(pb, "rb")); vids = {str(v): v for v in da["predictions"]}
        out = []; nf = 0
        for r in rows:
            v = vids.get(str(r["vid"]))
            if v is None: out.append(r); continue
            xa, xb = cat_chunks(da["predictions"][v]), cat_chunks(db["predictions"][v]); n = min(len(xa), len(xb))
            f, P1 = spec(xa[:n]); _, P2 = spec(xb[:n]); hr = float(60 * f[np.argmax(np.sqrt(P1 * P2))]); r = dict(r, pred=hr, pred_A=r.get("pred"))
            if "gt_lit" in r: r["err_lit"] = abs(hr - r["gt_lit"])
            r["err"] = abs(hr - r["gt"]); out.append(r); nf += 1
        json.dump(out, open(f"{R}/agg_{ENS}_s{S}_{ax}.json", "w")); print(f"[fusion] {ENS}_s{S} {ax}: {nf}/{len(rows)} videos fused")
print("FUSION_DONE")
