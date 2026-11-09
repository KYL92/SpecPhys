#!/usr/bin/env python3
"""[68 문서 §2, 파악] 학습 풀에서 정체성이 청크 심박을 얼마나 결정하는가: 청크 심박 라벨(u8 캐시 차분 라벨 → 누적합 → 선형 추세 제거 → Butterworth 0.6~3.3 Hz → 8배 영 채움 FFT 정점)을
정체성별로 묶어 η²(정체성 간 분산 / 전체 분산)와 정체성 안 심박 표준편차를 출처별로 낸다. 정체성: UBFC 1~41(잡음 얹은 309~341·코덱 사본 501~510 은 원본 UBFC 로), MCD 7xx/8xx 는 같은 사람의 안정/운동 직후로 보고 n%100,
PURE 9SSTT → SS, MMPD 700ABB → A, 합성 201~308·601~610 → 생성 정체성(sfhq_XX).
  python scripts/idhr_pool_coupling.py → results/idhr_pool_coupling.json"""
import csv, glob, json, os, re
import numpy as np
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"; R = "/SPECPHYS_ROOT/_workspace/data/results"
rows = list(csv.reader(open(f"{PF}/cache/DataFileLists/cotrain_allf_hyb4_u8_pool.csv")))[1:]
syn_id = {}
for c, base in (("train20h_tw", 200), ("train6hi_tw", 220), ("train6lo_tw", 230), ("train24dn", 0), ("train24dn2", 0)):
    for s in json.load(open(f"{PF}/configs/{c}.json"))["subjects"]: syn_id[base + int(s["id"])] = s["frames_dir"].split("/")[-1].replace("_frames", "")
for i, k in enumerate(range(1, 20, 2)): syn_id[601 + i] = syn_id[200 + k]
ubfc_of_cmp = {}
for n in range(501, 511):
    g = open(f"{PF}/CMP_REAL/subject{n}/ground_truth.txt").read()
    for u in range(1, 50):
        p = f"/SPECPHYS_ROOT/UBFC_dataset/DATASET_2/subject{u}/ground_truth.txt"
        if os.path.exists(p) and open(p).read() == g: ubfc_of_cmp[n] = u; break
def ident(n):
    if n <= 49: return "real", f"ubfc_{n}"
    if 201 <= n <= 308 or 601 <= n <= 610: return "synthetic", syn_id[n]
    if 309 <= n <= 349: return "real", f"ubfc_{n - 300}"
    if 501 <= n <= 510: return "real", f"ubfc_{ubfc_of_cmp.get(n, n)}"
    if 701 <= n <= 899: return "real", f"mcd_{n % 100}"
    if 90000 <= n < 100000: return "real", f"pure_{(n // 100) % 100}"
    if 700000 <= n < 800000: return "real", f"mmpd_{(n // 100) % 10}"
    return "other", str(n)
from scipy.signal import butter, filtfilt
_BA = butter(1, [0.6, 3.3], btype="bandpass", fs=30.0)
def hr(lab, fs=30.0):
    """정본 판독과 같은 청크 심박: 누적합 → 선형 추세 제거 → 1차 Butterworth 0.6~3.3 Hz → 8배 영 채움 FFT 정점(학습의 naive 라벨과 달리 MMPD 2배음 오선택을 피한다)."""
    x = np.cumsum(np.asarray(lab, np.float64)); t = np.arange(len(x)) - (len(x) - 1) / 2; x = x - x.mean(); x = x - (t @ x) / (t @ t) * t; x = filtfilt(*_BA, x)
    N = len(x) * 8; P = np.abs(np.fft.rfft(x, n=N)) ** 2; f = np.fft.rfftfreq(N, 1 / fs); m = (f >= 0.6) & (f <= 3.3); return float(f[m][np.argmax(P[m])] * 60)
data = []
for r in rows:
    p = r[1]; n = int(re.search(r"subject(\d+)_input", p).group(1)); src, idn = ident(n)
    data.append((src, idn, hr(np.load(p.replace("_input", "_label"))), n))
def eta(rs):
    v = np.array([x[2] for x in rs]); m = v.mean(); g = {}
    for x in rs: g.setdefault(x[1], []).append(x[2])
    ssb = sum(len(a) * (np.mean(a) - m) ** 2 for a in g.values()); return float(ssb / ((v - m) ** 2).sum()), float(np.mean([np.std(a) for a in g.values()])), len(g), len(rs)
out = {}
for name, sel in (("real", lambda x: x[0] == "real"), ("synthetic", lambda x: x[0] == "synthetic"), ("all", lambda x: True),
                  ("real_ubfc", lambda x: x[1].startswith("ubfc")), ("real_mcd", lambda x: x[1].startswith("mcd")), ("real_pure", lambda x: x[1].startswith("pure")), ("real_mmpd", lambda x: x[1].startswith("mmpd"))):
    rs = [x for x in data if sel(x)]; e, sd, ni, nc = eta(rs)
    hrs = np.array([x[2] for x in rs]); out[name] = dict(eta2=e, within_identity_sd=sd, identities=ni, chunks=nc, hr_p5=float(np.percentile(hrs, 5)), hr_p95=float(np.percentile(hrs, 95)))
    print(f"{name:12s} η²={e:.3f}  정체성 안 심박 SD={sd:5.1f} bpm  정체성 {ni:3d}  청크 {nc:4d}  심박 5~95 % {out[name]['hr_p5']:.0f}~{out[name]['hr_p95']:.0f}")
json.dump(out, open(f"{R}/idhr_pool_coupling.json", "w"), indent=1)
