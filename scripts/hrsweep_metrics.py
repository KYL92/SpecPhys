#!/usr/bin/env python3
"""[68 문서 §4] HRSWEEP40(풀 밖 정체성 4 × 심박 10, 장면 고정) 심박 추종 지표. pro6000 컨테이너에서 실행(클립 meta·예측 pickle).
클립 심박 = 청크 출력을 이어 붙인 뒤 정본 합성 판독(8배 영 채움 FFT 0.7~4.0 Hz 정점). 정답 = meta achieved_bpm.
출력: results/hrsweep_{MODEL}.json (클립별 정답·예측). 비교는 scripts/hrsweep_cmp.py.
[69 문서] 차분 정규화 라벨 모델(pickle label_type, 공개 PhysMamba·PhysFormer)은 정본 판독처럼 청크마다 누적합 + 선형 추세 제거를 먼저 한다(표준화 라벨 모델의 출력은 이전과 같다).
HRSWEEP_BP=1: 이어 붙인 뒤 3차 Butterworth 0.7~4.0 Hz 대역 통과(민감도 판독), 출력 이름 끝에 _bp.
  python scripts/hrsweep_metrics.py MODEL [MODEL ...]"""
import glob, json, pickle, sys
import numpy as np
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; RES = "/SPECPHYS_ROOT/_workspace/data/results"
import os; LO = os.environ.get("HRSWEEP_SET", "") == "LO"   # [68 §4] 어려운 판(맥파 0.003, 잡음 강함): 폴더 HRSWEEP40LO, 평가 ev_{M}_HRSWEEPLO, 출력 hrsweeplo_{M}.json
CFG = {s["id"]: s for s in json.load(open(f"{PFA}/configs/{'hrsweep40lo' if LO else 'hrsweep40'}.json"))["subjects"]}
BP = os.environ.get("HRSWEEP_BP", "") == "1"
from scipy.signal import butter, filtfilt
_BA3 = butter(3, [0.7, 4.0], btype="band", fs=30.0)
def prep(v, diff):
    v = np.asarray(v, np.float64)
    if diff:   # [69] 차분 정규화 출력 → 누적합 + 선형 추세 제거(정본 판독의 cumsum 단계)
        v = np.cumsum(v - v.mean()); t = np.arange(len(v)) - (len(v) - 1) / 2; v = v - v.mean(); v = v - (t @ v) / (t @ t) * t
    return (v - v.mean()) / (v.std() + 1e-8)
def clip_hr(x, fs=30.0):
    x = np.asarray(x, np.float64); x = x - x.mean(); N = len(x) * 8; P = np.abs(np.fft.rfft(x, n=N)) ** 2; f = np.fft.rfftfreq(N, 1 / fs); m = (f >= 0.7) & (f <= 4.0); return float(f[m][np.argmax(P[m])] * 60)
for M in sys.argv[1:]:
    pk = glob.glob(f"{PFA}/runs/ev_{M}_HRSWEEP{'LO' if LO else ''}/*/saved_test_outputs/*.pickle")
    if not pk: print("no pickle", M); continue
    d = pickle.load(open(pk[0], "rb")); out = {}; DIFF = d.get("label_type") == "DiffNormalized"
    for sub, ch in d["predictions"].items():
        sid = int(sub.replace("subject", "")); c = CFG[sid]; m = json.load(open(f"{PFA}/HRSWEEP40{'LO' if LO else ''}/subject{sid}/meta.json"))
        get = lambda t: t.cpu().numpy().ravel() if hasattr(t, "cpu") else np.asarray(t).ravel()
        y = np.concatenate([prep(get(ch[k]), DIFF) for k in sorted(ch)])
        if BP: y = filtfilt(*_BA3, y)
        out[sub] = dict(face=c["hrsweep_face"], target=c["bpm"], gt=float(m["achieved_bpm"]), pred=clip_hr(y))
    json.dump(dict(model=M, label_type=d.get("label_type"), bandpass=BP, clips=out), open(f"{RES}/hrsweep{'lo' if LO else ''}_{M}{'_bp' if BP else ''}.json", "w"), indent=1); print("hrsweep", "LO" if LO else "", M, len(out), d.get("label_type"), "bp" if BP else "")
