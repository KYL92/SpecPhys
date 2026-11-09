#!/usr/bin/env python3
"""2단계 결과 통합: 모델(base/T1/T2/T2L×3시드) × 조건(C1-C6, E1-E3) HR MAE 마스터 테이블,
T1-vs-T2 귀인 분리, 외삽 일반화, 시드 평균±표준편차. 결과 JSON+그림 저장."""
import glob, json, os, pickle
import numpy as np, torch
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
RES = "/SPECPHYS_ROOT/_workspace/data/results"
fs = 30
CONDS = ["C1", "C2", "C3", "C4", "C5", "C6", "E1", "E2", "E3"]
MODELS = ["base", "T1", "T2", "T2L_s100", "T2L_s200", "T2L_s300"]


def video_hr(chunks):
    sig = np.concatenate([chunks[k].detach().cpu().float().numpy().ravel() for k in sorted(chunks)])
    sig = np.cumsum(sig - sig.mean())
    b, a = butter(3, [0.7 / (fs / 2), 4.0 / (fs / 2)], btype="band")
    f = filtfilt(b, a, sig)
    fr = np.fft.rfftfreq(len(f) * 8, 1 / fs)
    pw = np.abs(np.fft.rfft(f - f.mean(), n=len(f) * 8)) ** 2
    band = (fr >= 0.7) & (fr <= 4.0)
    return float(fr[band][pw[band].argmax()] * 60)


def eval_run(model, cond):
    # 1단계 실행 경로(base/T2 × C*)와 2단계 경로(ev_*) 모두 탐색
    pats = [f"{PF}/runs/ev_{model}_{cond}/synth_{cond}/saved_test_outputs/*.pickle",
            f"{PF}/runs/{'base' if model=='base' else 'ft'}_{cond}/synth_{cond}/saved_test_outputs/*.pickle"
            if model in ("base", "T2") else "NONE"]
    pk = []
    for p in pats:
        pk = glob.glob(p)
        if pk:
            break
    if not pk:
        return None
    d = pickle.load(open(pk[0], "rb"))
    errs = []
    for subj, chunks in d["predictions"].items():
        gt = json.load(open(f"{PF}/TEST_{cond}/{subj}/meta.json"))["achieved_bpm"]
        errs.append(abs(video_hr(chunks) - gt))
    return float(np.mean(errs))


table = {m: {} for m in MODELS}
for m in MODELS:
    for c in CONDS:
        v = eval_run(m, c)
        if v is not None:
            table[m][c] = round(v, 2)

# T2L 시드 통계
t2l = {}
for c in CONDS:
    vals = [table[f"T2L_s{s}"].get(c) for s in (100, 200, 300)]
    vals = [v for v in vals if v is not None]
    if vals:
        t2l[c] = {"mean": round(float(np.mean(vals)), 2), "std": round(float(np.std(vals)), 2), "n": len(vals)}

out = {"per_model": table, "T2L_stats": t2l}
json.dump(out, open(f"{RES}/phase2_master.json", "w"), indent=1)

# 출력 표
hdr = f"{'cond':4s} " + " ".join(f"{m:>9s}" for m in ["base", "T1", "T2"]) + f" {'T2L(m±s)':>12s}"
print(hdr)
for c in CONDS:
    row = f"{c:4s} "
    for m in ["base", "T1", "T2"]:
        v = table[m].get(c)
        row += f"{v:9.2f} " if v is not None else f"{'--':>9s} "
    if c in t2l:
        row += f"{t2l[c]['mean']:6.2f}±{t2l[c]['std']:4.2f}"
    print(row)

# 그림: 조건×모델 MAE
labels = CONDS
x = np.arange(len(labels))
w = 0.2
fig, ax = plt.subplots(figsize=(12, 4.5))
for i, (m, lab, col) in enumerate([("base", "Pretrained", "#d62728"),
                                   ("T1", "T1: clean-only FT", "#ff9f40"),
                                   ("T2", "T2: noise-aug FT", "#2ca02c")]):
    vals = [table[m].get(c, np.nan) for c in labels]
    ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=col)
t2l_m = [t2l.get(c, {}).get("mean", np.nan) for c in labels]
t2l_s = [t2l.get(c, {}).get("std", 0) for c in labels]
ax.bar(x + 1.5 * w, t2l_m, w, yerr=t2l_s, capsize=3, label="T2L: expanded FT (3 seeds)", color="#1f77b4")
ax.set_xticks(x)
ax.set_xticklabels(["C1\nclean", "C2\npatch", "C3\nillum", "C4\nmotion", "C5\nil+mo", "C6\nall",
                    "E1\nillum↑↑", "E2\nmotion↑↑", "E3\nnovel"], fontsize=8)
ax.set_ylabel("HR MAE (BPM)")
ax.set_title("Attribution (T1 vs T2) and extrapolation (E1-E3) — 4 held-out identities")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RES}/phase2_attribution_extrapolation.png", dpi=110)
print("saved phase2_master.json + figure")
