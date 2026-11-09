#!/usr/bin/env python3
"""[61 문서 §7.4 가설 점검, 학습 없음] 회귀형 배경 조명 제거(패치 51b, RPPG_BGDC=2)가 고심박에서 맥박을 깎는가?
청크마다 BaseLoader 와 같은 규칙(배경 확률 ≥0.875·피부 ≥0.5, 31 프레임 이동평균 고역 통과, β∈[0,1.5])으로 β 를 구하고,
피부 평균 trace 에 대한 POS 맥박 SNR(정답 심박 ±6 bpm + 2배음 / 0.7–4 Hz 나머지, de Haan 방식)을 보정 전·후로 비교한다. 심박 대역별 집계.
  python scripts/bgrg_beta_diag.py DATASET(MMPD|VIPL) [--max N] → results/bgrg_beta_diag_{DATASET}.json"""
import json, os, sys
import numpy as np, cv2, pandas as pd
if not hasattr(np, "mat"): np.mat = np.asmatrix
sys.path.insert(0, "/SPECPHYS_ROOT/repos/rPPG-Toolbox")
from unsupervised_methods.methods.POS_WANG import POS_WANG
R = "/SPECPHYS_ROOT/_workspace/data/results"; L = "/SPECPHYS_ROOT/_workspace/data/physformer/cache/DataFileLists"; FS = 30
ds = sys.argv[1]; mx = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 10 ** 9
csv = {"MMPD": f"{L}/mmpd_zs_7_33.csv", "VIPL": f"{L}/vipl_all.csv"}[ds]
gtf = {"MMPD": f"{R}/agg_NF_REP_D_TA2_L_s600_MMPD.json", "VIPL": f"{R}/agg_NF_REP_D_TA2_L_s600_VIPL.json"}[ds]
GT = {r["vid"]: (r["gt_lit"] if ds == "VIPL" else r["gt"]) for r in json.load(open(gtf))}
def snr(sig, hr):
    f, P = np.fft.rfftfreq(len(sig) * 8, 1 / FS), np.abs(np.fft.rfft((sig - sig.mean()) * np.hanning(len(sig)), len(sig) * 8)) ** 2
    band = (f >= 0.7) & (f <= 4.0); h = hr / 60; tgt = (np.abs(f - h) <= 0.1) | (np.abs(f - 2 * h) <= 0.2)
    return 10 * np.log10(P[band & tgt].sum() / (P[band & ~tgt].sum() + 1e-12) + 1e-12)
def hpf(v, k=31): return v - np.stack([np.convolve(np.pad(v[:, c], (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid") for c in range(v.shape[1])], 1)
files = list(pd.read_csv(csv)["input_files"]); vids = {}
for f in files:
    b = os.path.basename(f); i = b.rindex("_input"); vids.setdefault(b[:i], []).append(f)
rng = np.random.default_rng(0); keys = sorted(v for v in vids if v in GT)
if ds == "VIPL": keys = [k for k in keys if GT[k] >= 110] + list(rng.choice([k for k in keys if GT[k] < 110], size=min(300, len(keys)), replace=False))
keys = keys[:mx]; rows = []
for n, v in enumerate(keys):
    for f in sorted(vids[v])[:4]:
        side = f.replace("_input", "_parse")
        if not os.path.exists(side): continue
        x = np.load(f)[:161].astype(np.float32); P = np.load(side).astype(np.float32) / 255.0; H, W = x.shape[1:3]
        bg = cv2.resize(P[1], (W, H)) >= 0.875; sk = cv2.resize(P[0], (W, H)) >= 0.5
        if bg.mean() < 0.03 or sk.mean() < 0.02: continue
        b = x[:, bg, :].mean(1); sm = x[:, sk, :].mean(1)
        rb = b / (b.mean(0, keepdims=True) + 1e-6) - 1.0; rs = sm / (sm.mean(0, keepdims=True) + 1e-6) - 1.0
        hb, hs = hpf(rb), hpf(rs); beta = np.clip((hb * hs).sum(0) / ((hb * hb).sum(0) + 1e-12), 0.0, 1.5)
        corr = np.array([np.corrcoef(hb[:, c], hs[:, c])[0, 1] for c in range(3)])
        smc = sm / np.clip(1.0 + beta[None, :] * rb, 0.2, 5.0)
        p0 = np.asarray(POS_WANG([sm[t].reshape(1, 1, 3) for t in range(len(sm))], FS)).squeeze(); p1 = np.asarray(POS_WANG([smc[t].reshape(1, 1, 3) for t in range(len(smc))], FS)).squeeze()
        rows.append(dict(vid=v, gt=float(GT[v]), beta_g=float(beta[1]), beta_mean=float(beta.mean()), corr_g=float(corr[1]), bg_rel_std=float(hb[:, 1].std()), skin_rel_std=float(hs[:, 1].std()), snr0=float(snr(p0, GT[v])), snr1=float(snr(p1, GT[v]))))
    if (n + 1) % 100 == 0: print(f"[diag] {n + 1}/{len(keys)}", flush=True)
json.dump(rows, open(f"{R}/bgrg_beta_diag_{ds}.json", "w"))
g = np.array([r["gt"] for r in rows]); print(f"[diag] {ds} chunks={len(rows)} videos={len(set(r['vid'] for r in rows))}")
from scipy.stats import wilcoxon
for nm, m in (("<70", g < 70), ("70-110", (g >= 70) & (g < 110)), ("≥110", g >= 110)):
    if m.sum() < 5: continue
    s0 = np.array([r["snr0"] for r, k in zip(rows, m) if k]); s1 = np.array([r["snr1"] for r, k in zip(rows, m) if k]); bt = np.array([r["beta_g"] for r, k in zip(rows, m) if k])
    cg = np.array([r["corr_g"] for r, k in zip(rows, m) if k]); rbs = np.array([r["bg_rel_std"] for r, k in zip(rows, m) if k]); rss = np.array([r["skin_rel_std"] for r, k in zip(rows, m) if k])
    print(f"  {nm:7s} n={m.sum():4d} β_G 평균 {bt.mean():.3f} (β>0 {np.mean(bt > 0):.2f}) · 피부-배경 상관 {np.median(cg):+.3f} · 배경/피부 변동 {np.median(rbs / (rss + 1e-9)):.2f} · SNR {s0.mean():+.2f} → {s1.mean():+.2f} dB (Δ {np.mean(s1 - s0):+.2f}, 악화 비율 {np.mean(s1 < s0):.2f}, p={wilcoxon(s1, s0).pvalue:.1e})")
