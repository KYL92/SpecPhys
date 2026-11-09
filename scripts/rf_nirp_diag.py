#!/usr/bin/env python3
"""[59 문서 §6.7] RhythmFormer + 스펙트럼 손실의 MR-NIRP(주행 RGB) 악화 기제 진단 — 학습 없음.
입력: 체인 74/75/77 RhythmFormer 셀(RF_POOL_L·RF_SPEC_L·RF_SPEC_NH_L ×3시드)의 MR-NIRP 3축 저장 예측(pickle, pro6000 에서 회수) + Alpha 의 float 캐시 real_NIRP_*(차분 정규화 프레임 → 전역 밝기 차분 trace) + 정본 agg json(영상 HR).
영상별: 예측 스펙트럼 첨도(자기 피크 ±3 bpm 질량), 정답 대역(gt ±3) 질량, 라벨 SNR 프록시(라벨의 gt 대역 질량), 전역 trace 피크(교란) → 실패(|err|>5) 분류 lock-in / 반감 / 배가 / 기타, 예측 피크 = 교란 피크 일치율.
  python scripts/rf_nirp_diag.py → results/rf_nirp_diag.{txt,json}"""
import glob, json, os, pickle, re, sys
import numpy as np
from scipy.stats import wilcoxon, spearmanr
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from agg_model2 import cat_chunks
R = "/SPECPHYS_ROOT/_workspace/data/results"; PK = "/SPECPHYS_TMP/rf_nirp"; CACHE = "/SPECPHYS_ROOT/_workspace/data/physformer/cache"
FS = 30.0; AXES = ["NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE"]; MODELS = {"RF_POOL_L": "고유 손실", "RF_SPEC_L": "+스펙트럼(1.0/0.5)", "RF_SPEC_NH_L": "+스펙트럼, 반감 항 0"}
def seeds(m): return [m, m + "_s200", m + "_s300"]
def hp(x, k=31):
    x = np.asarray(x, np.float64); ker = np.ones(k) / k; m = np.convolve(x, ker, mode="same"); return x - m
def spectrum(x, lo=30, hi=200):
    x = hp(x); x = x - x.mean(); n = len(x); nfft = 8 * n
    P = np.abs(np.fft.rfft(x * np.hanning(n), n=nfft)) ** 2; f = np.fft.rfftfreq(nfft, 1.0 / FS) * 60.0
    m = (f >= lo) & (f <= hi); return f[m], P[m] / (P[m].sum() + 1e-12)
def peaks(bpm, P, rel=0.3, min_sep=4.0, k=3):
    idx = [i for i in range(1, len(P) - 1) if P[i] >= P[i - 1] and P[i] >= P[i + 1] and P[i] >= rel * P.max()]
    idx.sort(key=lambda i: -P[i]); out = []
    for i in idx:
        if all(abs(bpm[i] - o) >= min_sep for o in out): out.append(float(bpm[i]))
        if len(out) >= k: break
    return out
def band(bpm, P, c, w=3.0): return float(P[np.abs(bpm - c) <= w].sum())
def vidnum(k): return int(re.sub(r"\D", "", str(k)))
def global_trace(ax, vid):
    fs = sorted(glob.glob(f"{CACHE}/real_{ax}/subject{vid}_input*.npy"), key=lambda p: int(re.search(r"_input(\d+)\.npy", p).group(1)))
    return np.concatenate([np.load(f).mean(axis=(1, 2, 3)) for f in fs]) if fs else None
def load_pickle(m, ax):
    ps = glob.glob(f"{PK}/ev_{m}_{ax}/*/saved_test_outputs/*.pickle"); return pickle.load(open(ps[0], "rb")) if ps else None
def agg_rows(m, ax):
    f = f"{R}/agg_{m}.json"; return {r["vid"]: r for r in json.load(open(f)).get(ax, [])} if os.path.exists(f) else {}
def classify(pred, gt, pk, tol_lock=3.0, tol_ok=5.0):
    if abs(pred - gt) <= tol_ok: return "ok"
    if abs(pred / gt - 0.5) < 0.1: return "half"
    if abs(pred / gt - 2.0) < 0.2: return "double"
    if any(abs(pred - p) <= tol_lock and abs(p - gt) > tol_ok for p in pk): return "lockin"
    return "other"
LINES = []; out = {}
def head(s): print(s); LINES.append(s)
GT = {}  # (ax, vid) -> nuisance peaks, label snr
for ax in AXES:
    head(f"\n=== {ax} (n=19) ===")
    per_model = {}
    for m0, desc in MODELS.items():
        acc = {}  # vid -> list over seeds of dict
        for m in seeds(m0):
            d = load_pickle(m, ax); rows = agg_rows(m, ax)
            if d is None: continue
            for vk in d["predictions"]:
                vid = vidnum(vk); p = cat_chunks(d["predictions"][vk]); l = cat_chunks(d["labels"][vk]); r = rows.get(vid)
                if r is None: continue
                if (ax, vid) not in GT:
                    g = global_trace(ax, vid); bg, Pg = spectrum(g) if g is not None else (None, None); bl, Pl = spectrum(l)
                    GT[(ax, vid)] = {"pk_g": peaks(bg, Pg) if g is not None else [], "label_snr": band(bl, Pl, r["gt"]), "gt": r["gt"]}
                bp, Pp = spectrum(p); pkp = float(bp[np.argmax(Pp)]); gt = r["gt"]; pk_g = GT[(ax, vid)]["pk_g"]
                acc.setdefault(vid, []).append(dict(err=r["err"], pred=r["pred"], gt=gt, sharp=band(bp, Pp, pkp), hrmass=band(bp, Pp, gt), cls=classify(r["pred"], gt, pk_g),
                                                     on_nuis=int(any(abs(pkp - q) <= 3.0 for q in pk_g)), sign=np.sign(r["pred"] - gt)))
        if not acc: head(f"  {m0}: (missing)"); continue
        vids = sorted(acc); V = {v: {k: float(np.mean([s[k] for s in acc[v]])) for k in ("err", "sharp", "hrmass", "on_nuis", "sign", "pred")} for v in vids}
        cls = [c for v in vids for c in [s["cls"] for s in acc[v]]]; nf = sum(c != "ok" for c in cls)
        snr = np.array([GT[(ax, v)]["label_snr"] for v in vids]); err = np.array([V[v]["err"] for v in vids]); rho = spearmanr(snr, err).correlation if len(vids) > 4 else float("nan")
        per_model[m0] = V
        head(f"  {m0:13s} {desc:20s} MAE {err.mean():5.2f} | 실패(|err|>5) {nf}/{len(cls)} = lock-in {sum(c == 'lockin' for c in cls)} 반감 {sum(c == 'half' for c in cls)} 배가 {sum(c == 'double' for c in cls)} 기타 {sum(c == 'other' for c in cls)} | 예측 첨도 {np.mean([V[v]['sharp'] for v in vids]):.3f} 정답대역 질량 {np.mean([V[v]['hrmass'] for v in vids]):.3f} 예측피크=교란피크 {np.mean([V[v]['on_nuis'] for v in vids]):.2f} | 오차 부호(pred−gt) 평균 {np.mean([V[v]['sign'] for v in vids]):+.2f} | ρ(라벨 SNR, err) {rho:+.2f}")
        out[f"{ax}|{m0}"] = dict(mae=float(err.mean()), fail=nf, n=len(cls), lockin=sum(c == "lockin" for c in cls), half=sum(c == "half" for c in cls), double=sum(c == "double" for c in cls), other=sum(c == "other" for c in cls),
                                 sharp=float(np.mean([V[v]["sharp"] for v in vids])), hrmass=float(np.mean([V[v]["hrmass"] for v in vids])), on_nuis=float(np.mean([V[v]["on_nuis"] for v in vids])), sign=float(np.mean([V[v]["sign"] for v in vids])), rho_snr_err=float(rho))
    if "RF_POOL_L" in per_model and "RF_SPEC_L" in per_model:
        A, B = per_model["RF_POOL_L"], per_model["RF_SPEC_L"]; vs = sorted(set(A) & set(B))
        for k, nm in (("sharp", "예측 첨도"), ("hrmass", "정답대역 질량"), ("err", "오차")):
            x = np.array([B[v][k] for v in vs]); y = np.array([A[v][k] for v in vs])
            try: p = wilcoxon(x, y).pvalue
            except ValueError: p = 1.0
            head(f"    Δ(스펙트럼 − 고유) {nm}: {x.mean() - y.mean():+.3f} (p={p:.3f})"); out[f"{ax}|delta_{k}"] = dict(d=float(x.mean() - y.mean()), p=float(p))
        lo = [v for v in vs if GT[(ax, v)]["label_snr"] < np.median([GT[(ax, u)]["label_snr"] for u in vs])]; hi = [v for v in vs if v not in lo]
        head(f"    라벨 SNR 하위 절반(n={len(lo)}): 고유 {np.mean([A[v]['err'] for v in lo]):.2f} → 스펙트럼 {np.mean([B[v]['err'] for v in lo]):.2f} | 상위 절반(n={len(hi)}): {np.mean([A[v]['err'] for v in hi]):.2f} → {np.mean([B[v]['err'] for v in hi]):.2f}")
        out[f"{ax}|snr_split"] = dict(lo_native=float(np.mean([A[v]["err"] for v in lo])), lo_spec=float(np.mean([B[v]["err"] for v in lo])), hi_native=float(np.mean([A[v]["err"] for v in hi])), hi_spec=float(np.mean([B[v]["err"] for v in hi])))
        worse = sorted(vs, key=lambda v: A[v]["err"] - B[v]["err"])[:5]
        gts = np.array([GT[(ax, v)]["gt"] for v in vs]); derr = np.array([B[v]["err"] - A[v]["err"] for v in vs]); rho_gt = spearmanr(gts, derr).correlation
        lowhr = [v for v in vs if GT[(ax, v)]["gt"] < 65]; hihr = [v for v in vs if GT[(ax, v)]["gt"] >= 65]
        head(f"    ρ(gt HR, Δerr 스펙트럼−고유) {rho_gt:+.2f} | gt<65 bpm(n={len(lowhr)}): 고유 {np.mean([A[v]['err'] for v in lowhr]):.2f} → 스펙트럼 {np.mean([B[v]['err'] for v in lowhr]):.2f} (스펙트럼 평균 예측 {np.mean([B[v]['pred'] if 'pred' in B[v] else 0 for v in lowhr]):.0f}) | gt≥65(n={len(hihr)}): {np.mean([A[v]['err'] for v in hihr]):.2f} → {np.mean([B[v]['err'] for v in hihr]):.2f}")
        out[f"{ax}|hr_split"] = dict(rho_gt_derr=float(rho_gt), low_native=float(np.mean([A[v]["err"] for v in lowhr])), low_spec=float(np.mean([B[v]["err"] for v in lowhr])), hi_native=float(np.mean([A[v]["err"] for v in hihr])), hi_spec=float(np.mean([B[v]["err"] for v in hihr])))
        head("    가장 악화된 영상 5(vid: 고유→스펙트럼 err, gt, 교란 피크, 라벨 SNR): " + "; ".join(f"{v}: {A[v]['err']:.1f}→{B[v]['err']:.1f}, gt {GT[(ax, v)]['gt']:.0f}, pk {[round(q) for q in GT[(ax, v)]['pk_g']]}, snr {GT[(ax, v)]['label_snr']:.2f}" for v in worse))
open(f"{R}/rf_nirp_diag.txt", "w").write("\n".join(LINES) + "\n"); json.dump(out, open(f"{R}/rf_nirp_diag.json", "w"), indent=1, ensure_ascii=False, default=float)
print(f"\n-> {R}/rf_nirp_diag.txt / .json")
