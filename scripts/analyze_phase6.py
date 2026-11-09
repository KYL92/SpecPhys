#!/usr/bin/env python3
"""6단계: 파형 고조파 다양화 학습(CoTNH)의 효과 분석.

CoTNH를 base / T2L_s100 / CoT / CoTN 과 동일 프로토콜로 비교한다.
- 실데이터 세트(MCD_A, MCD_B, REALTEST, RTNC, RTNI): 16번 보고서의 기존 표와 동일하게
  rPPG-Toolbox 평가 프로토콜(cumsum → detrend(λ=100) → butter1 0.6–3.3Hz → periodogram
  nfft=next_pow2)로 예측 HR과 GT HR(라벨 파형에 동일 처리)을 구한다.
- 합성 세트(C3, C6, E3): analyze_phase2.py 의 video_hr(n*8 zero-pad rfft, 0.7–4.0Hz) 대
  meta.json achieved_bpm — 기존 phase2/3/4 표와 동일.
- 고조파 배가율(pred/gt ∈ [1.8, 2.2]) / 반감율([0.45, 0.55]), 배가 사례의 기본파/피크 파워비,
  Wilcoxon signed-rank(대응 per-video 절대오차), 정확 McNemar(배가 플래그).
결과: _workspace/data/results/phase6_harmonic.json, phase6_harmonic_diversity.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import glob, json, pickle, sys
import numpy as np, torch
import scipy.signal as ss
from scipy import stats
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/SPECPHYS_ROOT"
PF = f"{ROOT}/_workspace/data/physformer"
RES = f"{ROOT}/_workspace/data/results"
sys.path.insert(0, f"{ROOT}/repos/rPPG-Toolbox")
from evaluation.post_process import _calculate_fft_hr, _detrend  # noqa: E402

fs = 30
MODELS = ["base", "T2L_s100", "CoT", "CoTN", "CoTNH"]
# 6b 확장: ablation(CoTNC=압축만, CoTNW=파형만, CoTNH2=실분포 정합 파형) + CoTNH-M(in-domain 적응, 제로샷 아님)
EXTRA_MODELS = ["CoTNC", "CoTNW", "CoTNH2", "CoTNH-M", "CoTA", "CoTAF", "CoTB", "CoTBH"]
INDOMAIN = {"CoTNH-M", "CoTA", "CoTAF", "CoTB", "CoTBH"}   # MCD 정체성/PURE 피험자를 학습에 포함 — 제로샷 아님
# 모델명 → 런 디렉토리 이름 alias (ev_{dir}_{set})
MODEL_DIR = {"CoTNH-M": "CoTNHM"}
# 2×2 ablation 설계: 파형 다양화(W) × 압축 증강(C)
ABL = {"CoTN": (0, 0), "CoTNC": (0, 1), "CoTNW": (1, 0), "CoTNH": (1, 1)}
if "--all" in sys.argv:
    MODELS = MODELS + EXTRA_MODELS
elif "--models" in sys.argv:
    MODELS = sys.argv[sys.argv.index("--models") + 1].split(",")
PROPOSED = [m for m in MODELS if m not in ("base", "T2L_s100", "CoT", "CoTN")]
PAIRS = [(a, b) for b in PROPOSED for a in ("base", "CoTN") if a in MODELS] + [("base", "CoTN")]
PAIRS += [(a, b) for a, b in [("CoTNC", "CoTNH"), ("CoTNW", "CoTNH"), ("CoTNH", "CoTNH2"), ("CoTNC", "CoTNH2"), ("CoTNW", "CoTNH2"), ("CoTNH", "CoTNH-M")] if a in MODELS and b in MODELS]
# PURE: 59트라이얼, subject 키 = 피험자*100+태스크(예: '1001'); PURE_ho = 피험자 07~10(id ≥ 700) held-out; PURE_st = steady 태스크(id%100==1)
REAL_SETS = {"MCD_B": "MCD_B", "MCD_A": "MCD_A", "REALTEST": "REALTEST", "RTNC": "RTNC", "RTNI": "RTNI",
             "PURE": "PURE", "PURE_ho": "PURE", "PURE_st": "PURE"}
SET_FILTER = {"PURE_ho": lambda sid: sid >= 700, "PURE_st": lambda sid: sid % 100 == 1}
SYN_SETS = ["C1", "C3", "C6", "E3"]
# REALTEST 는 CoTNH 만 ev_*_REALTEST 이고 나머지는 ev_*_RT
RUN_ALIAS = {"REALTEST": ["REALTEST", "RT"]}
DOUBLE = (1.8, 2.2)
HALF = (0.45, 0.55)

# 기존 보고 수치(검증용). MCD: 16번 보고서 5단계 표. 주의: 디렉토리 MCD_A=after(운동직후), MCD_B=before(안정)
REPORTED = {
    "MCD_B": {"base": 29.27, "T2L_s100": 46.89, "CoT": 49.61, "CoTN": 42.63},
    "MCD_A": {"base": 32.52, "T2L_s100": 50.14, "CoT": 35.99, "CoTN": 32.65},
    "REALTEST": {"base": 1.43, "T2L_s100": 1.87, "CoT": 1.43, "CoTN": 1.43},
    "RTNI": {"base": 30.32, "T2L_s100": 18.35, "CoT": 30.32, "CoTN": 0.00},
    "C3": {"base": None, "T2L_s100": 0.14, "CoT": 0.14, "CoTN": 0.14},
    "C6": {"base": None, "T2L_s100": 0.09, "CoT": 17.54, "CoTN": 0.09},
    "E3": {"base": None, "T2L_s100": 0.15, "CoT": 0.15, "CoTN": 0.15},
}
REPORTED_DOUBLING = {"MCD_B": {"base": 6, "CoTN": 10}}  # "base 6/20, CoTN 10/20" (안정 세트 기준으로 추정)
CLASSICAL = {"MCD_B": {"POS": 4.75, "CHROM": 4.92}, "MCD_A": {"POS": 12.17, "CHROM": 15.42}}
CLASSICAL_PURE = {"PURE": 2.17, "PURE_ho": 2.74}  # 16 보고서 7단계 표 POS(n=55 / held-out n=22); CHROM 4.76 / 7.01


def cat_chunks(chunks):
    return np.concatenate([chunks[k].detach().cpu().float().numpy().ravel() for k in sorted(chunks)])


def toolbox_hr(sig, lo=0.6, hi=3.3):
    """rPPG-Toolbox calculate_metric_per_video(diff_flag=True) 와 동일한 HR 추정. 필터된 파형도 반환."""
    x = _detrend(np.cumsum(sig), 100)
    b, a = ss.butter(1, [lo / fs * 2, hi / fs * 2], btype="bandpass")
    x = ss.filtfilt(b, a, np.double(x))
    return float(_calculate_fft_hr(x, fs=fs, low_pass=lo, high_pass=hi)), x


def video_hr(sig):
    """analyze_phase2.video_hr 와 동일(고해상 FFT)."""
    sig = np.cumsum(sig - sig.mean())
    b, a = butter(3, [0.7 / (fs / 2), 4.0 / (fs / 2)], btype="band")
    f = filtfilt(b, a, sig)
    fr = np.fft.rfftfreq(len(f) * 8, 1 / fs)
    pw = np.abs(np.fft.rfft(f - f.mean(), n=len(f) * 8)) ** 2
    band = (fr >= 0.7) & (fr <= 4.0)
    return float(fr[band][pw[band].argmax()] * 60), f


def fundamental_ratio(filtered, gt_bpm, tol_bpm=3.0):
    """고해상 스펙트럼에서 GT 기본파(±tol) 파워 / 대역 내 피크 파워."""
    fr = np.fft.rfftfreq(len(filtered) * 8, 1 / fs) * 60
    pw = np.abs(np.fft.rfft(filtered - filtered.mean(), n=len(filtered) * 8)) ** 2
    band = (fr >= 42) & (fr <= 240)
    peak = pw[band].max()
    fund = pw[(fr >= gt_bpm - tol_bpm) & (fr <= gt_bpm + tol_bpm)].max()
    return float(fund / peak)


def load_pickle(model, dset):
    names = RUN_ALIAS.get(dset, [REAL_SETS.get(dset, dset)])
    mdir = MODEL_DIR.get(model, model)
    for n in names:
        pk = glob.glob(f"{PF}/runs/ev_{mdir}_{n}/*/saved_test_outputs/*.pickle")
        if pk:
            return pickle.load(open(pk[0], "rb"))
    return None


def subj_key(s):
    return int(s.replace("subject", ""))


def eval_real(model, dset):
    d = load_pickle(model, dset)
    if d is None:
        return None
    rows = []
    flt = SET_FILTER.get(dset)
    for s in sorted(d["predictions"], key=subj_key):
        if flt is not None and not flt(subj_key(s)):
            continue
        pr = cat_chunks(d["predictions"][s])
        lb = cat_chunks(d["labels"][s])
        hp, pf = toolbox_hr(pr)
        hg, _ = toolbox_hr(lb)
        hp_hi, pf_hi = video_hr(pr)
        hg_hi, _ = video_hr(lb)
        ratio = hp / hg
        rows.append({"subject": s, "pred": round(hp, 2), "gt": round(hg, 2), "abs_err": round(abs(hp - hg), 2),
                     "ratio": round(ratio, 3),
                     "doubled": bool(DOUBLE[0] <= ratio <= DOUBLE[1]),
                     "halved": bool(HALF[0] <= ratio <= HALF[1]),
                     "pred_hires": round(hp_hi, 2), "gt_hires": round(hg_hi, 2),
                     "fund_ratio": round(fundamental_ratio(pf_hi, hg_hi), 4)})
    return rows


def eval_syn(model, dset):
    d = load_pickle(model, dset)
    if d is None:
        return None
    rows = []
    for s in sorted(d["predictions"], key=subj_key):
        gt = json.load(open(f"{PF}/TEST_{dset}/{s}/meta.json"))["achieved_bpm"]
        hp, pf = video_hr(cat_chunks(d["predictions"][s]))
        ratio = hp / gt
        rows.append({"subject": s, "pred": round(hp, 2), "gt": float(gt), "abs_err": round(abs(hp - gt), 2),
                     "ratio": round(ratio, 3),
                     "doubled": bool(DOUBLE[0] <= ratio <= DOUBLE[1]),
                     "halved": bool(HALF[0] <= ratio <= HALF[1]),
                     "fund_ratio": round(fundamental_ratio(pf, gt), 4)})
    return rows


def summarize(rows):
    e = np.array([r["abs_err"] for r in rows])
    p = np.array([r["pred"] for r in rows]); g = np.array([r["gt"] for r in rows])
    pear = float(stats.pearsonr(p, g)[0]) if len(rows) > 2 and p.std() > 0 and g.std() > 0 else float("nan")
    return {"n": len(rows), "MAE": round(float(e.mean()), 2), "median_AE": round(float(np.median(e)), 2),
            "RMSE": round(float(np.sqrt((e ** 2).mean())), 2), "pearson": round(pear, 3),
            "doubled": int(sum(r["doubled"] for r in rows)), "halved": int(sum(r["halved"] for r in rows)),
            "pred_ge140": int(sum(r["pred"] >= 140 for r in rows)),
            "accurate_le3": int(sum(r["abs_err"] <= 3 for r in rows))}


def wilcoxon_pair(rows_a, rows_b):
    a = np.array([r["abs_err"] for r in rows_a]); b = np.array([r["abs_err"] for r in rows_b])
    d = b - a
    nz = d[d != 0]
    out = {"n": int(len(d)), "n_nonzero": int(len(nz)), "mean_diff": round(float(d.mean()), 2),
           "median_diff": round(float(np.median(d)), 2)}
    if len(nz) < 5:
        out.update({"W": None, "p": None, "z": None, "r": None, "rank_biserial": None, "note": "nonzero pairs < 5"})
        return out
    w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="exact" if len(nz) <= 25 else "approx")
    # 효과 크기: matched-pairs rank-biserial r = (R+ - R-)/(R+ + R-), 및 z 기반 r
    ranks = stats.rankdata(np.abs(nz))
    rplus = ranks[nz > 0].sum(); rminus = ranks[nz < 0].sum()
    rb = (rplus - rminus) / (rplus + rminus)
    n = len(nz)
    mu = n * (n + 1) / 4; sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (min(rplus, rminus) - mu) / sd
    out.update({"W": float(w.statistic), "p": round(float(w.pvalue), 4), "z": round(float(z), 3),
                "r": round(float(abs(z) / np.sqrt(n)), 3), "rank_biserial": round(float(rb), 3),
                "n_b_worse": int((d > 0).sum()), "n_b_better": int((d < 0).sum())})
    return out


def mcnemar_pair(rows_a, rows_b):
    a = np.array([r["doubled"] for r in rows_a]); b = np.array([r["doubled"] for r in rows_b])
    b01 = int(((~a) & b).sum()); b10 = int((a & (~b)).sum())
    n_disc = b01 + b10
    p = float(stats.binomtest(b01, n_disc, 0.5).pvalue) if n_disc > 0 else 1.0
    return {"a_only": b10, "b_only": b01, "both": int((a & b).sum()), "neither": int(((~a) & (~b)).sum()),
            "p_exact": round(p, 4), "n_discordant": n_disc}


def sign_test(d):
    """부호검정(0 제외) + 중앙값 차이 + 부호 비율."""
    d = np.asarray(d, float); nz = d[d != 0]
    pos = int((nz > 0).sum()); neg = int((nz < 0).sum())
    p = float(stats.binomtest(pos, len(nz), 0.5).pvalue) if len(nz) else 1.0
    return {"n": int(len(d)), "n_nonzero": int(len(nz)), "n_pos": pos, "n_neg": neg,
            "median": round(float(np.median(d)), 2), "mean": round(float(d.mean()), 2), "p_sign": round(p, 4)}


def wilcoxon_diff(d):
    """단일 표본 Wilcoxon(차이 벡터 vs 0) + r, rank-biserial."""
    d = np.asarray(d, float); nz = d[d != 0]
    if len(nz) < 5:
        return {"W": None, "p": None, "r": None, "rank_biserial": None, "note": "nonzero < 5"}
    w = stats.wilcoxon(nz, alternative="two-sided", method="exact" if len(nz) <= 25 else "approx")
    ranks = stats.rankdata(np.abs(nz)); rplus = ranks[nz > 0].sum(); rminus = ranks[nz < 0].sum()
    n = len(nz); mu = n * (n + 1) / 4; sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (min(rplus, rminus) - mu) / sd
    return {"W": float(w.statistic), "p": round(float(w.pvalue), 4), "z": round(float(z), 3),
            "r": round(float(abs(z) / np.sqrt(n)), 3), "rank_biserial": round(float((rplus - rminus) / (rplus + rminus)), 3)}


def factorial_2x2(pv):
    """pv: {model: rows} (같은 영상 순서). per-video 절대오차에 대해
    W 주효과 = mean over C of (AE[W=1,C]-AE[W=0,C]); C 주효과 = mean over W of (AE[W,C=1]-AE[W,C=0]);
    상호작용 = (CoTNH-CoTNW)-(CoTNC-CoTN). 각각 부호검정 + Wilcoxon + 효과크기.
    배가 플래그: 요인별 McNemar(정확) — 두 수준을 합친 영상×조건 쌍 기준 및 단순 효과."""
    if not all(m in pv for m in ABL):
        return None
    ae = {m: np.array([r["abs_err"] for r in pv[m]]) for m in ABL}
    db = {m: np.array([r["doubled"] for r in pv[m]]) for m in ABL}
    n = len(ae["CoTN"])
    wA = ((ae["CoTNW"] - ae["CoTN"]) + (ae["CoTNH"] - ae["CoTNC"])) / 2
    cA = ((ae["CoTNC"] - ae["CoTN"]) + (ae["CoTNH"] - ae["CoTNW"])) / 2
    ix = (ae["CoTNH"] - ae["CoTNW"]) - (ae["CoTNC"] - ae["CoTN"])
    out = {"n": n, "cell_MAE": {m: round(float(ae[m].mean()), 2) for m in ABL},
           "cell_doubled": {m: int(db[m].sum()) for m in ABL}}
    for name, d in [("W_main", wA), ("C_main", cA), ("interaction", ix),
                    ("W_simple_noC", ae["CoTNW"] - ae["CoTN"]), ("W_simple_withC", ae["CoTNH"] - ae["CoTNC"]),
                    ("C_simple_noW", ae["CoTNC"] - ae["CoTN"]), ("C_simple_withW", ae["CoTNH"] - ae["CoTNW"])]:
        out[name] = {"sign": sign_test(d), "wilcoxon": wilcoxon_diff(d),
                     "cohen_dz": round(float(d.mean() / d.std(ddof=1)), 3) if d.std(ddof=1) > 0 else None}
    # 배가: 요인 주효과를 (영상, 다른 요인 수준) 쌍 2n개로 스택한 McNemar; 단순효과 McNemar
    def mc(a, b):
        b01 = int(((~a) & b).sum()); b10 = int((a & (~b)).sum()); nd = b01 + b10
        return {"a_only": b10, "b_only": b01, "p_exact": round(float(stats.binomtest(b01, nd, 0.5).pvalue), 4) if nd else 1.0}
    out["doubling"] = {
        "W_main_stacked": mc(np.r_[db["CoTN"], db["CoTNC"]], np.r_[db["CoTNW"], db["CoTNH"]]),
        "C_main_stacked": mc(np.r_[db["CoTN"], db["CoTNW"]], np.r_[db["CoTNC"], db["CoTNH"]]),
        "W_simple_noC": mc(db["CoTN"], db["CoTNW"]), "W_simple_withC": mc(db["CoTNC"], db["CoTNH"]),
        "C_simple_noW": mc(db["CoTN"], db["CoTNC"]), "C_simple_withW": mc(db["CoTNW"], db["CoTNH"])}
    # 로그 절대오차의 순위 기반 2요인(정렬 순위 ANOVA, 참고용): rank(AE) 를 종속변수로 OLS
    try:
        import statsmodels.formula.api as smf, statsmodels.api as sm, pandas as pd
        df = pd.DataFrame({"ae": np.concatenate([ae[m] for m in ABL]),
                           "W": np.concatenate([[ABL[m][0]] * n for m in ABL]),
                           "C": np.concatenate([[ABL[m][1]] * n for m in ABL]),
                           "vid": np.tile(np.arange(n), 4)})
        df["rk"] = stats.rankdata(df["ae"])
        fit = smf.ols("rk ~ C(vid) + W * C", data=df).fit()
        an = sm.stats.anova_lm(fit, typ=2)
        out["rank_anova_typ2"] = {k: {"F": round(float(an.loc[k, "F"]), 3), "p": round(float(an.loc[k, "PR(>F)"]), 4),
                                      "df": int(an.loc[k, "df"])} for k in ["W", "C", "W:C"]}
        out["rank_anova_typ2"]["resid_df"] = int(an.loc["Residual", "df"])
    except Exception as e:  # noqa
        out["rank_anova_typ2"] = {"error": str(e)}
    return out


def main():
    out = {"protocol": {
        "real_sets": "toolbox: cumsum→detrend(λ=100)→butter1 0.6–3.3Hz→periodogram nfft=next_pow2; GT=label 동일 처리",
        "syn_sets": "phase2 video_hr: cumsum→butter3 0.7–4.0Hz→rfft n*8; GT=meta.achieved_bpm",
        "doubling": f"pred/gt in {DOUBLE}", "halving": f"pred/gt in {HALF}",
        "fund_ratio": "고해상 rfft(n*8)에서 GT±3BPM 최대 파워 / 42–240BPM 대역 피크 파워",
        "note": "디렉토리 MCD_A=after(운동직후), MCD_B=before(안정). 16번 보고서 표의 '안정'=MCD_B."},
        "per_video": {}, "summary": {}, "verification": {}, "tests": {}, "fund_ratio": {}, "classical_reported": CLASSICAL,
        "indomain_models": sorted(INDOMAIN), "ablation_2x2": {}}
    out["protocol"]["PURE"] = "59 trials, subject key = subject*100+task; PURE_ho = id>=700 (subjects 07-10); PURE_st = task 01 steady only"
    out["protocol"]["canonical"] = "정본 HR 정의 = toolbox 프로토콜(pred/gt 열, 배가 = ratio 1.8–2.2); *_hires 열은 보조"
    sets = list(REAL_SETS) + SYN_SETS
    for ds in sets:
        out["per_video"][ds] = {}; out["summary"][ds] = {}
        for m in MODELS:
            rows = eval_real(m, ds) if ds in REAL_SETS else eval_syn(m, ds)
            if rows is None:
                continue
            out["per_video"][ds][m] = rows
            out["summary"][ds][m] = summarize(rows)
    # 검증: 기존 보고 수치와 비교
    for ds, rep in REPORTED.items():
        for m, v in rep.items():
            if v is None or m not in out["summary"].get(ds, {}):
                continue
            got = out["summary"][ds][m]["MAE"]
            out["verification"][f"{ds}/{m}"] = {"reported": v, "recomputed": got, "match": bool(abs(got - v) < 0.06)}
    for ds, rep in REPORTED_DOUBLING.items():
        for m, v in rep.items():
            if m not in out["summary"].get(ds, {}):
                continue
            got = out["summary"][ds][m]["doubled"]
            out["verification"][f"{ds}/{m}/doubled"] = {"reported": v, "recomputed": got, "match": got == v}
    # 통계 검정
    for ds in sets:
        out["tests"][ds] = {}
        for a, b in PAIRS:
            if a in out["per_video"][ds] and b in out["per_video"][ds]:
                out["tests"][ds][f"{a}_vs_{b}"] = {"wilcoxon_abs_err": wilcoxon_pair(out["per_video"][ds][a], out["per_video"][ds][b]),
                                                   "mcnemar_doubling": mcnemar_pair(out["per_video"][ds][a], out["per_video"][ds][b])}
    # MCD 두 상태 합산(40영상) 검정
    for a, b in PAIRS:
        if not (a in out["per_video"]["MCD_B"] and b in out["per_video"]["MCD_B"] and a in out["per_video"]["MCD_A"] and b in out["per_video"]["MCD_A"]):
            continue
        ra = out["per_video"]["MCD_B"][a] + out["per_video"]["MCD_A"][a]
        rb = out["per_video"]["MCD_B"][b] + out["per_video"]["MCD_A"][b]
        out["tests"].setdefault("MCD_pooled", {})[f"{a}_vs_{b}"] = {"wilcoxon_abs_err": wilcoxon_pair(ra, rb), "mcnemar_doubling": mcnemar_pair(ra, rb)}
        out["summary"].setdefault("MCD_pooled", {})[a] = summarize(ra)
        out["summary"]["MCD_pooled"][b] = summarize(rb)
    # 2×2 ablation(파형 × 압축): 상태별 + 40영상 풀링
    for ds in ["MCD_B", "MCD_A"]:
        r = factorial_2x2(out["per_video"][ds])
        if r:
            out["ablation_2x2"][ds] = r
    if all(m in out["per_video"]["MCD_B"] and m in out["per_video"]["MCD_A"] for m in ABL):
        out["ablation_2x2"]["MCD_pooled"] = factorial_2x2({m: out["per_video"]["MCD_B"][m] + out["per_video"]["MCD_A"][m] for m in ABL})
    # 배가 사례의 기본파/피크 파워비
    for ds in ["MCD_B", "MCD_A"]:
        out["fund_ratio"][ds] = {}
        for m in MODELS:
            rows = out["per_video"][ds].get(m, [])
            if not rows:
                continue
            fr = [r["fund_ratio"] for r in rows if r["doubled"]]
            ok = [r["fund_ratio"] for r in rows if r["abs_err"] <= 3]
            out["fund_ratio"][ds][m] = {"n_doubled": len(fr),
                                        "doubled_median": round(float(np.median(fr)), 4) if fr else None,
                                        "doubled_min": round(float(min(fr)), 4) if fr else None,
                                        "doubled_max": round(float(max(fr)), 4) if fr else None,
                                        "accurate_median": round(float(np.median(ok)), 4) if ok else None}
    json.dump(out, open(f"{RES}/phase6_harmonic.json", "w"), indent=1, ensure_ascii=False)

    # 콘솔 표
    ZS = [m for m in MODELS if m not in INDOMAIN]; ID = [m for m in MODELS if m in INDOMAIN]
    print("== HR MAE (RMSE) [doubled/halved/pred>=140] == (zero-shot | in-domain)")
    print(f"{'set':10s}" + "".join(f"{m:>25s}" for m in ZS) + "  ||" + "".join(f"{m + '(ID)':>25s}" for m in ID))
    for ds in sets + ["MCD_pooled"]:
        row = f"{ds:10s}"
        for m in ZS + ["||"] + ID:
            if m == "||":
                row += "  ||"; continue
            s = out["summary"][ds].get(m)
            row += f"{s['MAE']:8.2f} ({s['RMSE']:5.2f}) [{s['doubled']:2d}/{s['halved']:2d}/{s['pred_ge140']:2d}]" if s else f"{'--':>25s}"
        print(row)
    print("\n== verification ==")
    for k, v in out["verification"].items():
        print(f"{k:22s} reported={v['reported']}  recomputed={v['recomputed']}  {'OK' if v['match'] else 'MISMATCH'}")
    print("\n== MCD per-video (pred/gt, D=doubled) ==")
    for ds in ["MCD_B", "MCD_A"]:
        print(ds)
        subs = [r["subject"] for r in out["per_video"][ds]["base"]]
        for i, s in enumerate(subs):
            print(f"  {s:10s} gt={out['per_video'][ds]['base'][i]['gt']:6.1f} " +
                  " ".join(f"{m}={out['per_video'][ds][m][i]['pred']:6.1f}{'D' if out['per_video'][ds][m][i]['doubled'] else ' '}" for m in MODELS if m in out['per_video'][ds]))
    print("\n== tests ==")
    for ds, t in out["tests"].items():
        for k, v in t.items():
            w = v["wilcoxon_abs_err"]; mc = v["mcnemar_doubling"]
            print(f"{ds:10s} {k:14s} W={w.get('W')} p={w.get('p')} r={w.get('r')} rb={w.get('rank_biserial')} mdiff={w['median_diff']} | McNemar a_only={mc['a_only']} b_only={mc['b_only']} p={mc['p_exact']}")
    print("\n== fund_ratio ==", json.dumps(out["fund_ratio"], indent=1))
    print("\n== 2x2 ablation (W=waveform diversity, C=compression aug) ==")
    for ds, r in out["ablation_2x2"].items():
        print(ds, "cells MAE", r["cell_MAE"], "doubled", r["cell_doubled"])
        for k in ["W_main", "C_main", "interaction", "W_simple_noC", "W_simple_withC", "C_simple_noW", "C_simple_withW"]:
            sg = r[k]["sign"]; wx = r[k]["wilcoxon"]
            print(f"  {k:15s} median={sg['median']:7.2f} mean={sg['mean']:7.2f} +/-={sg['n_pos']}/{sg['n_neg']} p_sign={sg['p_sign']} | W={wx.get('W')} p={wx.get('p')} r={wx.get('r')} rb={wx.get('rank_biserial')} dz={r[k]['cohen_dz']}")
        print("  doubling", json.dumps(r["doubling"]))
        print("  rank ANOVA", json.dumps(r.get("rank_anova_typ2")))
    if "PURE" in out["summary"]:
        print("\n== PURE (59 / held-out id>=700 / steady task01) MAE [doubled] ==")
        for m in MODELS:
            cells = []
            for ds in ["PURE", "PURE_ho", "PURE_st"]:
                s = out["summary"][ds].get(m)
                cells.append(f"{s['MAE']:6.2f} [{s['doubled']:2d}/{s['n']:2d}]" if s else f"{'--':>14s}")
            print(f"  {m + (' (ID)' if m in INDOMAIN else ''):16s}" + "  ".join(cells))

    # 그림
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [2.2, 1.3, 1.3]})
    cols = {"base": "#7f7f7f", "T2L_s100": "#1f77b4", "CoT": "#2ca02c", "CoTN": "#ff7f0e", "CoTNH": "#d62728",
            "CoTNC": "#bcbd22", "CoTNW": "#17becf", "CoTNH2": "#9467bd", "CoTNH-M": "#e377c2"}
    MODELS_F = [m for m in MODELS if any(m in out["summary"][k] for k in out["summary"])]
    nM = len(MODELS_F); off = (nM - 1) / 2
    ax = axes[0]
    labs = ["MCD_B\nrest", "MCD_A\npost-ex", "PURE\n59", "PURE\nheld-out", "REALTEST\nclean", "RTNC\nreal+clean", "RTNI\nreal+illum", "C3", "C6", "E3"]
    keys = ["MCD_B", "MCD_A", "PURE", "PURE_ho", "REALTEST", "RTNC", "RTNI", "C3", "C6", "E3"]
    keys = [k for k in keys if k in out["summary"]]; labs = labs[:len(keys)]
    x = np.arange(len(keys)); w = 0.8 / nM
    for i, m in enumerate(MODELS_F):
        vals = [out["summary"][k].get(m, {}).get("MAE", np.nan) for k in keys]
        ax.bar(x + (i - off) * w, vals, w, label=m + (" (in-domain)" if m in INDOMAIN else ""), color=cols.get(m, None))
    for xi, k in enumerate(["MCD_B", "MCD_A"]):
        ax.hlines(CLASSICAL[k]["POS"], xi - 0.45, xi + 0.45, colors="k", linestyles="--", lw=1)
        ax.text(xi + 0.46, CLASSICAL[k]["POS"], f"POS {CLASSICAL[k]['POS']}", fontsize=7, va="center")
    for k, v in CLASSICAL_PURE.items():
        if k in keys:
            xi = keys.index(k); ax.hlines(v, xi - 0.45, xi + 0.45, colors="k", linestyles="--", lw=1)
            ax.text(xi + 0.46, v, f"POS {v}", fontsize=7, va="center")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8); ax.set_ylabel("HR MAE (BPM)")
    ax.set_title("HR MAE by evaluation set (CoTNH = harmonic-diverse waveform training)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    for i, m in enumerate(MODELS_F):
        vals = [out["summary"][k].get(m, {}).get("doubled", np.nan) for k in ["MCD_B", "MCD_A"]]
        ax.bar(np.arange(2) + (i - off) * w, vals, w, color=cols.get(m, None), label=m)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["MCD_B rest", "MCD_A post-ex"]); ax.set_ylabel("videos with 2x doubling (/20)")
    ax.set_ylim(0, 20); ax.set_title("Harmonic doubling count", fontsize=10); ax.grid(axis="y", alpha=0.3)
    ax = axes[2]
    for i, m in enumerate(MODELS_F):
        pts = [r["fund_ratio"] * 100 for k in ["MCD_B", "MCD_A"] for r in out["per_video"][k].get(m, []) if r["doubled"]]
        ax.scatter(np.full(len(pts), i) + np.random.RandomState(0).uniform(-0.15, 0.15, len(pts)), pts, color=cols.get(m, None), s=18)
        if not pts:
            ax.text(i, 10, "n=0", ha="center", fontsize=7)
    ax.set_xticks(range(nM)); ax.set_xticklabels(MODELS_F, fontsize=7, rotation=30)
    ax.set_ylabel("fundamental / peak power (%) in doubled videos"); ax.set_yscale("log")
    ax.set_title("Residual fundamental in doubled outputs (MCD)", fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{RES}/phase6_harmonic_diversity.png", dpi=120)

    # 2×2 ablation 전용 그림
    if out["ablation_2x2"]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
        for ai, (ds, ttl) in enumerate([("MCD_B", "MCD_B rest (n=20)"), ("MCD_A", "MCD_A post-exercise (n=20)")]):
            ax = axes[ai]; r = out["ablation_2x2"].get(ds)
            if not r:
                continue
            for c, (lab, ls) in enumerate([("no compression", "--"), ("+ compression", "-")]):
                ys = [r["cell_MAE"][m] for m in ABL if ABL[m][1] == c]
                ax.plot([0, 1], ys, ls, marker="o", color=["#7f7f7f", "#d62728"][c], label=lab)
                for xw, m in zip([0, 1], [m for m in ABL if ABL[m][1] == c]):
                    ax.annotate(f"{m}\n{r['cell_MAE'][m]:.2f} / D{r['cell_doubled'][m]}", (xw, r["cell_MAE"][m]), textcoords="offset points", xytext=(6, 4), fontsize=7)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["neurokit waveform", "harmonic-diverse waveform"]); ax.set_xlim(-0.3, 1.5)
            ax.set_ylabel("HR MAE (BPM)"); ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8)
            ax.text(0.02, 0.02, f"W main p={r['W_main']['sign']['p_sign']}, C main p={r['C_main']['sign']['p_sign']}, WxC p={r['interaction']['sign']['p_sign']} (sign test)", transform=ax.transAxes, fontsize=7)
        ax = axes[2]; r = out["ablation_2x2"].get("MCD_pooled")
        if r:
            pv = {m: out["per_video"]["MCD_B"][m] + out["per_video"]["MCD_A"][m] for m in ABL}
            for i, m in enumerate(ABL):
                ys = [x["abs_err"] for x in pv[m]]
                ax.scatter(np.full(len(ys), i) + np.random.RandomState(1).uniform(-0.18, 0.18, len(ys)), ys, s=14, color=cols.get(m), alpha=0.8)
                ax.hlines(np.median(ys), i - 0.3, i + 0.3, colors="k", lw=1.5)
            ax.set_xticks(range(4)); ax.set_xticklabels([f"{m}\nW={ABL[m][0]} C={ABL[m][1]}" for m in ABL], fontsize=8)
            ax.set_ylabel("per-video |HR error| (BPM), pooled 40"); ax.set_title("Pooled MCD per-video error (bar = median)", fontsize=10); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{RES}/phase6_ablation_2x2.png", dpi=120)
        print("saved", f"{RES}/phase6_ablation_2x2.png")
    print("saved", f"{RES}/phase6_harmonic.json", f"{RES}/phase6_harmonic_diversity.png")


if __name__ == "__main__":
    main()
