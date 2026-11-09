#!/usr/bin/env python3
"""최종 통계검정 (정본 프로토콜 per-video 오차 기반, 대응표본):
1) 신규 제로샷 4축: CoTAF vs POS, 그리고 운전 대동작 CoTA vs CoTAF
2) MCD·PURE·UBFC: CoTAF vs CoTA
3) PURE: CoTAF vs POS
4) 순수 제로샷 축에서 코퍼스 수 단조 경향: base→CoTNH-M→CoTA→CoTAF (Friedman + Page)
Wilcoxon signed-rank(양측, zero_method='wilcox'), 효과크기 r=|Z|/√n.
"""
import glob, json, os, re, sys
import numpy as np
import scipy.signal as ss
from scipy import stats

sys.path.insert(0, "/SPECPHYS_ROOT/repos/rPPG-Toolbox")
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from evaluation.post_process import _calculate_fft_hr  # noqa
from unsupervised_methods.methods.POS_WANG import POS_WANG  # noqa
from agg_model2 import fast_detrend  # noqa

fs = 30
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
RES = "/SPECPHYS_ROOT/_workspace/data/results"


def canon_hr(sig, diff=False):
    x = np.cumsum(sig) if diff else np.asarray(sig, dtype=float)
    x = fast_detrend(x, 100)
    b, a = ss.butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype="bandpass")
    x = ss.filtfilt(b, a, np.double(x))
    return float(_calculate_fft_hr(x, fs=fs, low_pass=0.6, high_pass=3.3))


def classical_pervideo(cache_dir):
    """unsup 캐시(Raw 72²)에서 POS per-video 오차 dict {vidnum: err}"""
    vids = {}
    for f in glob.glob(cache_dir + "/*_input*.npy"):
        m = re.match(r"(.+)_input(\d+)\.npy", os.path.basename(f))
        vids.setdefault(m.group(1), []).append((int(m.group(2)), f))
    out = {}
    for vid, files in sorted(vids.items()):
        x = np.concatenate([np.load(f) for _, f in sorted(files)])[..., :3]
        y = np.concatenate([np.load(f.replace("_input", "_label")) for _, f in sorted(files)])
        gt = canon_hr(y)
        try:
            hp = canon_hr(np.asarray(POS_WANG(x, fs)).ravel())
        except Exception:
            continue
        n = int(re.search(r"(\d+)$", vid).group(1))
        out[n] = abs(hp - gt)
    return out


def model_pervideo(model, axis, suffix=""):
    j = json.load(open(f"{RES}/agg_{model}{suffix}.json"))
    return {r["vid"]: r["err"] for r in j[axis]}


def wilcox(a, b, label):
    """a,b: {vid:err} — 공통 vid 대응표본"""
    ks = sorted(set(a) & set(b))
    x = np.array([a[k] for k in ks]); y = np.array([b[k] for k in ks])
    d = x - y
    if np.allclose(d, 0):
        return dict(label=label, n=len(ks), mae_a=float(x.mean()), mae_b=float(y.mean()), p=1.0, r=0.0)
    try:
        st = stats.wilcoxon(x, y, zero_method="wilcox", method="approx")
        z = abs(stats.norm.ppf(st.pvalue / 2))
        r = float(z / np.sqrt((d != 0).sum()))
        p = float(st.pvalue)
    except ValueError:
        p, r = float("nan"), float("nan")
    return dict(label=label, n=len(ks), mae_a=float(x.mean()), mae_b=float(y.mean()),
                W=float(st.statistic), p=p, r=r)


results = {"tests": [], "trend": {}}

# --- 고전 per-video (신규 4축 + PURE)
pos = {}
for tag, cache in [("DLCN", "unsup_DLCN"), ("NIRP_STILL", "unsup_NIRP_STILL"),
                   ("NIRP_SMALL", "unsup_NIRP_SMALL"), ("NIRP_LARGE", "unsup_NIRP_LARGE"),
                   ("PURE", "real_PURE_unsup")]:
    pos[tag] = classical_pervideo(f"{PF}/cache/{cache}")
    json.dump(pos[tag], open(f"{RES}/pos_pervideo_{tag}.json", "w"))
    print(f"[pos] {tag}: n={len(pos[tag])} MAE={np.mean(list(pos[tag].values())):.2f}")

# 1) 신규 축: CoTAF vs POS / 대동작 CoTA vs CoTAF
for ax in ["DLCN", "NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE"]:
    caf = model_pervideo("CoTAF", ax, "_newsets")
    results["tests"].append(wilcox(pos[ax], caf, f"{ax}: POS vs CoTAF"))
results["tests"].append(wilcox(model_pervideo("CoTA", "NIRP_LARGE", "_newsets"),
                               model_pervideo("CoTAF", "NIRP_LARGE", "_newsets"),
                               "NIRP_LARGE: CoTA vs CoTAF"))

# 2) 주요 축: CoTAF vs CoTA
for ax in ["MCD_B", "MCD_A", "PURE", "REALTEST"]:
    results["tests"].append(wilcox(model_pervideo("CoTA", ax),
                                   model_pervideo("CoTAF", ax), f"{ax}: CoTA vs CoTAF"))

# 3) PURE: POS vs CoTAF
results["tests"].append(wilcox(pos["PURE"], model_pervideo("CoTAF", "PURE"), "PURE: POS vs CoTAF"))

# 4) 순수 제로샷 축 단조 경향 (Friedman + Page): base→CoTNHM→CoTA→CoTAF
order = ["base", "CoTNHM", "CoTA", "CoTAF"]
mats = []
for ax in ["DLCN", "NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE"]:
    cols = [model_pervideo(m, ax, "_newsets") for m in order]
    ks = sorted(set.intersection(*[set(c) for c in cols]))
    mats.append(np.array([[c[k] for c in cols] for k in ks]))
M = np.vstack(mats)  # (n_videos, 4)
fr = stats.friedmanchisquare(*[M[:, i] for i in range(4)])
# Page's L (감소 경향: 순위가 base에서 최대 → CoTAF에서 최소를 기대)
ranks = np.apply_along_axis(stats.rankdata, 1, M)
L = float((ranks * np.arange(1, 5)).sum())
n, k = M.shape
EL = n * k * (k + 1) ** 2 / 4
VL = n * k ** 2 * (k + 1) * (k ** 2 - 1) / 144
zL = (EL - L) / np.sqrt(VL)  # L<E[L] 이면 내림차순 경향(뒤 모델일수록 오차 작음)
results["trend"] = dict(order=order, n_videos=int(n),
                        friedman_chi2=float(fr.statistic), friedman_p=float(fr.pvalue),
                        page_L=L, page_z=float(zL),
                        page_p_onesided=float(stats.norm.sf(zL)),
                        mean_by_model={m: float(M[:, i].mean()) for i, m in enumerate(order)})

json.dump(results, open(f"{RES}/stats_final.json", "w"), indent=1)
for t in results["tests"]:
    print(f"{t['label']:32s} n={t['n']:3d} MAE {t['mae_a']:6.2f} vs {t['mae_b']:6.2f} "
          f"p={t.get('p', float('nan')):.4g} r={t.get('r', float('nan')):.2f}")
tr = results["trend"]
print(f"TREND {tr['order']} n={tr['n_videos']} Friedman p={tr['friedman_p']:.4g} "
      f"Page z={tr['page_z']:.2f} p(1측)={tr['page_p_onesided']:.4g} means={tr['mean_by_model']}")
