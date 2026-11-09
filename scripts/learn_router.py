#!/usr/bin/env python3
"""[체인 45 / 트랙 A] 학습된 라우터 — GT 모션 라벨 없이 관측 특징 + 출력 내재 특징으로 클립별 추정기 선택.
후보(3-way): 기준 TW=CoTAF_TW_L, 하이브리드 HYB4=CoTAM_HYB4_L (각 5시드 풀링 오차), POS.
라벨: 주 = oracle argmin 분류, 보조 = 후보별 오차 회귀→argmin. 분류기: 주 = 로지스틱 회귀, 보조 = 얕은 HistGBM.
분할: held-out 7~33 피험자 분리 5-fold(GroupKFold). 보조: 1~6(학습 분할) 학습 → 7~33 평가.
MMPD 메타 라벨(light/motion/exercise/skin)은 특징으로 절대 미사용 — motion은 층화·GT-모션 라우팅 비교군에만.
입력: results/agg_{TW,HYB4}[_sN]_MMPD.json, agg_POS_MMPD.json, mmpd_obs_features.json, ev_* pickle(출력 특징)
출력: results/router_cv.json
"""
import glob, json, pickle, re, sys, warnings
import numpy as np
import scipy.signal as ss
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
sys.path.insert(0, "/SPECPHYS_ROOT/repos/rPPG-Toolbox")
from agg_model2 import cat_chunks, fast_detrend, PF
from evaluation.post_process import _calculate_fft_hr
warnings.filterwarnings("ignore")

RES = f"{PF}/../results"
SEEDS = ("", "_s200", "_s300", "_s400", "_s500")
FPS = 30
CAND = ["TW", "HYB4", "POS"]
MMPD_RE = re.compile(r"subject(\d+)_L(\d)_MO(\d)_E(\d)_S(\d)_GE(\d)_GL(\d)_H(\d)_MA(\d)")
OBS_F = ["f_disp_mean", "f_disp_max", "f_disp_p90", "f_center_std", "f_size_std", "f_det_rate", "f_face_frac",
         "f_diff_all", "f_diff_face", "f_diff_face_std", "f_lum_mean", "f_lum_std", "f_lum_drift", "f_lum_band_ratio", "f_skin_frac"]
OUT_F = ["snr_tw", "snr_hyb", "f_pos_snr", "hr_tw", "dis_tw_hyb", "dis_tw_pos", "dis_hyb_pos", "hrstd_tw", "hrstd_hyb"]
RNG = 0

def key(r): return (r["subject"], r["light"], r["motion"], r["exercise"], r["skin"])

def wp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 8 or np.allclose(a, b): return float("nan")
    return float(wilcoxon(a, b).pvalue)

def tb_filter(x):
    x = fast_detrend(np.asarray(x, float) - np.mean(x), 100)
    b, a = ss.butter(1, [0.6 / FPS * 2, 3.3 / FPS * 2], btype="bandpass")
    return ss.filtfilt(b, a, x)

def hr_snr(sig):
    """정본 전처리(cumsum→detrend→0.6–3.3) 후 HR + 자기 피크 SNR(toolbox 정의, 기준=자기 피크)."""
    x = tb_filter(np.cumsum(sig))
    hr = float(_calculate_fft_hr(x, fs=FPS, low_pass=0.6, high_pass=3.3))
    N = 1 << int(np.ceil(np.log2(len(x))))
    f, p = ss.periodogram(x, fs=FPS, nfft=N, detrend=False)
    f0, dv = hr / 60, 0.1
    h1 = (f >= f0 - dv) & (f <= f0 + dv); h2 = (f >= 2 * f0 - dv) & (f <= 2 * f0 + dv)
    band = (f >= 0.6) & (f <= 3.3); rest = band & ~h1 & ~h2
    return hr, float(10 * np.log10((p[h1].sum() + p[h2].sum()) / (p[rest].sum() + 1e-12)))

# ---------- 1. 오차(라벨) ----------
def mm(m): return {key(r): r for r in json.load(open(f"{RES}/agg_{m}_MMPD.json"))}
tw = [mm(f"CoTAF_TW_L{s}") for s in SEEDS]; hb = [mm(f"CoTAM_HYB4_L{s}") for s in SEEDS]; pos = mm("POS")
ks = sorted(set.intersection(*[set(d) for d in tw + hb], set(pos)))
E = np.array([[np.mean([d[k]["err"] for d in tw]), np.mean([d[k]["err"] for d in hb]), pos[k]["err"]] for k in ks])
E_seed = {"TW": np.array([[d[k]["err"] for d in tw] for k in ks]), "HYB4": np.array([[d[k]["err"] for d in hb] for k in ks])}
SUBJ = np.array([k[0] for k in ks]); MO = np.array([k[2] for k in ks]); GT = np.array([tw[0][k]["gt"] for k in ks])
HO = SUBJ >= 7
Y = E.argmin(1)
print(f"클립 n={len(ks)} held-out={HO.sum()} oracle 라벨 비중(held-out) TW/HYB4/POS = "
      f"{np.bincount(Y[HO], minlength=3) / HO.sum()}")

# ---------- 2. 출력 내재 특징 (pickle, 시드별 → 평균/std) ----------
def out_feats(model):
    per = {}
    for s in SEEDS:
        for pk in sorted(glob.glob(f"{PF}/runs/ev_{model}{s}_MMPD_B*/*/saved_test_outputs/*.pickle")):
            d = pickle.load(open(pk, "rb"))
            for vid in d["predictions"]:
                m = MMPD_RE.match(str(vid))
                k = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
                per.setdefault(k, []).append(hr_snr(cat_chunks(d["predictions"][vid])))
    return per
print("출력 특징 계산 중 (pickle 60개)...", flush=True)
OT, OH = out_feats("CoTAF_TW_L"), out_feats("CoTAM_HYB4_L")
# 정합성: 시드1 HR이 agg JSON pred와 일치하는지
chk = np.mean([abs(OT[k][0][0] - tw[0][k]["pred"]) for k in ks]); print(f"정합성 |HR(pickle)−HR(agg)| 평균 = {chk:.3f} BPM")

# ---------- 3. 관측 특징 ----------
obs = {key(r): r for r in json.load(open(f"{RES}/mmpd_obs_features.json"))}
missing = [k for k in ks if k not in obs]
if missing: print(f"경고: 관측 특징 누락 {len(missing)}클립 → 제외");
sel = np.array([k in obs for k in ks])
ks = [k for k, s in zip(ks, sel) if s]; E, Y, SUBJ, MO, GT, HO = E[sel], Y[sel], SUBJ[sel], MO[sel], GT[sel], HO[sel]
E_seed = {m: v[sel] for m, v in E_seed.items()}
rows = []
for k in ks:
    o = obs[k]; t = np.array(OT[k]); h = np.array(OH[k])
    f = {c: o[c] for c in OBS_F}
    f.update(f_pos_snr=o["f_pos_snr"], snr_tw=t[:, 1].mean(), snr_hyb=h[:, 1].mean(), hr_tw=t[:, 0].mean(),
             dis_tw_hyb=abs(t[:, 0].mean() - h[:, 0].mean()), dis_tw_pos=abs(t[:, 0].mean() - o["f_pos_hr"]),
             dis_hyb_pos=abs(h[:, 0].mean() - o["f_pos_hr"]), hrstd_tw=t[:, 0].std(), hrstd_hyb=h[:, 0].std())
    rows.append(f)
FEATS = OBS_F + OUT_F
X = np.array([[r[c] for c in FEATS] for r in rows], float)
# POS 정합성(관측 파일의 POS HR vs 정본 POS JSON)
chk2 = np.mean([abs(obs[k]["f_pos_hr"] - pos[k]["pred"]) for k in ks]); print(f"정합성 |POS HR(재계산)−POS HR(정본)| 평균 = {chk2:.3f} BPM")

# ---------- 4. 평가 유틸 ----------
def composite(choice, mask):
    return E[np.arange(len(E)), choice][mask]
def summarize(name, choice, mask=HO, ref=None):
    c = composite(choice, mask); base = {n: E[mask, i] for i, n in enumerate(CAND)}
    gtr = np.where(MO > 1, 1, 0); g = composite(gtr, mask); orc = E[mask].min(1)
    mo = MO[mask]
    out = dict(name=name, n=int(mask.sum()), MAE=float(c.mean()),
               still=float(c[mo == 1].mean()), motion=float(c[mo > 1].mean()), walking=float(c[mo == 4].mean()),
               p_vs_TW=wp(base["TW"], c), p_vs_HYB4=wp(base["HYB4"], c), p_vs_POS=wp(base["POS"], c), p_vs_GTroute=wp(g, c),
               p_motion_vs_HYB4=wp(base["HYB4"][mo > 1], c[mo > 1]), p_motion_vs_TW=wp(base["TW"][mo > 1], c[mo > 1]),
               regret_vs_oracle=float((c - orc).mean()), acc_vs_oracle=float((choice[mask] == Y[mask]).mean()),
               share=[float((choice[mask] == i).mean()) for i in range(3)])
    print(f"{name:34s} MAE {out['MAE']:6.2f} | 정지 {out['still']:5.2f} 움직임 {out['motion']:5.2f} Walking {out['walking']:5.2f} | "
          f"p vsTW {out['p_vs_TW']:.4f} vsHYB4 {out['p_vs_HYB4']:.4f} vsGT라우팅 {out['p_vs_GTroute']:.4f} | "
          f"regret {out['regret_vs_oracle']:.2f} acc {out['acc_vs_oracle']:.2f} share {np.round(out['share'], 2)}", flush=True)
    return out

results = dict(n_total=int(len(ks)), n_heldout=int(HO.sum()), features=FEATS, baselines=[], routers=[], aux=[])
print("\n=== 기준선 (held-out 7~33, 5시드 풀링 오차) ===")
for i, n in enumerate(CAND):
    results["baselines"].append(summarize(f"항상-{n}", np.full(len(E), i)))
results["baselines"].append(summarize("GT-모션 라우팅(정지=TW,움직임=HYB4)", np.where(MO > 1, 1, 0)))
results["baselines"].append(summarize("oracle 3-way(TW/HYB4/POS)", E.argmin(1)))
results["baselines"].append(summarize("oracle 2-way(TW/HYB4)", E[:, :2].argmin(1)))
# 보조: PhysMamba 하이브리드(3시드) 포함 4-way oracle
try:
    pm = [mm(f"PM_HYB4_L{s}") for s in ("", "_s200", "_s300")]
    EP = np.array([np.mean([d[k]["err"] for d in pm]) for k in ks])
    E4 = np.column_stack([E, EP]); o4 = E4.min(1)[HO]
    results["aux"].append(dict(name="oracle 4-way(+PM_HYB4 3시드)", MAE=float(o4.mean()), PM_alone=float(EP[HO].mean())))
    print(f"[보조] oracle 4-way(+PhysMamba 하이브리드) {o4.mean():.2f} (PM 단독 {EP[HO].mean():.2f})")
except FileNotFoundError:
    pass

# ---------- 5. 라우터 CV ----------
def make_model(kind, task):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)) if task == "cls" \
            else make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, class_weight="balanced", random_state=RNG) if task == "cls" \
        else HistGradientBoostingRegressor(max_depth=3, max_iter=100, learning_rate=0.05, random_state=RNG)

def fit_predict(Xtr, Etr, Xte, task, kind, ncand):
    """반환: 선택 인덱스(0..ncand-1)"""
    if task == "cls":
        ytr = Etr[:, :ncand].argmin(1)
        if len(np.unique(ytr)) < 2: return np.full(len(Xte), int(np.bincount(ytr).argmax()))
        m = make_model(kind, "cls").fit(Xtr, ytr); pr = m.predict(Xte)
        return pr
    preds = np.column_stack([make_model(kind, "reg").fit(Xtr, Etr[:, j]).predict(Xte) for j in range(ncand)])
    return preds.argmin(1)

def cv_router(task, kind, fcols, ncand, name):
    cols = [FEATS.index(c) for c in fcols]
    Xh, Eh, Sh = X[HO][:, cols], E[HO], SUBJ[HO]
    choice = np.zeros(len(Eh), int)
    for tr, te in GroupKFold(5).split(Xh, groups=Sh):
        choice[te] = fit_predict(Xh[tr], Eh[tr], Xh[te], task, kind, ncand)
    full = np.zeros(len(E), int); full[HO] = choice
    out = summarize(name, full); out.update(task=task, kind=kind, fset=fcols if len(fcols) < len(FEATS) else "all", ncand=ncand,
                                            choice=full[HO].tolist())
    return out

print("\n=== 라우터 (held-out 7~33 피험자 분리 5-fold CV) ===")
FSETS = {"all": FEATS, "obs": OBS_F, "out": OUT_F}
for task in ("cls", "reg"):
    for kind in ("lr", "gbm"):
        for fs, fcols in FSETS.items():
            for ncand in (3, 2):
                tag = "[주] " if (task, kind, fs, ncand) == ("cls", "lr", "all", 3) else ""
                results["routers"].append(cv_router(task, kind, fcols, ncand, f"{tag}{task}-{kind}-{fs}-{ncand}way"))

# ---------- 6. 보조: 학습 분할(1~6) 학습 → held-out 평가 (출력 특징 낙관 편향 명시) ----------
print("\n=== 보조: MMPD 1~6(학습 분할, HYB4 학습 데이터) 학습 → 7~33 평가 ===")
TR = ~HO
for kind in ("lr", "gbm"):
    for task in ("cls", "reg"):
        ch = fit_predict(X[TR], E[TR], X[HO], task, kind, 3)
        full = np.zeros(len(E), int); full[HO] = ch
        o = summarize(f"train1-6→7-33 {task}-{kind}-all-3way", full); o.update(task=task, kind=kind, split="train16")
        results["aux"].append(o)

# ---------- 7. 보조: 관측 모션 검출기 (GT 모션은 검출기 학습 *목표*로만; 특징은 관측만) → 정지=TW/움직임=HYB4 ----------
print("\n=== 보조: 관측 모션 검출기(얼굴 추적 특징 → 움직임 여부) → 조건부 라우팅 ===")
cols = [FEATS.index(c) for c in OBS_F]
Xh, Sh = X[HO][:, cols], SUBJ[HO]; ymo = (MO[HO] > 1).astype(int)
prob = np.zeros(len(ymo))
for tr, te in GroupKFold(5).split(Xh, groups=Sh):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)).fit(Xh[tr], ymo[tr])
    prob[te] = m.predict_proba(Xh[te])[:, 1]
auc = float(roc_auc_score(ymo, prob)); det = (prob > 0.5).astype(int)
full = np.zeros(len(E), int); full[HO] = det
o = summarize("관측 모션 검출기 라우팅(정지=TW,움직임=HYB4)", full)
o.update(detector_auc=auc, detector_acc=float((det == ymo).mean())); results["aux"].append(o)
print(f"   모션 검출 AUC={auc:.3f} acc={o['detector_acc']:.3f}")
# 단일 특징 임계(배치 가장 단순): 얼굴 상자 변위 평균의 CV 임계
d = X[HO][:, FEATS.index("f_disp_mean")]
det1 = np.zeros(len(ymo), int)
for tr, te in GroupKFold(5).split(Xh, groups=Sh):
    ths = np.quantile(d[tr], np.linspace(0.05, 0.95, 37)); best = max(ths, key=lambda t: ((d[tr] > t) == ymo[tr]).mean())
    det1[te] = (d[te] > best).astype(int)
full = np.zeros(len(E), int); full[HO] = det1
o = summarize("단일 특징(얼굴 변위) 임계 라우팅", full); o.update(detector_acc=float((det1 == ymo).mean())); results["aux"].append(o)

# ---------- 8. 주 라우터 순열 중요도 ----------
print("\n=== 주 라우터(cls-lr-all-3way) 순열 중요도 (test fold 셔플 시 합성 MAE 증가, 5-fold 평균) ===")
Xh, Eh, Sh = X[HO], E[HO], SUBJ[HO]
imp = np.zeros(len(FEATS)); rng = np.random.default_rng(RNG)
for tr, te in GroupKFold(5).split(Xh, groups=Sh):
    ytr = Eh[tr].argmin(1); m = make_model("lr", "cls").fit(Xh[tr], ytr)
    base = Eh[te][np.arange(len(te)), m.predict(Xh[te])].mean()
    for j in range(len(FEATS)):
        acc = 0
        for _ in range(5):
            Xp = Xh[te].copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            acc += Eh[te][np.arange(len(te)), m.predict(Xp)].mean() - base
        imp[j] += acc / 5
imp /= 5
order = np.argsort(-imp)
for j in order[:12]: print(f"   {FEATS[j]:18s} +{imp[j]:.3f}")
results["perm_importance"] = {FEATS[j]: float(imp[j]) for j in range(len(FEATS))}
# 특징-오차 단순 상관(스피어만, held-out)
from scipy.stats import spearmanr
results["spearman_feat_vs_err"] = {c: {n: float(spearmanr(X[HO][:, i], E[HO][:, j]).correlation) for j, n in enumerate(CAND)}
                                   for i, c in enumerate(FEATS)}
results["spearman_feat_vs_motion"] = {c: float(spearmanr(X[HO][:, i], MO[HO]).correlation) for i, c in enumerate(FEATS)}

# ---------- 9. 사전 등록 판정 ----------
prim = next(r for r in results["routers"] if r["name"].startswith("[주]"))
hyb = next(b for b in results["baselines"] if b["name"] == "항상-HYB4"); gtr = next(b for b in results["baselines"] if "GT-모션" in b["name"])
if prim["MAE"] < hyb["MAE"] and prim["p_vs_HYB4"] < 0.05 and prim["MAE"] <= gtr["MAE"]:
    verdict = "자기 라우팅 성공"
elif not (prim["p_vs_HYB4"] < 0.05 and prim["MAE"] > hyb["MAE"]) and not (prim["p_vs_GTroute"] < 0.05 and prim["MAE"] > gtr["MAE"]):
    verdict = "모션 라벨 없이 동등(부분 성공)"
else:
    verdict = "부정 결과"
best = min(results["routers"], key=lambda r: r["MAE"])
results["verdict"] = dict(primary=prim["name"], primary_MAE=prim["MAE"], verdict=verdict, best_variant=best["name"], best_MAE=best["MAE"],
                          best_p_vs_HYB4=best["p_vs_HYB4"], note="best_variant는 사후 선택(다중 비교) — 판정은 주 라우터로만")
print(f"\n판정(사전 등록 기준, 주 라우터): {verdict} | 주 {prim['MAE']:.2f} vs HYB4 {hyb['MAE']:.2f} (p={prim['p_vs_HYB4']:.4f}) "
      f"vs GT라우팅 {gtr['MAE']:.2f} (p={prim['p_vs_GTroute']:.4f}) | 사후 최선 변형 {best['name']} {best['MAE']:.2f} (p vs HYB4 {best['p_vs_HYB4']:.4f})")
json.dump(results, open(f"{RES}/router_cv.json", "w"), indent=1, ensure_ascii=False)
print("ROUTER_DONE")
