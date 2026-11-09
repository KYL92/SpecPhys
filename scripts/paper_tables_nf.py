#!/usr/bin/env python3
"""[56 문서] 논문(FactorizePhys 양식) 결과 표 생성 — 정본 agg_*.json 만 읽어 영상 단위 지표(MAE±SE, RMSE, MAPE, Pearson r)를 계산한다.
  python scripts/paper_tables_nf.py  → results/paper_tables_nf.md (마크다운 표 전부) + results/paper_tables_nf.json (그림·PPT 용 숫자)
규칙(CLAUDE.md): zero-shot 과 in-domain 을 한 표에 섞지 않음; VIPL 은 문헌 GT(gt_lit); MMPD 는 held-out 피험자 7–33; PURE 는 held-out 피험자 07–10(vid ≥ 701);
  다중 시드는 시드별 지표의 평균(MAE 의 ±SE 는 영상별 시드 평균 오차의 표준오차); 공개 가중치 = rPPG-Toolbox UBFC-rPPG 체크포인트(학습 데이터 불일치 각주)."""
import glob, json, os, re
import numpy as np

R = "/SPECPHYS_ROOT/_workspace/data/results"
OUT_MD = f"{R}/paper_tables_nf.md"; OUT_JS = f"{R}/paper_tables_nf.json"
L = []
def emit(s=""): L.append(s)


SEED_LOCK = {"NF_SKREF_REP_L": [600, 700, 800, 900, 1000], "NF_SKREF_X15_L": [600, 700, 800, 900, 1000], "NF_ENS_AB_L": [600, 700, 800, 900, 1000], "NF_NSP2_L": [1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]}   # [62 §14] 최종 설정은 선택에 쓰지 않은 확인 시드, 기존 설정은 보고한 시드 그대로


def seed_files(m, suffix=""):
    if m in SEED_LOCK: return [f for f in (f"{R}/agg_{m}_s{x}{suffix}.json" for x in SEED_LOCK[m]) if os.path.exists(f)]
    pat = re.compile(rf"agg_{re.escape(m)}_s\d+{re.escape(suffix)}\.json$")
    fs = [f"{R}/agg_{m}{suffix}.json"] + sorted(x for x in glob.glob(f"{R}/agg_{m}_s[0-9]*{suffix}.json") if pat.search(os.path.basename(x)))
    return [f for f in fs if os.path.exists(f)]


def rows_axis(f, ax):
    """(vid, gt, pred, meta) 행 — VIPL meta = scenario, MMPD meta = motion"""
    d = json.load(open(f))
    if ax == "VIPL":
        return [(r["vid"], float(r["gt_lit"]), float(r["pred"]), r.get("scenario")) for r in d if r.get("gt_lit") is not None]
    if ax == "MMPD":
        return [((r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]), float(r["gt"]), float(r["pred"]), r["motion"]) for r in d if 7 <= r["subject"] <= 33]
    if ax == "MMPD_WALK":
        return [((r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]), float(r["gt"]), float(r["pred"]), r["motion"]) for r in d if 7 <= r["subject"] <= 33 and r["motion"] == 4]
    if ax == "MMPD_LIT":   # [58 문서, 체인 74] MMPD 문헌 분할 테스트 피험자 27–33 (140 영상; in-domain 셀·zero-shot 참고 모두 같은 영상)
        return [((r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]), float(r["gt"]), float(r["pred"]), r["motion"]) for r in d if 27 <= r["subject"] <= 33]
    if ax == "MMPD_MOTION":
        return [((r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]), float(r["gt"]), float(r["pred"]), r["motion"]) for r in d if 7 <= r["subject"] <= 33 and r["motion"] != 1]
    if isinstance(d, list):   # POS: agg_POS_NIRP.json(axis 필드) / pos_pervideo_{AX}.json
        rows = [r for r in d if r.get("axis", ax) == ax and "gt" in r and "pred" in r]
        return [(r["vid"], float(r["gt"]), float(r["pred"]), None) for r in rows]
    rows = d.get(ax if ax != "PURE_HO" else "PURE") or []
    if ax == "PURE_HO": rows = [r for r in rows if int(r["vid"]) >= 701]
    return [(r["vid"], float(r["gt"]), float(r["pred"]), None) for r in rows]


VIPL_ALIAS = {"FactorizePhys": "FPbase", "RhythmFormer": "RFbase", "PhysFormer": "PFbase", "PFbase": "PFbase"}
def files_for(m, ax):
    if ax == "VIPL": return seed_files(VIPL_ALIAS.get(m, m), "_VIPL")
    if ax.startswith("MMPD"): return seed_files(m, "_MMPD")
    if m == "POS" and ax.startswith("NIRP"): return [f"{R}/agg_POS_NIRP.json"] if os.path.exists(f"{R}/agg_POS_NIRP.json") else []
    if m == "POS" and os.path.exists(f"{R}/pos_pervideo_{ax}.json"): return [f"{R}/pos_pervideo_{ax}.json"]
    return seed_files(m, "")


def metrics(m, ax, sel=None):
    """시드별 지표 평균 + MAE 의 영상 단위 SE(시드 평균 오차). sel: (vid, gt, pred) -> bool"""
    fs = files_for(m, ax)
    if not fs: return None
    per_seed = []; per_vid = {}
    for f in fs:
        rows = rows_axis(f, ax)
        if sel: rows = [r for r in rows if sel(r)]
        if len(rows) < 3: continue
        gt = np.array([r[1] for r in rows]); pr = np.array([r[2] for r in rows]); e = np.abs(pr - gt)
        per_seed.append(dict(mae=e.mean(), rmse=np.sqrt(((pr - gt) ** 2).mean()), mape=(e / gt * 100).mean(), r=(np.corrcoef(pr, gt)[0, 1] if np.std(pr) > 0 and np.std(gt) > 0 else float("nan")), n=len(rows)))
        for v, g, p, _ in rows: per_vid.setdefault(v, []).append(abs(p - g))
    if not per_seed: return None
    ev = np.array([np.mean(x) for x in per_vid.values()])
    out = {k: float(np.mean([s[k] for s in per_seed])) for k in ("mae", "rmse", "mape", "r")}
    out.update(n=int(per_seed[0]["n"]), seeds=len(per_seed), se=float(ev.std(ddof=1) / np.sqrt(len(ev))) if len(ev) > 1 else 0.0, mae_seeds=[round(float(s["mae"]), 2) for s in per_seed])
    return out


def fmt(mt, best=False):
    if mt is None: return "–"
    s = f"{mt['mae']:.2f} ± {mt['se']:.2f}"
    return f"**{s}**" if best else s


# ----- 모델 정의(표시 이름, 저장소 이름, 학습 데이터, 목적함수/주목 모듈) -----
PUBLIC = "UBFC-rPPG (public weights)"
MODELS = [
    ("POS [classical]", "POS", "–", "chrominance projection", "classical"),
    ("CHROM [classical]", "CHROM", "–", "chrominance projection", "classical"),
    ("PhysNet", "PhysNet_arch", PUBLIC, "3D-CNN, no attention", "public"),
    ("EfficientPhys", "EfficientPhys_arch", PUBLIC, "2D-CNN, SASN", "public"),
    ("PhysFormer", "PFbase", PUBLIC, "TD-MHSA", "public"),
    ("RhythmFormer", "RhythmFormer", PUBLIC, "hierarchical TD-attention", "public"),
    ("FactorizePhys", "FactorizePhys", PUBLIC, "FSAM (rank-1 NMF)", "public"),
    ("PhysFormer + PPG-injected corpus", "CoTAF_TW_L", "UBFC ckpt → corpus (203 clips)", "TD-MHSA; Neg-Pearson + HR", "ours"),
    ("FactorizePhys + hybrid corpus, decomposition loss", "FP_OFF_HYB", "UBFC ckpt → hybrid pool (366)", "FSAM rank-1 + noise head", "ours"),
    ("SpecPhys-B (ours): FactorizePhys + hybrid corpus + spectral guidance", "NF_HYB_L", "UBFC ckpt → hybrid pool (366)", "NF-FSAM rank-2 + spectral losses", "ours"),
    ("SpecPhys-L (ours): full training recipe, no projection", "NF_SKREF_REP_L", "UBFC ckpt → hybrid pool (366)", "SpecPhys-B + doubling penalty, double-speed augmentation, skin weighting, nuisance-reference exclusion", "ours"),
    ("SpecPhys-H (ours): high-rate member", "NF_SKREF_X15_L", "UBFC ckpt → hybrid pool (366)", "SpecPhys-L + 1.5x replay of real MMPD training clips", "ours"),
    ("SpecPhys-R (ours): chunk routing of SpecPhys-L and SpecPhys-H", "NF_ENS_AB_L", "UBFC ckpt → hybrid pool (366)", "inference-time routing (70 / 105 bpm)", "ours"),
    ("SpecPhys (ours, final): SpecPhys-L + noise-subspace projection", "NF_NSP2_L", "UBFC ckpt → hybrid pool (366)", "parameter-free projection of the clip's noise references + spectral losses", "ours"),
]
ABL = [
    ("SpecPhys-B, full (rank-2, nuisance supervision)", "NF_HYB_L"),
    ("– nuisance supervision & decorrelation (NF_SUP=0)", "NF_HYB_NOSUP_L"),
    ("– rank-2 (rank-1 temporal layout)", "NF_HYB_R1_L"),
    ("original FSAM layout + spectral losses only", "NF_HYB_ORIG_L"),
    ("rank-1 + decomposition loss, no spectral losses (FP_OFF_HYB)", "FP_OFF_HYB"),
    ("public FactorizePhys (no corpus, no spectral losses)", "FactorizePhys"),
]
SS = [
    ("SpecPhys-B baseline", "NF_HYB_L"),
    ("+ global-trace self-supervision", "NF_SS_TR_L"),
    ("+ attention spatial regularization", "NF_SS_AT_L"),
    ("+ both", "NF_SS_BOTH_L"),
    ("+ local + global nuisance proxy", "NF_SS_LOC_L"),
    ("+ augmentation consistency", "NF_SS_AUG_L"),
    ("+ augmentation consistency + global trace", "NF_SS_AUGTR_L"),
    ("readout-consistent chunk-HR label (patch 40)", "NF_HYB_TB_L"),
]
AXES = [("VIPL-HR (zero-shot, n=2,378, literature GT)", "VIPL"), ("MMPD held-out subjects 7–33 (n=540)", "MMPD"), ("PURE held-out subjects 07–10 (n=24)", "PURE_HO"),
        ("UBFC-rPPG held-out subjects 42–49 (n=8)", "REALTEST"), ("MR-NIRP RGB still (n=19)", "NIRP_STILL"), ("MR-NIRP RGB small motion (n=19)", "NIRP_SMALL"), ("MR-NIRP RGB large motion (n=19)", "NIRP_LARGE"),
        ("DLCN (n=20)", "DLCN"), ("MCD-rPPG set A (n=20)", "MCD_A"), ("MCD-rPPG set B (n=20)", "MCD_B"), ("controlled flicker 1.1 Hz + drift (RTNI, n=8)", "RTNI"), ("controlled flicker + motion + sensor noise (RTNC, n=8)", "RTNC")]

JS = {"cross": {}, "vipl_bands": {}, "mmpd_motion": {}, "ablation": {}, "selfsup": {}, "indomain": {}, "phase0": {}}
# ===================== Table 2 style: cross-dataset =====================
emit("# Paper tables — spectrum-guided FactorizePhys on the PPG-injected corpus (generated by scripts/paper_tables_nf.py; source: results/agg_*.json)\n")
emit("Conventions: HR metrics per video (MAE ± standard error over videos, RMSE, MAPE %, Pearson r across videos); multi-seed models report the mean over seeds (seed count in parentheses). Zero-shot and in-domain tables are never mixed. Public weights are the rPPG-Toolbox UBFC-rPPG checkpoints (training data differs from ours; see footnote). VIPL-HR uses the device HR ground truth (gt_HR.csv) as in the literature; PURE is restricted to the held-out subjects 07–10 and MMPD to held-out subjects 7–33.\n")
emit("## Table 2. Cross-dataset (zero-shot) performance evaluation for HR estimation\n")
for title, ax in AXES:
    emit(f"### {title}\n")
    emit("| Model | Training data | Attention / objective | MAE ± SE ↓ | RMSE ↓ | MAPE (%) ↓ | Corr ↑ | seeds |"); emit("|---|---|---|---|---|---|---|---|")
    res = [(name, metrics(key, ax), tr, att, kind) for name, key, tr, att, kind in MODELS]
    best = min((mt["mae"] for _, mt, *_ in res if mt), default=None)
    for name, mt, tr, att, kind in res:
        if mt is None: emit(f"| {name} | {tr} | {att} | – | – | – | – | – |"); continue
        b = best is not None and abs(mt["mae"] - best) < 1e-9
        rstr = "–" if np.isnan(mt["r"]) else f"{mt['r']:.2f}"
        emit(f"| {name} | {tr} | {att} | {fmt(mt, b)} | {mt['rmse']:.2f} | {mt['mape']:.1f} | {rstr} | {mt['seeds']} |")
        JS["cross"].setdefault(ax, {})[name] = mt
    emit("")
# ===================== VIPL bands / scenarios =====================
emit("## Table 3. VIPL-HR by heart-rate band and scenario (MAE, literature GT; zero-shot)\n")
BANDS = [("<60 bpm", lambda r: r[1] < 60), ("60–90", lambda r: 60 <= r[1] < 90), ("90–120", lambda r: 90 <= r[1] < 120), ("≥120", lambda r: r[1] >= 120)]
def vipl_sel_scn(scn):
    return lambda r: r[3] == scn
SCN = [("v1 stable", vipl_sel_scn("v1")), ("v2 large motion", vipl_sel_scn("v2")), ("v7 post-exercise", vipl_sel_scn("v7")), ("v9 phone, motion", vipl_sel_scn("v9"))]
VM = [("POS", "POS"), ("PhysFormer (public)", "PFbase"), ("RhythmFormer (public)", "RFbase"), ("FactorizePhys (public)", "FPbase"), ("PhysFormer + corpus", "CoTAF_TW_L"), ("SpecPhys-B (ours)", "NF_HYB_L"), ("SpecPhys-L (no projection)", "NF_SKREF_REP_L"), ("SpecPhys-H (high-rate member)", "NF_SKREF_X15_L"), ("SpecPhys-R (routed pair)", "NF_ENS_AB_L"), ("SpecPhys (ours, final)", "NF_NSP2_L")]
emit("| Model | all | " + " | ".join(b for b, _ in BANDS) + " | " + " | ".join(s for s, _ in SCN) + " |"); emit("|---|---|" + "---|" * (len(BANDS) + len(SCN)))
for name, key in VM:
    cells = []
    for lab, sel in [("all", None)] + BANDS + SCN:
        mt = metrics(key, "VIPL", sel); cells.append("–" if mt is None else f"{mt['mae']:.2f}"); JS["vipl_bands"].setdefault(name, {})[lab] = None if mt is None else mt["mae"]
    emit(f"| {name} | " + " | ".join(cells) + " |")
emit("\nn per band: " + ", ".join(f"{lab} {metrics('NF_HYB_L', 'VIPL', sel)['n']}" for lab, sel in BANDS) + "; per scenario: " + ", ".join(f"{lab} {metrics('NF_HYB_L', 'VIPL', sel)['n']}" for lab, sel in SCN) + "\n")
# ===================== MMPD by motion =====================
emit("## Table 4. MMPD held-out subjects 7–33 by motion label (MAE; zero-shot)\n")
MOT = [("stationary", 1), ("rotation", 2), ("talking", 3), ("walking", 4)]
def mmpd_motion_sel(mv):
    return lambda r: r[3] == mv
MM = [("POS", "POS"), ("CHROM", "CHROM"), ("RhythmFormer (public)", "RhythmFormer"), ("FactorizePhys (public)", "FactorizePhys"), ("PhysFormer + corpus", "CoTAF_TW_L"), ("FactorizePhys + corpus, decomposition", "FP_OFF_HYB"), ("SpecPhys-B (ours)", "NF_HYB_L"), ("SpecPhys-L (no projection)", "NF_SKREF_REP_L"), ("SpecPhys-H (high-rate member)", "NF_SKREF_X15_L"), ("SpecPhys-R (routed pair)", "NF_ENS_AB_L"), ("SpecPhys (ours, final)", "NF_NSP2_L")]
emit("| Model | all (540) | " + " | ".join(f"{m} ({metrics('NF_HYB_L', 'MMPD', mmpd_motion_sel(v))['n']})" for m, v in MOT) + " |"); emit("|---|---|" + "---|" * len(MOT))
for name, key in MM:
    cells = []
    for lab, sel in [("all", None)] + [(m, mmpd_motion_sel(v)) for m, v in MOT]:
        mt = metrics(key, "MMPD", sel); cells.append("–" if mt is None else f"{mt['mae']:.2f}"); JS["mmpd_motion"].setdefault(name, {})[lab] = None if mt is None else mt["mae"]
    emit(f"| {name} | " + " | ".join(cells) + " |")
emit("")
# ===================== Ablation =====================
emit("## Table 5. Ablation of the factorization structure and the objective (zero-shot; MAE)\n")
ABL_AX = [("VIPL-HR", "VIPL"), ("MMPD 7–33", "MMPD"), ("MMPD walking", "MMPD_WALK"), ("MR-NIRP small", "NIRP_SMALL"), ("MR-NIRP large", "NIRP_LARGE"), ("RTNI flicker", "RTNI"), ("PURE 07–10", "PURE_HO")]
emit("| Variant | seeds | " + " | ".join(a for a, _ in ABL_AX) + " |"); emit("|---|---|" + "---|" * len(ABL_AX))
for name, key in ABL:
    cells = []; ns = None
    for lab, ax in ABL_AX:
        mt = metrics(key, ax); cells.append("–" if mt is None else f"{mt['mae']:.2f}"); ns = ns or (mt and mt["seeds"]); JS["ablation"].setdefault(name, {})[lab] = None if mt is None else mt["mae"]
    emit(f"| {name} | {ns or '–'} | " + " | ".join(cells) + " |")
emit("")
# ===================== Self-supervised separation signals =====================
emit("## Table 6. Self-supervised separation signals on real clips without nuisance ground truth (zero-shot; MAE; all cells 3 seeds)\n")
SS_AX = [("VIPL all", "VIPL", None), ("VIPL v7 post-exercise", "VIPL", vipl_sel_scn("v7")), ("VIPL ≥120", "VIPL", lambda r: r[1] >= 120), ("MMPD all", "MMPD", None), ("MMPD walking", "MMPD", mmpd_motion_sel(4)), ("MR-NIRP large", "NIRP_LARGE", None), ("MR-NIRP small", "NIRP_SMALL", None), ("RTNI", "RTNI", None)]
emit("| Cell | " + " | ".join(a for a, _, _ in SS_AX) + " |"); emit("|---|" + "---|" * len(SS_AX))
for name, key in SS:
    cells = []
    for lab, ax, sel in SS_AX:
        mt = metrics(key, ax, sel); cells.append("–" if mt is None else f"{mt['mae']:.2f}"); JS["selfsup"].setdefault(name, {})[lab] = None if mt is None else mt["mae"]
    emit(f"| {name} | " + " | ".join(cells) + " |")
emit("")
# ===================== In-domain =====================
emit("## Table 7. In-domain two-stage adaptation (labels of the target dataset used; subject-exclusive 5-fold; never compared in the same column as zero-shot)\n")
def vipl_cv(tags, sel=None):
    acc = {}; meta = {}
    for t in tags:
        f = f"{R}/agg_CV_{t}_VIPL.json"
        if not os.path.exists(f): continue
        for r in json.load(open(f)):
            if r.get("gt_lit") is None: continue
            acc.setdefault(r["vid"], []).append((r["gt_lit"], r["pred"])); meta[r["vid"]] = r.get("scenario")
    if not acc: return None
    return _cv_metrics(acc, meta, sel, sum(os.path.exists(f"{R}/agg_CV_{t}_VIPL.json") for t in tags))


def _cv_metrics(acc, meta, sel, nseeds):
    """acc: vid -> [(gt, pred) per seed]; 시드별 지표의 평균(앙상블 금지) + 영상별 시드 평균 오차의 SE. sel(vid_row) with row=(vid, gt, pred, meta)"""
    ns = max(len(x) for x in acc.values()); per_seed = []; per_vid = {}
    for k in range(ns):
        rows = [(v, x[k][0], x[k][1], meta.get(v)) for v, x in acc.items() if len(x) > k]
        if sel: rows = [r for r in rows if sel(r)]
        if len(rows) < 3: continue
        gt = np.array([r[1] for r in rows]); pr = np.array([r[2] for r in rows]); e = np.abs(pr - gt)
        per_seed.append(dict(mae=e.mean(), rmse=np.sqrt(((pr - gt) ** 2).mean()), mape=(e / gt * 100).mean(), r=np.corrcoef(pr, gt)[0, 1], n=len(rows)))
        for v, g, p, _ in rows: per_vid.setdefault(v, []).append(abs(p - g))
    if not per_seed: return None
    ev = np.array([np.mean(x) for x in per_vid.values()])
    out = {k: float(np.mean([s[k] for s in per_seed])) for k in ("mae", "rmse", "mape", "r")}
    out.update(n=int(per_seed[0]["n"]), seeds=nseeds, se=float(ev.std(ddof=1) / np.sqrt(len(ev))), mae_seeds=[round(float(s["mae"]), 2) for s in per_seed]); return out
VIPL_IN = [("VIPL-only, augmentation + validation (CV_BASEAV_L)", ["BASEAV_L"]), ("corpus PhysFormer, device-label co-training (CV_TWg_L)", ["TWg_L"]), ("SpecPhys-B 2-stage: corpus checkpoint → fold adaptation (ours)", ["NF_L", "NFs2_L", "NFs3_L"]), ("SpecPhys-B 2-stage + global-trace self-supervision (seed 1)", ["NFSS_L"])]
emit("### VIPL-HR subject-exclusive 5-fold (n=2,378, literature GT)\n"); emit("| Setting | seeds | MAE ± SE | RMSE | MAPE | Corr | v7 post-exercise | ≥120 bpm |"); emit("|---|---|---|---|---|---|---|---|")
for name, tags in VIPL_IN:
    mt = vipl_cv(tags)
    if mt is None: emit(f"| {name} | – | – | – | – | – | – | – |"); continue
    v7 = vipl_cv(tags, vipl_sel_scn("v7")); hi = vipl_cv(tags, lambda r: r[1] >= 120)
    emit(f"| {name} | {mt['seeds']} | {mt['mae']:.2f} ± {mt['se']:.2f} | {mt['rmse']:.2f} | {mt['mape']:.1f} | {mt['r']:.2f} | {v7['mae']:.2f} | {hi['mae']:.2f} |"); JS["indomain"][f"VIPL|{name}"] = dict(mt, v7=v7["mae"], hi=hi["mae"])
zs = metrics("NF_HYB_L", "VIPL"); emit(f"| (reference) SpecPhys-B zero-shot | 3 | {zs['mae']:.2f} ± {zs['se']:.2f} | {zs['rmse']:.2f} | {zs['mape']:.1f} | {zs['r']:.2f} | {metrics('NF_HYB_L', 'VIPL', vipl_sel_scn('v7'))['mae']:.2f} | {metrics('NF_HYB_L', 'VIPL', lambda r: r[1] >= 120)['mae']:.2f} |\n")
def mmpd_cv(tags, sel=None, subs=range(1, 34)):
    acc = {}
    for t in tags:
        f = f"{R}/agg_MCV5_{t}_MMPD.json"
        if not os.path.exists(f): continue
        for r in json.load(open(f)):
            if r["subject"] in subs: acc.setdefault(r["vid"], []).append((r["gt"], r["pred"], r["motion"]))
    if not acc: return None
    meta = {v: x[0][2] for v, x in acc.items()}; acc2 = {v: [(g, p) for g, p, _ in x] for v, x in acc.items()}
    return _cv_metrics(acc2, meta, sel, sum(os.path.exists(f"{R}/agg_MCV5_{t}_MMPD.json") for t in tags))
MMPD_IN = [("MMPD-only (MCV5_BASE_L)", ["BASE_L", "BASE_L_s200", "BASE_L_s300"]), ("PhysFormer joint: corpus + MMPD folds (MCV5_HYB_L)", ["HYB_L", "HYB_L_s200", "HYB_L_s300"]), ("PhysFormer 2-stage: corpus checkpoint → fold (MCV5_PT_L)", ["PT_L"]), ("SpecPhys-B 2-stage: corpus checkpoint → fold (ours, readout-consistent labels)", ["NF_L", "NFs2_L", "NFs3_L"]), ("SpecPhys-L 2-stage: fold adaptation, no projection", ["SKA_L_s600", "SKA_L_s700", "SKA_L_s800"]), ("SpecPhys-H 2-stage: fold adaptation of the high-rate member", ["SKB_L_s600", "SKB_L_s700", "SKB_L_s800"]), ("SpecPhys-R 2-stage: routing of the two adapted members (routed pair)", ["SKC_L_s600", "SKC_L_s700", "SKC_L_s800"]), ("SpecPhys 2-stage: fold adaptation (ours, final)", ["SKN_L_s600", "SKN_L_s700", "SKN_L_s800"])]
emit("### MMPD subject-exclusive 5-fold (660 clips, all 33 subjects held out once)\n"); emit("| Setting | seeds | MAE ± SE (660) | RMSE | walking (132) | held-out 7–33 all (540) | held-out 7–33 walking (108) |"); emit("|---|---|---|---|---|---|---|")
for name, tags in MMPD_IN:
    mt = mmpd_cv(tags)
    if mt is None: emit(f"| {name} | – | – | – | – | – | – |"); continue
    w = mmpd_cv(tags, lambda r: r[3] == 4); ho = mmpd_cv(tags, subs=range(7, 34)); how = mmpd_cv(tags, lambda r: r[3] == 4, subs=range(7, 34))
    emit(f"| {name} | {mt['seeds']} | {mt['mae']:.2f} ± {mt['se']:.2f} | {mt['rmse']:.2f} | {w['mae']:.2f} | {ho['mae']:.2f} | {how['mae']:.2f} |"); JS["indomain"][f"MMPD|{name}"] = dict(mt, walk=w["mae"], ho=ho["mae"], ho_walk=how["mae"])
zs = metrics("NF_HYB_L", "MMPD"); zw = metrics("NF_HYB_L", "MMPD", mmpd_motion_sel(4)); emit(f"| (reference) SpecPhys-B zero-shot, held-out 7–33 only | 3 | – | – | – | {zs['mae']:.2f} ± {zs['se']:.2f} | {zw['mae']:.2f} |\n")
def nirp_cv(tags, ax):
    acc = {}
    for t in tags:
        f = f"{R}/agg_NRCV_{t}.json"
        if not os.path.exists(f): continue
        for r in json.load(open(f)).get(ax, []): acc.setdefault(r["vid"], []).append((r["gt"], r["pred"]))
    if not acc: return None
    return _cv_metrics(acc, {}, None, sum(os.path.exists(f"{R}/agg_NRCV_{t}.json") for t in tags))
emit("### MR-NIRP RGB subject-exclusive 5-fold (19 subjects, driving sessions)\n"); emit("| Setting | seeds | still | small motion | large motion |"); emit("|---|---|---|---|---|")
for name, tags in [("PhysFormer in-domain: VIPL + corpus + MR-NIRP folds (chain 47, NRCV_HYBN_L)", ["HYBN_L"]), ("SpecPhys-B 2-stage: corpus checkpoint → MR-NIRP fold (ours, readout-consistent labels)", ["NF_L", "NFs2_L", "NFs3_L"])]:
    mts = [nirp_cv(tags, ax) for ax in ("NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE")]
    emit(f"| {name} | {mts[0]['seeds'] if mts[0] else '–'} | " + " | ".join("–" if m is None else f"{m['mae']:.2f} ± {m['se']:.2f}" for m in mts) + " |"); JS["indomain"][f"NIRP|{name}"] = [None if m is None else m["mae"] for m in mts]
emit(f"| (reference) SpecPhys-B zero-shot | 3 | " + " | ".join(f"{metrics('NF_HYB_L', ax)['mae']:.2f} ± {metrics('NF_HYB_L', ax)['se']:.2f}" for ax in ("NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE")) + " |\n")
# ===================== Table 10. Backbone-matched comparison on the identical pool (체인 74, 57 문서) =====================
emit("## Table 10. Backbone-matched comparison on the identical 366-clip pool (zero-shot MAE; each backbone fine-tuned from its public UBFC-rPPG weights with its native loss; 3 seeds unless noted)\n")
BB_AX = [("VIPL-HR", "VIPL"), ("MMPD 7–33", "MMPD"), ("MR-NIRP small", "NIRP_SMALL"), ("MR-NIRP large", "NIRP_LARGE"), ("RTNI flicker", "RTNI"), ("PURE 07–10", "PURE_HO")]
BB_ROWS = [("PhysFormer", [("public weights", "PhysFormer"), ("+ pool, native loss", "CoTAM_L"), ("+ pool, decomposition loss", "CoTAM_HYB4_L"), ("+ pool, native + spectral losses", "PF_SPEC_L")]),
           ("RhythmFormer", [("public weights", "RhythmFormer"), ("+ pool, native loss", "RF_POOL_L"), ("+ pool, native + spectral losses", "RF_SPEC_L")]),
           ("PhysMamba", [("public weights", "PMbase"), ("+ pool, native loss", "PM_POOL_L"), ("+ pool, decomposition loss (128-frame clips)", "PM_HYB4_L"), ("+ pool, native + spectral losses", "PM_SPEC_L"), ("+ pool, native + spectral losses (weight 2.0)", "PM_SPEC_W20_L")]),
           ("FactorizePhys", [("public weights", "FactorizePhys"), ("+ pool, decomposition loss", "FP_OFF_HYB"), ("+ pool, spectral losses (SpecPhys-B, ours)", "NF_HYB_L")])]
emit("| Backbone | setting | " + " | ".join(a for a, _ in BB_AX) + " | seeds |"); emit("|---|---|" + "---|" * (len(BB_AX) + 1))
for bb, settings in BB_ROWS:
    for nm, key in settings:
        mts = [metrics(key, ax) for _, ax in BB_AX]; seeds = max([m["seeds"] for m in mts if m] or [0])
        emit(f"| {bb} | {nm} | " + " | ".join("–" if m is None else f"{m['mae']:.2f}" for m in mts) + f" | {seeds if seeds else '–'} |")
        JS.setdefault("backbone", {})[f"{bb}|{nm}"] = {ax: (None if m is None else m["mae"]) for (_, ax), m in zip(BB_AX, mts)}
emit("\nPublic PhysMamba weights (PMbase) are evaluated on VIPL-HR and MMPD only; the PhysMamba decomposition-loss cell was trained with 128-frame clips and evaluated on MMPD only. \"native + spectral losses\" = the backbone's own loss plus the SpecPhys-B output-spectrum terms (HR-band likelihood 1.0, halving penalty 0.5; chains 75–76). Statistical tests: results/backbone_judge.txt (B1, B2) and results/crossbb_judge.txt (X1–X4).\n")
# ===================== Table 11. MMPD literature split (in-domain) =====================
emit("## Table 11. MMPD literature split — train subjects 1–23, test subjects 27–33 (140 clips; in-domain, labels of the target dataset used; never compared with zero-shot columns)\n")
MOT = [("stationary", 1), ("rotation", 2), ("talking", 3), ("walking", 4)]
LIT_ROWS = [("PhysFormer, MMPD 1–23 only", "ML_BASE_L", "in-domain"), ("PhysFormer, corpus + MMPD 1–23, decomposition loss", "ML_HYB_L", "in-domain"),
            ("SpecPhys-B, MMPD 1–23 only (from public FactorizePhys weights)", "ML_NFO_L", "in-domain"), ("SpecPhys-B two-stage: corpus checkpoint → MMPD 1–23 + pool", "ML_NF_L", "in-domain"),
            ("(reference) SpecPhys-B zero-shot on the same 140 clips", "NF_HYB_L", "zero-shot"), ("(reference) public FactorizePhys zero-shot", "FactorizePhys", "zero-shot"), ("(reference) public RhythmFormer zero-shot", "RhythmFormer", "zero-shot"), ("(reference) POS", "POS", "classical")]
emit("| Setting | kind | seeds | MAE ± SE | RMSE | Corr | " + " | ".join(m for m, _ in MOT) + " |"); emit("|---|---|---|---|---|---|" + "---|" * len(MOT))
for nm, key, kind in LIT_ROWS:
    mt = metrics(key, "MMPD_LIT")
    if mt is None: emit(f"| {nm} | {kind} | – | – | – | – | " + " | ".join("–" for _ in MOT) + " |"); continue
    per = [metrics(key, "MMPD_LIT", sel=(lambda r, mv=mv: r[3] == mv)) for _, mv in MOT]
    rstr = "–" if np.isnan(mt["r"]) else f"{mt['r']:.2f}"
    emit(f"| {nm} | {kind} | {mt['seeds']} | {mt['mae']:.2f} ± {mt['se']:.2f} | {mt['rmse']:.2f} | {rstr} | " + " | ".join("–" if p is None else f"{p['mae']:.2f}" for p in per) + " |")
    JS.setdefault("mmpd_lit", {})[nm] = dict(mae=mt["mae"], se=mt["se"], rmse=mt["rmse"], r=mt["r"], seeds=mt["seeds"], motion={m: (None if p is None else p["mae"]) for (m, _), p in zip(MOT, per)})
emit("\nLiterature intra-MMPD values (RhythmFormer paper; different readout and split granularity, orientation only): PhysFormer 11.99, PhysFormer++ 4.88, RhythmFormer 3.07 bpm. Statistical tests: results/backbone_judge.txt (family B3).\n")
# ===================== Phase 0 taxonomy =====================
p0 = f"{R}/selfsup_phase0.json"
if os.path.exists(p0):
    P = json.load(open(p0)); emit("## Table 8. Failure taxonomy of the zero-shot SpecPhys-B model in its weak regimes (videos × 3 seeds; failure = |error| > 5 bpm)\n")
    emit("| Regime | n | ok | lock-in on a nuisance peak | halving/doubling | prior collapse | other | lock-in share of failures | shuffled-prediction null | excess (z) |"); emit("|---|---|---|---|---|---|---|---|---|---|")
    EN = {"운동 후 v7": "v7 post-exercise", "심박 ≥90 (v7 제외)": "HR ≥90 bpm (excl. v7)", "심박 ≥120": "HR ≥120 bpm", "보행 m4": "walking", "대화 m3": "talking", "회전 m2": "rotation", "운동 후 e1": "after exercise", "대동작 전체": "large motion"}
    for ax, grp in P.items():
        for g0, d in grp.items():
            g = EN.get(g0, g0)
            m = d.get("models", {}).get("NF_HYB_L")
            if not m: continue
            fr = m["frac"]; nl = m.get("fail_lockin_null", {})
            emit(f"| {ax} {g} | {d['n']} | {fr['ok']:.0f}% | {fr['lockin']:.0f}% | {fr['half_double']:.0f}% | {fr['prior']:.0f}% | {fr['other']:.0f}% | {100 * m['fail_lockin']:.0f}% | {100 * nl.get('shuffle_mean', float('nan')):.0f}% | {100 * nl.get('excess', float('nan')):+.1f} pt ({nl.get('z', float('nan')):+.1f}) |")
            JS["phase0"][f"{ax} {g}"] = dict(n=d["n"], frac=fr, fail_lockin=m["fail_lockin"], null=nl)
    emit("")
# ===================== statistical tests (from judges) =====================
emit("## Table 9. Paired Wilcoxon tests per video (3-seed pooled), Holm-corrected within each pre-registered family (from results/nf_judge.json, selfsup_judge.json, candidates_judge.json)\n")
emit("| Family / hypothesis | n | reference → cell | Δ MAE | p | Holm | verdict |"); emit("|---|---|---|---|---|---|---|")
for jf, fam in (("nf_judge.json", "SpecPhys-B vs public FactorizePhys (48 §3)"), ("selfsup_judge.json", "self-supervised separation (53 §4)"), ("candidates_judge.json", "remaining candidates (55 §2)")):
    f = f"{R}/{jf}"
    if not os.path.exists(f): continue
    J = json.load(open(f)); prim = J.get("primary", {})
    items = []
    for k, v in prim.items():
        if isinstance(v, dict) and "d" in v: items.append((k, v))
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, dict) and "d" in v2: items.append((k2, v2))
    for k, v in items:
        emit(f"| {fam}: {k} | {v['n']} | {v['b']:.2f} → {v['a']:.2f} | {v['d']:+.2f} | {v['p']:.4f} | {v.get('holm', float('nan')):.4f} | {'supported' if v.get('supported') else 'rejected'} |")
emit("")
open(OUT_MD, "w").write("\n".join(L) + "\n"); json.dump(JS, open(OUT_JS, "w"), indent=1, ensure_ascii=False)
print(f"-> {OUT_MD} ({len(L)} lines), {OUT_JS}")
