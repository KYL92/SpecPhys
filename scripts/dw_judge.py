#!/usr/bin/env python3
"""체인 56 판정(41 문서): 도메인 고정 가중 코퍼스 셀 vs 앵커(채택 CoTAF_TW_L / 분해 CoTAM_HYB4_L) + 최강 기존 방법(공개 FactorizePhys: VIPL FPbase, MMPD FactorizePhys).
  python scripts/dw_judge.py phase1|gate|final
phase1/gate: 시드 100 vs 앵커 시드 100; final: 시드 100/200/300 영상별 평균 풀링(앵커도 3시드; FactorizePhys 공개 가중치는 단일). 영상 단위 쌍대 Wilcoxon, 1차 가족 Holm.
게이트(Phase 2): VIPL 문헌 GT Δ<0(vs 채택 모델)·p<.10 이고 강건성 가드 통과(RTNI·RTNC MAE<5, PURE held-out<2)."""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"
CELLS = ["DW_MCD2_L", "DW_MCD4_L", "DW_MCD4_HYB_L"]; ERM, HYB = "CoTAF_TW_L", "CoTAM_HYB4_L"
mode = sys.argv[1] if len(sys.argv) > 1 else "phase1"
SEEDS = [""] if mode in ("phase1", "gate") else ["", "_s200", "_s300"]


def wp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float("nan") if len(x) < 6 or np.allclose(x, y) else float(wilcoxon(x, y).pvalue)


def load(m, sfx=""):
    f = f"{R}/agg_{m}{sfx}.json"; return json.load(open(f)) if os.path.exists(f) else None


def pool(ds):
    ds = [d for d in ds if d]
    if not ds: return {}, 0
    ks = set.intersection(*[set(d) for d in ds]); return {k: float(np.mean([d[k] for d in ds])) for k in ks}, len(ds)


def metric_maps(m, seeds=None):
    per = {}
    for s in (seeds or SEEDS):
        d = load(m + s)
        if d:
            for ax, rows in d.items():
                if not (isinstance(rows, list) and rows and "err" in rows[0]): continue
                per.setdefault(ax, []).append({str(r["vid"]): float(r["err"]) for r in rows})
                if ax == "PURE": per.setdefault("PURE_heldout", []).append({str(r["vid"]): float(r["err"]) for r in rows if int(r["vid"]) >= 701})
            nm = {}
            for ax in ("NIRP_SMALL", "NIRP_LARGE"): nm.update({ax + ":" + str(r["vid"]): float(r["err"]) for r in d.get(ax, []) if "err" in r})
            if nm: per.setdefault("NIRP_motion", []).append(nm)
        mm = load(m + s, "_MMPD")
        if mm:
            rows = [r for r in mm if r["subject"] >= 7]
            per.setdefault("MMPD_all", []).append({(r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]): float(r["err"]) for r in rows})
            per.setdefault("MMPD_motion", []).append({(r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]): float(r["err"]) for r in rows if r["motion"] != 1})
        vv = load(m + s, "_VIPL")
        if vv:
            per.setdefault("VIPL_lit", []).append({r["vid"]: float(r["err_lit"]) for r in vv if r["err_lit"] is not None})
            per.setdefault("VIPL_canon", []).append({r["vid"]: float(r["err"]) for r in vv})
            per.setdefault("VIPL_motion(v2+v9)", []).append({r["vid"]: float(r["err_lit"]) for r in vv if r["scenario"] in ("v2", "v9") and r["err_lit"] is not None})
    return {k: pool(v) for k, v in per.items()}


def cmp(a, b, metric, MA, MB):
    if metric not in MA or metric not in MB: return None
    (x, na), (y, nb) = MA[metric], MB[metric]; ks = sorted(set(x) & set(y), key=str)
    if not ks: return None
    xa = np.array([x[k] for k in ks]); yb = np.array([y[k] for k in ks])
    return dict(a=a, b=b, metric=metric, n=len(ks), seeds=[na, nb], mae_a=float(xa.mean()), mae_b=float(yb.mean()), delta=float(yb.mean() - xa.mean()), p=wp(xa, yb))


def holm(tests):
    ps = [(i, t["p"]) for i, t in enumerate(tests) if t and not np.isnan(t["p"])]; ps.sort(key=lambda z: z[1]); m = len(ps); prev = 0.0
    for r, (i, p) in enumerate(ps): adj = min(1.0, max(prev, (m - r) * p)); tests[i]["p_holm"] = adj; prev = adj
    return tests


def show(t):
    if t is None: return
    ph = f" Holm={t['p_holm']:.4f}" if "p_holm" in t else ""
    print(f"  {t['metric']:20s} n={t['n']:4d} {t['a']}({t['seeds'][0]}s) {t['mae_a']:6.2f} | {t['b']}({t['seeds'][1]}s) {t['mae_b']:6.2f} | Δ {t['delta']:+6.2f} p={t['p']:.4f}{ph}")


M = {m: metric_maps(m) for m in [ERM, HYB] + CELLS}
FP = metric_maps("FPbase", [""]); FPM = metric_maps("FactorizePhys", [""])
if "MMPD_all" in FPM: FP["MMPD_all"] = FPM["MMPD_all"]; FP["MMPD_motion"] = FPM["MMPD_motion"]
avail = [c for c in CELLS if M[c]]
print(f"=== 체인 56 판정 mode={mode} seeds={SEEDS} cells={avail}")
res = {"mode": mode, "F1_vs_anchor": [], "F2_vs_factorizephys": [], "guard": {}, "secondary": []}
for c in avail:
    anc = HYB if "_HYB_" in c else ERM
    res["F1_vs_anchor"].append(cmp(anc, c, "VIPL_lit", M[anc], M[c]))
    res["F2_vs_factorizephys"].append(cmp("FPbase", c, "VIPL_lit", FP, M[c]))
    g = {}
    for met in ("RTNI", "RTNC", "PURE_heldout"):
        if met in M[c]: g[met] = float(np.mean(list(M[c][met][0].values())))
    # [41 §3 개정 02:55, 게이트 실행 전] PURE held-out 가드는 절대값 2 또는 '자기 앵커 + 1' 중 큰 쪽(분해 앵커의 PURE held-out 은 5.9 로 절대 기준 2 가 앵커보다 엄격했음)
    anc_pure = float(np.mean(list(M[anc]["PURE_heldout"][0].values()))) if "PURE_heldout" in M[anc] else 99
    g["anchor_PURE"] = anc_pure; g["PURE_limit"] = max(2.0, anc_pure + 1.0)
    g["pass"] = bool(g.get("RTNI", 99) < 5 and g.get("RTNC", 99) < 5 and g.get("PURE_heldout", 99) < g["PURE_limit"]); res["guard"][c] = g
    for met in ("MMPD_all", "MMPD_motion", "VIPL_motion(v2+v9)", "NIRP_motion", "DLCN", "MCD_A", "MCD_B", "NIRP_STILL", "PURE_heldout", "RTNI", "RTNC"):
        res["secondary"].append(cmp(anc, c, met, M[anc], M[c]))
    for met in ("MMPD_all", "MMPD_motion"): res["secondary"].append(cmp("FactorizePhys", c, met, FP, M[c]))
for fam in ("F1_vs_anchor", "F2_vs_factorizephys"):
    res[fam] = holm([t for t in res[fam] if t]); print(f"--- {fam} (Holm {len(res[fam])})")
    for t in res[fam]: show(t)
print("--- robustness guard (RTNI/RTNC<5, PURE held-out<2):", json.dumps(res["guard"]))
print("--- secondary (uncorrected)")
res["secondary"] = [t for t in res["secondary"] if t]
for t in res["secondary"]: show(t)
json.dump(res, open(f"{R}/dw_judge_{mode if mode != 'gate' else 'phase1'}.json", "w"), indent=1, ensure_ascii=False)
if mode == "gate":
    go = [t["b"] for t in res["F1_vs_anchor"] if t["delta"] < 0 and (np.isnan(t["p"]) or t["p"] < 0.10) and res["guard"][t["b"]]["pass"]]
    print(" ".join(go) if go else "NONE")
