#!/usr/bin/env python3
"""체인 55 판정(40 문서): 도메인 일반화 기준선(GroupDRO·VREx·DANN) × {공식 코퍼스 TW 풀, 하이브리드 풀+분해 손실} vs 앵커(ERM 공식 모델 CoTAF_TW_L, 분해 모델 CoTAM_HYB4_L).
  python scripts/dg_judge.py phase1   # 시드 100 끼리(앵커도 시드 100 run 만) — Phase 2 게이트용
  python scripts/dg_judge.py final    # 시드 100/200/300 영상별 평균 풀링(앵커도 같은 3 시드) — 본 판정
  python scripts/dg_judge.py gate     # phase1 규칙으로 Phase 2 로 넘길 방법 이름만 출력(마지막 줄, 공백 구분; 없으면 'NONE')
영상 단위 쌍대 Wilcoxon, 1차 가족 Holm. 출력: results/dg_judge_{mode}.json + 표."""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"
METHODS = ["GDRO", "VREX", "DANN"]; ERM, HYB = "CoTAF_TW_L", "CoTAM_HYB4_L"
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


def metric_maps(m):
    """모델 m 의 시드별 {metric_name: {key: err}} 목록 → metric 별 풀링"""
    per = {}
    for s in SEEDS:
        d = load(m + s)
        if d:
            for ax, rows in d.items():
                if not (isinstance(rows, list) and rows and "err" in rows[0]): continue
                per.setdefault(ax, []).append({str(r["vid"]): float(r["err"]) for r in rows})
                if ax == "PURE":  # held-out 피험자 7~10, 움직임 trial 03~06
                    per.setdefault("PURE_heldout", []).append({str(r["vid"]): float(r["err"]) for r in rows if int(r["vid"]) >= 701})
                    per.setdefault("PURE_motion", []).append({str(r["vid"]): float(r["err"]) for r in rows if int(r["vid"]) >= 701 and 3 <= int(r["vid"]) % 100 <= 6})
            nm = {}
            for ax in ("NIRP_SMALL", "NIRP_LARGE"):
                nm.update({ax + ":" + str(r["vid"]): float(r["err"]) for r in d.get(ax, []) if "err" in r})
            if nm: per.setdefault("NIRP_motion", []).append(nm)
        mm = load(m + s, "_MMPD")
        if mm:
            rows = [r for r in mm if r["subject"] >= 7]
            per.setdefault("MMPD_all", []).append({r["vid"]: float(r["err"]) for r in rows})
            per.setdefault("MMPD_motion", []).append({r["vid"]: float(r["err"]) for r in rows if r["motion"] != 1})
            per.setdefault("MMPD_stationary", []).append({r["vid"]: float(r["err"]) for r in rows if r["motion"] == 1})
        vv = load(m + s, "_VIPL")
        if vv:
            per.setdefault("VIPL_lit", []).append({r["vid"]: float(r["err_lit"]) for r in vv if r["err_lit"] is not None})
            per.setdefault("VIPL_canon", []).append({r["vid"]: float(r["err"]) for r in vv})
            per.setdefault("VIPL_motion(v2+v9)", []).append({r["vid"]: float(r["err_lit"]) for r in vv if r["scenario"] in ("v2", "v9") and r["err_lit"] is not None})
    return {k: pool(v) for k, v in per.items()}


def cmp(a, b, metric, MA, MB):
    """Δ = b − a (음수 = b 우세)"""
    if metric not in MA or metric not in MB: return None
    (x, na), (y, nb) = MA[metric], MB[metric]; ks = sorted(set(x) & set(y))
    if not ks: return None
    xa = np.array([x[k] for k in ks]); yb = np.array([y[k] for k in ks])
    return dict(a=a, b=b, metric=metric, n=len(ks), seeds=[na, nb], mae_a=float(xa.mean()), mae_b=float(yb.mean()), delta=float(yb.mean() - xa.mean()), p=wp(xa, yb))


def holm(tests):
    ps = [(i, t["p"]) for i, t in enumerate(tests) if t and not np.isnan(t["p"])]; ps.sort(key=lambda z: z[1]); m = len(ps); prev = 0.0
    for r, (i, p) in enumerate(ps):
        adj = min(1.0, max(prev, (m - r) * p)); tests[i]["p_holm"] = adj; prev = adj
    return tests


def show(t, tag=""):
    if t is None: return
    ph = f" Holm={t['p_holm']:.4f}" if "p_holm" in t else ""
    print(f"  {tag:6s}{t['metric']:20s} n={t['n']:4d} {t['a']}({t['seeds'][0]}s) {t['mae_a']:6.2f} | {t['b']}({t['seeds'][1]}s) {t['mae_b']:6.2f} | Δ {t['delta']:+6.2f} p={t['p']:.4f}{ph}")


M = {}
cells = [f"DG_{m}_L" for m in METHODS] + [f"DG_{m}_HYB_L" for m in METHODS]
for m in [ERM, HYB] + cells: M[m] = metric_maps(m)
avail = [c for c in cells if M[c]]
print(f"=== 체인 55 판정 mode={mode} seeds={SEEDS} cells={avail}")
res = {"mode": mode, "seeds": SEEDS, "F1_dg_vs_erm": [], "F2_decomp_on_dg": [], "F3_ours_vs_dg": [], "secondary": []}
# F1: DG(TW 풀) vs ERM 앵커 — VIPL 문헌 GT 전체, MMPD held-out 전체 (Holm 6)
for m in METHODS:
    for met in ("VIPL_lit", "MMPD_all"):
        t = cmp(ERM, f"DG_{m}_L", met, M[ERM], M[f"DG_{m}_L"]); res["F1_dg_vs_erm"].append(t)
# F2: 분해 손실 on DG — DG_m_HYB vs DG_m — MMPD 움직임 (Holm 3); 2차 NIRP/PURE/VIPL 움직임
for m in METHODS:
    res["F2_decomp_on_dg"].append(cmp(f"DG_{m}_L", f"DG_{m}_HYB_L", "MMPD_motion", M[f"DG_{m}_L"], M[f"DG_{m}_HYB_L"]))
    for met in ("NIRP_motion", "PURE_motion", "VIPL_motion(v2+v9)", "MMPD_all", "VIPL_lit"):
        res["secondary"].append(cmp(f"DG_{m}_L", f"DG_{m}_HYB_L", met, M[f"DG_{m}_L"], M[f"DG_{m}_HYB_L"]))
# F3: 우리 분해 모델 vs 지표별 최선 DG 셀(풀링 MAE 최소) — VIPL 문헌 GT, MMPD 전체 (Holm 2); 2차 전 쌍
for met in ("VIPL_lit", "MMPD_all"):
    best = min([c for c in avail if met in M[c] and M[c][met][0]], key=lambda c: np.mean(list(M[c][met][0].values())), default=None)
    if best: res["F3_ours_vs_dg"].append(cmp(best, HYB, met, M[best], M[HYB]))
    for c in avail:
        res["secondary"].append(cmp(c, HYB, met, M[c], M[HYB])); res["secondary"].append(cmp(c, ERM, met, M[c], M[ERM]))
for c in avail:
    for met2 in ("MMPD_motion", "NIRP_motion", "PURE_motion", "DLCN", "MCD_A", "MCD_B", "PURE_heldout", "NIRP_STILL", "RTNI", "RTNC"):
        res["secondary"].append(cmp(ERM, c, met2, M[ERM], M[c]))
for fam in ("F1_dg_vs_erm", "F2_decomp_on_dg", "F3_ours_vs_dg"):
    res[fam] = holm([t for t in res[fam] if t]); print(f"--- {fam} (Holm {len(res[fam])})")
    for t in res[fam]: show(t)
print("--- secondary (uncorrected)")
for t in res["secondary"]:
    if t: show(t)
res["secondary"] = [t for t in res["secondary"] if t]
json.dump(res, open(f"{R}/dg_judge_{mode if mode != 'gate' else 'phase1'}.json", "w"), indent=1, ensure_ascii=False)
if mode == "gate":  # Phase 2 규칙: 방법 m 의 TW 셀 또는 HYB 셀이 앵커 대비 1차 지표(VIPL 문헌 GT 또는 MMPD 전체)에서 Δ<0 이고 비보정 p<.10
    go = []
    for m in METHODS:
        ok = False
        for c, anc in ((f"DG_{m}_L", ERM), (f"DG_{m}_HYB_L", HYB)):
            for met in ("VIPL_lit", "MMPD_all"):
                t = cmp(anc, c, met, M[anc], M[c])
                if t and t["delta"] < 0 and (np.isnan(t["p"]) or t["p"] < 0.10): ok = True
        if ok: go.append(m)
    print(" ".join(go) if go else "NONE")
