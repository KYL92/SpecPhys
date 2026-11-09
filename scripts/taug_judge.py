#!/usr/bin/env python3
"""체인 50 판정(32 문서 §6): 시간 배속 증강(A) + 검증 epoch 선택(V) 셀의 사전 등록 판정.
  python scripts/taug_judge.py phase1 [FOLD=1]   → 표 + results/taug_phase1.json (winner: TWgA|TWgV|TWgAV|none, h_t2: bool)
  python scripts/taug_judge.py phase2 TAG         → 5-fold 합산 CV_{TAG}_VIPL vs CV_TW_L / CV_HYB_L / CV_REAL_L [/ CV_TWg_L]
읽기: 셀의 자체 테스트 출력(agg_{M}_VIPL.json; V 셀 = 검증 최적 epoch, 아니면 최종 epoch) + epoch 곡선 agg_EP_{M}_E{e}_VIPL.json(E9 = 최종 epoch 읽기).
판정 규칙: 영상 단위 쌍대 Wilcoxon(문헌 GT 주·정본 부), 공동학습 1차 가족 3건 Holm, 붕괴 = 문헌 MAE≥10 또는 r<.3, H-T2 = 최종 epoch 문헌 MAE<8 & r>.5 & 반감<20,
zero-shot 게이트 = DLCN MAE≤2.5 & MR-NIRP 움직임(피험자 쌍대) 유의 악화 아님."""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"


def wp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float("nan") if len(x) < 6 or np.allclose(x, y) else float(wilcoxon(x, y).pvalue)


def rows(m):
    f = f"{R}/agg_{m}_VIPL.json"; return {r["vid"]: r for r in json.load(open(f))} if os.path.exists(f) else None


def summ(a):
    ks = [k for k in a if a[k]["err_lit"] is not None]
    e = np.array([a[k]["err"] for k in ks]); l = np.array([a[k]["err_lit"] for k in ks]); g = np.array([a[k]["gt_lit"] for k in ks]); p = np.array([a[k]["pred"] for k in ks])
    rat = np.array([a[k]["ratio"] for k in ks]); hi = g >= 100; v7 = np.array([a[k]["scenario"] == "v7" for k in ks])
    return dict(n=len(ks), mae=float(e.mean()), mae_lit=float(l.mean()), r_lit=float(np.corrcoef(g, p)[0, 1]), halving=int(np.sum(np.abs(rat - 0.5) < 0.1)),
                big=int(np.sum(l > 15)), mae_lit_ge100=float(l[hi].mean()) if hi.any() else float("nan"), mae_lit_v7=float(l[v7].mean()) if v7.any() else float("nan"))


def paired(a, b):
    ks = [k for k in a if k in b and a[k]["err_lit"] is not None and b[k]["err_lit"] is not None]
    la = np.array([a[k]["err_lit"] for k in ks]); lb = np.array([b[k]["err_lit"] for k in ks]); ea = np.array([a[k]["err"] for k in ks]); eb = np.array([b[k]["err"] for k in ks])
    return dict(n=len(ks), d_lit=float(lb.mean() - la.mean()), p_lit=wp(la, lb), d=float(eb.mean() - ea.mean()), p=wp(ea, eb))


def holm(ps):
    idx = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


def axes(m):
    f = f"{R}/agg_{m}.json"
    if not os.path.exists(f): return None
    d = json.load(open(f)); return {ax: {int(r["vid"]): r["err"] for r in d.get(ax, [])} for ax in ("NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE", "DLCN")}


def gate(m, ref="VF1_TWg_L"):
    A, B = axes(m), axes(ref)
    if A is None or B is None: return dict(ok=None, note="no transfer agg")
    dl = float(np.mean(list(A["DLCN"].values()))) if A["DLCN"] else float("nan")
    pairs = [(A[ax][s], B[ax][s]) for ax in ("NIRP_SMALL", "NIRP_LARGE") for s in A[ax] if s in B[ax]]
    y, x = np.array(pairs).T if pairs else (np.array([]), np.array([]))
    p = wp(x, y); d = float(y.mean() - x.mean()) if len(y) else float("nan")
    ok = (dl <= 2.5) and not (p < 0.05 and d > 0)
    return dict(ok=bool(ok), dlcn=dl, nirp_motion=float(y.mean()) if len(y) else None, nirp_motion_ref=float(x.mean()) if len(x) else None, delta=d, p=p)


def phase1(fold=1):
    ref = f"VF{fold}_TWg_L"; Rr = rows(ref); base_ref = rows(f"VF{fold}_BASE_L")
    cells = {f"VF{fold}_TWgA_L": "공동학습+증강(최종)", f"VF{fold}_TWgV_L": "공동학습+검증(검증최적)", f"VF{fold}_TWgAV_L": "공동학습+증강+검증(검증최적)",
             f"VF{fold}_BASEA_L": "VIPL단독+증강(최종)", f"VF{fold}_BASEV_L": "VIPL단독+검증(검증최적)", f"VF{fold}_BASEAV_L": "VIPL단독+증강+검증(검증최적)"}
    out = {"fold": fold, "ref": ref, "cells": {}, "curves": {}, "tests": {}}
    print(f"=== Phase 1 (fold {fold}) 읽기 요약: n / 정본 MAE / 문헌 MAE / r / 반감 / |err|>15 / GT≥100 MAE / v7 MAE")
    for m, desc in list(cells.items()) + [(ref, "앵커 TWg(최종)"), (f"VF{fold}_BASE_L", "VIPL단독 기존(최종)")]:
        a = rows(m)
        if a is None: print(f"  {m:16s} missing"); continue
        s = summ(a); out["cells"][m] = s
        print(f"  {m:16s} {desc:22s} n={s['n']} {s['mae']:5.2f} {s['mae_lit']:5.2f} r={s['r_lit']:.2f} hlv={s['halving']:3d} big={s['big']:3d} ge100={s['mae_lit_ge100']:5.1f} v7={s['mae_lit_v7']:5.1f}")
    print("=== epoch 곡선(문헌 GT MAE, E0..E9; V 셀의 최종 epoch 읽기 = E9)")
    for m in cells:
        cur = []
        for e in range(10):
            a = rows(f"EP_{m}_E{e}"); cur.append(round(summ(a)["mae_lit"], 2) if a else None)
        out["curves"][m] = cur; print(f"  {m:16s} {cur}")
    # 1차 가족(공동학습 3건) vs 앵커, Holm
    fam = [f"VF{fold}_TWgA_L", f"VF{fold}_TWgV_L", f"VF{fold}_TWgAV_L"]; ps = []; res = {}
    for m in fam:
        a = rows(m)
        if a is None or Rr is None: continue
        t = paired(Rr, a); res[m] = t; ps.append(t["p_lit"] if np.isfinite(t["p_lit"]) else 1.0)
    adj = holm(ps) if ps else []
    print(f"=== 공동학습 1차 비교 vs {ref} (Δ = 셀 − 앵커, 문헌 GT; Holm 보정 p)")
    winner, best = "none", None
    for (m, t), pa in zip(res.items(), adj):
        t["p_holm"] = float(pa); g = gate(m, ref); t["gate"] = g; out["tests"][m] = t
        sig = (pa < 0.05) and (t["d_lit"] < 0)
        print(f"  {m:16s} n={t['n']} Δ문헌 {t['d_lit']:+.2f} p={t['p_lit']:.4f} holm={pa:.4f} | Δ정본 {t['d']:+.2f} p={t['p']:.4f} | gate={g.get('ok')} (DLCN {g.get('dlcn')}, NIRP 움직임 Δ{g.get('delta')} p={g.get('p')}) | {'유의 개선' if sig else 'ns'}")
        if sig and g.get("ok") is not False and (best is None or out["cells"][m]["mae_lit"] < best): winner, best = m.split("_")[1], out["cells"][m]["mae_lit"]
    # run 내부: 검증 최적(자체 테스트) vs 최종(E9)
    print("=== run 내부 비교: 검증 최적 epoch 읽기 vs 같은 run 최종 epoch(E9) (Δ = 최적 − 최종)")
    for m in (f"VF{fold}_TWgV_L", f"VF{fold}_TWgAV_L", f"VF{fold}_BASEV_L", f"VF{fold}_BASEAV_L"):
        a, e9 = rows(m), rows(f"EP_{m}_E9")
        if a is None or e9 is None: continue
        t = paired(e9, a); out["tests"][m + "_bv_vs_last"] = t
        print(f"  {m:16s} 최종 {summ(e9)['mae_lit']:5.2f} → 최적 {summ(a)['mae_lit']:5.2f} Δ{t['d_lit']:+.2f} p={t['p_lit']:.4f} | 정본 Δ{t['d']:+.2f} p={t['p']:.4f}")
    # H-T2 / H-E1 / H-E2
    def nocollapse(s): return s is not None and s["mae_lit"] < 8 and s["r_lit"] > 0.5 and s["halving"] < 20
    sA = out["cells"].get(f"VF{fold}_BASEA_L"); h_t2 = nocollapse(sA)
    sBV = out["cells"].get(f"VF{fold}_BASEV_L"); tBV = out["tests"].get(f"VF{fold}_BASEV_L_bv_vs_last")
    h_e1 = bool(sBV and tBV and tBV["p_lit"] < 0.05 and tBV["d_lit"] < 0 and sBV["mae_lit"] < 8)
    tTV = out["tests"].get(f"VF{fold}_TWgV_L_bv_vs_last"); h_e2 = None if tTV is None else ("유익" if (tTV["p_lit"] < 0.05 and tTV["d_lit"] < 0) else ("악화" if (tTV["p_lit"] < 0.05 and tTV["d_lit"] > 0) else "무해"))
    out.update(winner=winner, h_t1=winner != "none", h_t2=bool(h_t2), h_e1=h_e1, h_e2=h_e2)
    print(f"=== 판정: H-T1 winner={winner} | H-T2(VIPL단독+증강 비붕괴)={h_t2} | H-E1(검증 선택이 VIPL단독 구제)={h_e1} | H-E2(공동학습 검증 선택)={h_e2}")
    json.dump(out, open(f"{R}/taug_phase1.json", "w"), indent=1, ensure_ascii=False); return out


def phase2(tag):
    a = rows(f"CV_{tag}_VIPL".replace("_VIPL", ""))
    if a is None: print(f"missing CV_{tag}"); return
    s = summ(a); print(f"=== 5-fold CV_{tag}: n={s['n']} 정본 {s['mae']:.2f} 문헌 {s['mae_lit']:.2f} r={s['r_lit']:.2f} 반감 {s['halving']} GT≥100 {s['mae_lit_ge100']:.1f}")
    out = {"tag": tag, "summary": s, "vs": {}}
    for ref in ("CV_TW_L", "CV_HYB_L", "CV_REAL_L", "CV_TWg_L", "CV_BASE_L"):
        b = rows(ref)
        if b is None: continue
        t = paired(b, a); out["vs"][ref] = t; print(f"  vs {ref:10s} Δ문헌 {t['d_lit']:+.2f} p={t['p_lit']:.4f} | Δ정본 {t['d']:+.2f} p={t['p']:.4f} (Δ = CV_{tag} − ref)")
    json.dump(out, open(f"{R}/taug_phase2_{tag}.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "phase1": phase1(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    elif mode == "phase2": phase2(sys.argv[2])
    else: raise SystemExit(__doc__)
