#!/usr/bin/env python3
"""[체인 53, 36 문서 §2] UBFC 기반 요인 실험 판정: 백본별 U0→U1(합성 데이터)→U2(분해 손실)→U3(실 노이즈 증강) 쌍대 + 백본 간·공개 가중치 대비.
  python scripts/ubfc_judge.py [phase1|phase2]   → 표 + results/ubfc_judge.json, 마지막 줄 PHASE2=0|1 (Phase 1 뒤 시드 추가 규칙: 1차 비교 중 p<.05 또는 |Δ|≥0.3 이 하나라도)
시드 ""/_s200/_s300 을 영상(피험자) 단위 평균으로 풀링. 1차 축: MMPD held-out 움직임(324편)·VIPL 전체(장치 기준값); Holm 은 백본별 1차 6건."""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"; SEEDS = ("", "_s200", "_s300")
PUB = {"PF": ("base", "base", "PFbase"), "FP": ("FactorizePhys", "FactorizePhys", "FPbase")}  # (13축, MMPD, VIPL) 공개 가중치 집계 이름


def wp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float("nan") if len(x) < 6 or np.allclose(x, y) else float(wilcoxon(x, y).pvalue)


def load(f):
    return json.load(open(f)) if os.path.exists(f) else None


def pool(dicts):
    dicts = [d for d in dicts if d]
    if not dicts: return {}, 0
    ks = set.intersection(*[set(d) for d in dicts]); return {k: float(np.mean([d[k] for d in dicts])) for k in ks}, len(dicts)


def series(m, pub=None):
    """모델 m(시드 풀링) → {축키: {vid: err}}; pub=(axes,mmpd,vipl) 이름이면 공개 가중치(단일)."""
    out = {}
    names = [m + s for s in SEEDS] if pub is None else None
    # 13축
    ds = [load(f"{R}/agg_{n}.json") for n in (names or [pub[0]])]
    axes = {}
    for d in ds:
        if not d: continue
        for ax, rows in d.items():
            if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "err" in rows[0]:
                axes.setdefault(ax, []).append({str(r["vid"]): float(r["err"]) for r in rows})
    for ax, lst in axes.items(): out[ax], _ = pool(lst)
    # MMPD held-out
    ds = [load(f"{R}/agg_{n}_MMPD.json") for n in (names or [pub[1]])]
    lst = [{(r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]): float(r["err"]) for r in d if r["subject"] >= 7} for d in ds if d]
    if lst:
        allm, _ = pool(lst); out["MMPD"] = allm; out["MMPD_motion"] = {k: v for k, v in allm.items() if k[2] != 1}; out["MMPD_walk"] = {k: v for k, v in allm.items() if k[2] == 4}; out["MMPD_still"] = {k: v for k, v in allm.items() if k[2] == 1}
    # VIPL
    ds = [load(f"{R}/agg_{n}_VIPL.json") for n in (names or [pub[2]])]
    lst = [{r["vid"]: float(r["err_lit"]) for r in d if r["err_lit"] is not None} for d in ds if d]
    if lst:
        allv, _ = pool(lst); out["VIPL"] = allv
        meta = {r["vid"]: r for d in ds if d for r in d}; out["VIPL_motion"] = {k: v for k, v in allv.items() if meta[k]["scenario"] in ("v2", "v9")}
    out["_nseeds"] = sum(1 for n in (names or []) if load(f"{R}/agg_{n}.json"))
    return out


def contrast(a, b, key):
    if key not in a or key not in b: return None
    ks = sorted(set(a[key]) & set(b[key]))
    if len(ks) < 6: return None
    x = np.array([a[key][k] for k in ks]); y = np.array([b[key][k] for k in ks])
    return dict(n=len(ks), a=float(x.mean()), b=float(y.mean()), delta=float(y.mean() - x.mean()), p=wp(x, y))


def holm(ps):
    idx = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(idx): run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


AXES = ["MMPD", "MMPD_motion", "MMPD_walk", "MMPD_still", "VIPL", "VIPL_motion", "REALTEST", "PURE", "MCD_A", "MCD_B", "DLCN", "NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE", "RTNI", "RTNC", "C3", "C6", "E3"]
PRIMARY = [("U0", "U1"), ("U1", "U2"), ("U2", "U3")]; PRIMARY_AXES = ["MMPD_motion", "VIPL"]
LABEL = {"U0": "UBFC만", "U1": "+가상얼굴", "U2": "+분해손실", "U3": "+실노이즈증강"}


def main():
    out = {}; phase2 = False
    S = {bb: {c: series(f"{bb}_{c}") for c in ("U0", "U1", "U2", "U3")} for bb in ("PF", "FP")}
    for bb in ("PF", "FP"):
        print(f"=== 백본 {'PhysFormer' if bb=='PF' else 'FactorizePhys'} (시드 수: " + ", ".join(f"{c}={S[bb][c]['_nseeds']}" for c in S[bb]) + ")")
        print("  셀 요약(MAE): " + " | ".join(f"{c}({LABEL[c]}) MMPD {np.mean(list(S[bb][c]['MMPD'].values())):.2f}/움직임 {np.mean(list(S[bb][c]['MMPD_motion'].values())):.2f}, VIPL {np.mean(list(S[bb][c]['VIPL'].values())):.2f}" if "MMPD" in S[bb][c] and "VIPL" in S[bb][c] else f"{c} (미완)" for c in S[bb]))
        prim = []
        for a, b in PRIMARY + [("U0", "U3")]:
            for ax in AXES:
                r = contrast(S[bb][a], S[bb][b], ax)
                if r is None: continue
                out[f"{bb}:{a}->{b}:{ax}"] = r
                if (a, b) in PRIMARY and ax in PRIMARY_AXES: prim.append((f"{bb}:{a}->{b}:{ax}", r))
                flag = "*" if (r["p"] < 0.05) else " "
                print(f"  {a}→{b} ({LABEL[a]}→{LABEL[b]}) {ax:12s} n={r['n']:4d} {r['a']:6.2f} → {r['b']:6.2f} Δ{r['delta']:+6.2f} p={r['p']:.4f}{flag}")
        if prim:
            adj = holm([r["p"] if np.isfinite(r["p"]) else 1.0 for _, r in prim]); print("  -- 1차 가족 Holm --")
            for (nm, r), pa in zip(prim, adj):
                out[nm]["p_holm"] = float(pa); sig = pa < 0.05 and r["delta"] < 0; print(f"  {nm:24s} Δ{r['delta']:+.2f} p={r['p']:.4f} holm={pa:.4f} {'유의 개선' if sig else ''}")
                if r["p"] < 0.05 or abs(r["delta"]) >= 0.3: phase2 = True
        pub = series(None, PUB[bb])
        for ax in ("MMPD", "MMPD_motion", "VIPL", "VIPL_motion", "DLCN", "NIRP_SMALL", "RTNI", "PURE", "MCD_A"):
            r = contrast(pub, S[bb]["U3"], ax)
            if r: out[f"{bb}:PUB->U3:{ax}"] = r; print(f"  공개가중치→U3 {ax:12s} n={r['n']:4d} {r['a']:6.2f} → {r['b']:6.2f} Δ{r['delta']:+6.2f} p={r['p']:.4f}")
    print("=== 백본 간(같은 데이터·손실): PhysFormer U3 → FactorizePhys U3")
    for ax in AXES:
        r = contrast(S["PF"]["U3"], S["FP"]["U3"], ax)
        if r: out[f"PF_U3->FP_U3:{ax}"] = r; print(f"  {ax:12s} n={r['n']:4d} {r['a']:6.2f} → {r['b']:6.2f} Δ{r['delta']:+6.2f} p={r['p']:.4f}")
    json.dump(out, open(f"{R}/ubfc_judge.json", "w"), indent=1, ensure_ascii=False); print(f"PHASE2={int(phase2)}")


if __name__ == "__main__":
    main()
