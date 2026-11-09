#!/usr/bin/env python3
"""VIPL-HR 판정 (사전 등록 _workspace/30_vipl_benchmark.md).
  python scripts/vipl_judge.py zeroshot   기준 TW(5시드 풀링) vs 하이브리드 HYB4(5시드 풀링) vs 소프 vs POS/CHROM — 전체·시나리오·소스별, 영상 단위 Wilcoxon
  python scripts/vipl_judge.py cv         CV_BASE_L vs CV_REAL_L vs CV_TW_L vs CV_HYB_L (+ _s200/_s300 있으면 시드 풀링) — 문헌 지표(SD/MAE/RMSE/r, 문헌 GT) + 쌍대 검정
출력: results/vipl_judge.json. 풀링 = 시드별 영상 오차의 평균(pool5_judge.py 와 동일 정의)."""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
RES = "/SPECPHYS_ROOT/_workspace/data/results"
SCEN = {"v1": "stable", "v2": "motion", "v3": "talking", "v4": "dark", "v5": "bright", "v6": "long-dist", "v7": "exercise", "v8": "phone-stable", "v9": "phone-motion"}


def load(name):
    p = f"{RES}/agg_{name}_VIPL.json"
    return {r["vid"]: r for r in json.load(open(p))} if os.path.exists(p) else None


def pooled(names, key):
    """시드별 agg 를 영상 단위로 풀링: err = 시드별 오차의 평균(pool5_judge 정의), preds = 시드별 예측 목록. 공통 영상만."""
    ds = [d for d in (load(n) for n in names) if d]
    if not ds: return None, 0
    vids = set.intersection(*[set(d) for d in ds])
    out = {}
    for v in vids:
        rs = [d[v] for d in ds]
        if any(r.get(key) is None for r in rs): continue
        r = dict(rs[0]); r["err"] = float(np.mean([x[key] for x in rs])); r["preds"] = [float(x["pred"]) for x in rs]; r["pred"] = float(np.mean(r["preds"]))
        out[v] = r
    return out, len(ds)


def lit_metrics(d, gkey):
    """문헌 지표(SD/MAE/RMSE/r)를 시드별로 계산해 시드 평균(단일 모델 기대 성능). 'ens' 는 시드 평균 예측(앙상블) 기준 참고치."""
    if not d: return None
    ns = len(next(iter(d.values()))["preds"]); g = np.array([r[gkey] for r in d.values()])
    per = []
    for i in range(ns):
        p = np.array([r["preds"][i] for r in d.values()]); e = p - g
        per.append(dict(SD=float(e.std()), MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())), r=float(np.corrcoef(g, p)[0, 1])))
    m = {k: float(np.mean([x[k] for x in per])) for k in ("SD", "MAE", "RMSE", "r")}; m["n"] = len(g); m["seeds"] = ns
    pe = np.array([r["pred"] for r in d.values()]) - g
    m["ens_MAE"] = float(np.abs(pe).mean()); m["ens_RMSE"] = float(np.sqrt((pe ** 2).mean()))
    return m


def paired(a, b, sel=lambda r: True):
    vids = [v for v in a if v in b and sel(a[v])]
    if len(vids) < 10: return dict(n=len(vids))
    ea = np.array([a[v]["err"] for v in vids]); eb = np.array([b[v]["err"] for v in vids])
    try: p = float(wilcoxon(ea, eb).pvalue)
    except ValueError: p = 1.0
    return dict(n=len(vids), a=float(ea.mean()), b=float(eb.mean()), diff=float((eb - ea).mean()), p=p)


def table(pairs, label):
    print(f"\n[{label}] 전체 + 시나리오 + 소스 (오차 = 영상 MAE, p = Wilcoxon, b−a<0 이면 b 우세)")
    for nm, (a, b) in pairs.items():
        if a is None or b is None: print(f"  {nm}: 자료 없음"); continue
        r = paired(a, b); print(f"  {nm:34s} all   n={r.get('n')} a={r.get('a', 0):.2f} b={r.get('b', 0):.2f} Δ={r.get('diff', 0):+.2f} p={r.get('p', 1):.4f}")
        for sc, snm in SCEN.items():
            r = paired(a, b, lambda x, sc=sc: x["scenario"] == sc)
            if r.get("n", 0) >= 10: print(f"      {sc} {snm:13s} n={r['n']:4d} a={r['a']:.2f} b={r['b']:.2f} Δ={r['diff']:+.2f} p={r['p']:.4f}")
        for s in (1, 2, 3):
            r = paired(a, b, lambda x, s=s: x["source"] == s)
            if r.get("n", 0) >= 10: print(f"      source{s}          n={r['n']:4d} a={r['a']:.2f} b={r['b']:.2f} Δ={r['diff']:+.2f} p={r['p']:.4f}")


def zeroshot(out):
    seeds = ["", "_s200", "_s300", "_s400", "_s500"]
    tw, ntw = pooled([f"CoTAF_TW_L{s}" for s in seeds], "err"); hy, nhy = pooled([f"CoTAM_HYB4_L{s}" for s in seeds], "err")
    soup, _ = pooled(["CoTAM_HYB4_SOUP_L"], "err"); pos, _ = pooled(["POS"], "err"); chrom, _ = pooled(["CHROM"], "err")
    print(f"제로샷: TW 시드 {ntw}, HYB4 시드 {nhy}")
    for nm, d in (("TW pooled", tw), ("HYB4 pooled", hy), ("SOUP", soup), ("POS", pos), ("CHROM", chrom)):
        if d:
            m = lit_metrics(d, "gt"); ml = lit_metrics({v: r for v, r in d.items() if r.get("gt_lit") is not None}, "gt_lit")
            print(f"  {nm:12s} n={m['n']} seeds={m['seeds']} [정본] MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} r={m['r']:.2f} (ens MAE {m['ens_MAE']:.2f}) | [문헌GT] SD={ml['SD']:.2f} MAE={ml['MAE']:.2f} RMSE={ml['RMSE']:.2f} r={ml['r']:.2f}")
            out[f"zeroshot/{nm}"] = dict(tb=m, lit=ml)
    table({"TW→HYB4 (풀링)": (tw, hy), "HYB4→SOUP": (hy, soup), "POS→TW": (pos, tw), "POS→HYB4": (pos, hy)}, "제로샷")
    for nm, (a, b) in {"TW→HYB4": (tw, hy), "HYB4→SOUP": (hy, soup), "POS→HYB4": (pos, hy)}.items():
        if a and b:
            out[f"zeroshot_pair/{nm}"] = {"all": paired(a, b), **{sc: paired(a, b, lambda x, sc=sc: x["scenario"] == sc) for sc in SCEN}}


def cv(out):
    cells = {}
    for tag in ("BASE_L", "REAL_L", "TW_L", "HYB_L"):
        names = [f"CV_{tag}"] + [f"CV_{tag}_s{s}" for s in (200, 300)]
        d, n = pooled([x for x in names if load(x)], "err"); dl, _ = pooled([x for x in names if load(x)], "err_lit")
        if d: cells[tag] = (d, dl, n)
    print("교차검증 셀(시드 수):", {k: v[2] for k, v in cells.items()})
    for tag, (d, dl, n) in cells.items():
        m = lit_metrics(d, "gt"); ml = lit_metrics(dl, "gt_lit")
        print(f"  {tag:7s} seeds={n} n={m['n']} [정본] SD={m['SD']:.2f} MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} r={m['r']:.2f} (ens MAE {m['ens_MAE']:.2f}) | [문헌GT] SD={ml['SD']:.2f} MAE={ml['MAE']:.2f} RMSE={ml['RMSE']:.2f} r={ml['r']:.2f}")
        out[f"cv/{tag}"] = dict(seeds=n, tb=m, lit=ml, per_scenario={sc: lit_metrics({v: r for v, r in dl.items() if r["scenario"] == sc}, "gt_lit")["MAE"] for sc in SCEN})
    g = lambda t: cells.get(t, (None,))[0]
    gl = lambda t: cells.get(t, (None, None))[1]
    table({"BASE→TW (VIPL+가상얼굴 공동학습)": (g("BASE_L"), g("TW_L")), "BASE→HYB (VIPL+하이브리드 분해)": (g("BASE_L"), g("HYB_L")), "TW→HYB": (g("TW_L"), g("HYB_L")), "BASE→REAL (VIPL+실 161만)": (g("BASE_L"), g("REAL_L")), "REAL→TW (가상얼굴 42 순수 기여)": (g("REAL_L"), g("TW_L"))}, "교차검증 정본 GT")
    table({"BASE→TW [문헌GT]": (gl("BASE_L"), gl("TW_L")), "BASE→HYB [문헌GT]": (gl("BASE_L"), gl("HYB_L")), "TW→HYB [문헌GT]": (gl("TW_L"), gl("HYB_L")), "REAL→TW [문헌GT]": (gl("REAL_L"), gl("TW_L"))}, "교차검증 문헌 GT")
    for nm, (a, b) in {"BASE→TW": (g("BASE_L"), g("TW_L")), "BASE→HYB": (g("BASE_L"), g("HYB_L")), "TW→HYB": (g("TW_L"), g("HYB_L")), "BASE→REAL": (g("BASE_L"), g("REAL_L")), "REAL→TW": (g("REAL_L"), g("TW_L")), "REAL→HYB": (g("REAL_L"), g("HYB_L"))}.items():
        if a and b:
            out[f"cv_pair/{nm}"] = {"all": paired(a, b), **{sc: paired(a, b, lambda x, sc=sc: x["scenario"] == sc) for sc in SCEN},
                                    **{f"source{s}": paired(a, b, lambda x, s=s: x["source"] == s) for s in (1, 2, 3)}}
    # 문헌 참고선 (동일 프로토콜 쌍대 아님 — 학습 레시피·얼굴 검출·재표본 상이)
    out["literature"] = {"PhysFormer (CVPR22, Table 1)": dict(SD=7.74, MAE=4.97, RMSE=7.79, r=0.78), "RhythmFormer (2024, Table 1)": dict(MAE=4.51, RMSE=7.98, r=0.78),
                         "PhysFormer++": dict(MAE=4.88, RMSE=7.62, r=0.80), "Dual-GAN": dict(SD=7.63, MAE=4.93, RMSE=7.68, r=0.81), "RhythmNet": dict(SD=8.11, MAE=5.30, RMSE=8.14, r=0.76),
                         "POS (PhysFormer 표)": dict(SD=15.3, MAE=11.5, RMSE=17.2, r=0.30), "CHROM (PhysFormer 표)": dict(SD=15.1, MAE=11.4, RMSE=16.9, r=0.28),
                         "PhysNet (PhysFormer 표)": dict(SD=14.9, MAE=10.8, RMSE=14.8, r=0.20), "DeepPhys (PhysFormer 표)": dict(SD=13.6, MAE=11.0, RMSE=13.8, r=0.11)}


if __name__ == "__main__":
    out = json.load(open(f"{RES}/vipl_judge.json")) if os.path.exists(f"{RES}/vipl_judge.json") else {}
    mode = sys.argv[1] if len(sys.argv) > 1 else "zeroshot"
    (zeroshot if mode == "zeroshot" else cv)(out)
    json.dump(out, open(f"{RES}/vipl_judge.json", "w"), indent=1)
