#!/usr/bin/env python3
"""[체인 73 레인 D, 55 문서 후보 ②] fold 모델들의 축별 영상 행을 held-out 피험자만 모아 가상 CV 모델로 합친다(agg_nirp_cv.py 의 5-fold·임의 태그 일반화).
  python scripts/agg_fold_axes.py --folds nirp_folds.json --pattern 'NR{k}_NF_L' --tag NF_L [--axes NIRP_STILL NIRP_SMALL NIRP_LARGE] [--out NRCV]
    → results/agg_{OUT}_{TAG}.json = {axis: rows}, 각 축의 영상(vid = 피험자 번호)은 자기 held-out fold 모델의 행만(누수 assert) + 축별 MAE 출력.
  예: 앵커  python scripts/agg_fold_axes.py --folds nirp_folds.json --pattern 'NF{k}_HYBN_L' --tag HYBN_L   (체인 47 PhysFormer in-domain)"""
import json, os, sys
import numpy as np

PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; R = f"{PFA}/../results"


def main():
    a = sys.argv[1:]
    def opt(name, default=None, n=1):
        if name in a:
            i = a.index(name); v = a[i + 1:i + 1 + n]; del a[i:i + 1 + n]; return v[0] if n == 1 else v
        return default
    folds = opt("--folds", "nirp_folds.json"); pattern = opt("--pattern"); tag = opt("--tag"); out = opt("--out", "NRCV")
    axes = None
    if "--axes" in a:
        i = a.index("--axes"); axes = a[i + 1:]; del a[i:]
    axes = axes or ["NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE"]
    F = {int(k): set(int(x) for x in v) for k, v in json.load(open(folds if folds.startswith("/") else f"{PFA}/{folds}")).items()}
    comb = {ax: [] for ax in axes}; seen = {ax: set() for ax in axes}
    for k, held in F.items():
        m = pattern.format(k=k); f = f"{R}/agg_{m}.json"
        if not os.path.exists(f): print(f"missing {f}"); continue
        d = json.load(open(f))
        for ax in axes:
            rows = [r for r in d.get(ax, []) if int(r["vid"]) in held]
            for r in rows:
                assert int(r["vid"]) not in seen[ax], f"duplicate vid {r['vid']} in {ax}"; seen[ax].add(int(r["vid"]))
            comb[ax].extend(rows)
    json.dump(comb, open(f"{R}/agg_{out}_{tag}.json", "w"), indent=1)
    for ax, rows in comb.items():
        e = np.array([r["err"] for r in rows]) if rows else np.array([np.nan])
        print(f"{ax:11s} n={len(rows):3d} MAE={np.nanmean(e):6.2f} dbl={sum(abs(r['ratio'] - 2) < 0.2 for r in rows)} hlv={sum(abs(r['ratio'] - 0.5) < 0.1 for r in rows)}")
    print(f"-> {R}/agg_{out}_{tag}.json")


if __name__ == "__main__":
    main()
