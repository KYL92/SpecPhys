#!/usr/bin/env python3
"""[체인 70, 50 문서 §2b] 추가 평가 축(사후) 영상 단위 MAE — 정본 읽기(agg_model2.hr_for, pickle label_type 기준), 시드 접미 _s200/_s300 자동 풀링.
  python scripts/extra_axes.py --axes NIRP_GSTILL NIRP_GSMALL ... --models CoTAF_TW_L CoTAF_FR03_L ...
출력: 표 + results/extra_axes.json ({model: {axis: {vid: err}}}, 기존 항목과 병합)"""
import glob, json, os, pickle, sys
import numpy as np
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from agg_model2 import cat_chunks, hr_for, PF  # noqa
R = f"{PF}/../results"; OUT = f"{R}/extra_axes.json"


def per_video(model, ax):
    pk = glob.glob(f"{PF}/runs/ev_{model}_{ax}/*/saved_test_outputs/*.pickle")
    if not pk: return None
    d = pickle.load(open(pk[0], "rb")); lt = d.get("label_type"); out = {}
    for vid in d["predictions"]:
        p = cat_chunks(d["predictions"][vid]); l = cat_chunks(d["labels"][vid])
        out[str(vid)] = float(abs(hr_for(p, lt) - hr_for(l, lt)))
    return out


def pooled(model, ax):
    acc = {}; n = 0
    for m in [model] + [f"{model}_s{s}" for s in (200, 300)]:
        pv = per_video(m, ax)
        if pv is None: continue
        n += 1
        for v, e in pv.items(): acc.setdefault(v, []).append(e)
    return ({v: float(np.mean(e)) for v, e in acc.items()}, n) if acc else (None, 0)


if __name__ == "__main__":
    a = sys.argv[1:]; axes = a[a.index("--axes") + 1:a.index("--models")]; models = a[a.index("--models") + 1:]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print(f"{'모델':18s} " + " ".join(f"{x:>15s}" for x in axes) + "   (영상 단위 MAE, 시드 풀링 n)")
    for m in models:
        res.setdefault(m, {}); cells = []
        for ax in axes:
            pv, n = pooled(m, ax)
            if pv is None: cells.append(f"{'–':>15s}"); continue
            res[m][ax] = pv; cells.append(f"{np.mean(list(pv.values())):9.2f} ({n}s)")
        print(f"{m:18s} " + " ".join(cells))
    json.dump(res, open(OUT, "w"), indent=0); print(f"-> {OUT}")
