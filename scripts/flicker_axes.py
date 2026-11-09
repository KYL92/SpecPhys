#!/usr/bin/env python3
"""[체인 68, 49 문서 §2] 띠 밖 플리커 축 RTNI(1.1 Hz)·RTNI17(1.7 Hz)·RTNI24(2.4 Hz) 영상 단위 MAE — 정본 읽기(agg_model2.hr_for, pickle label_type 기준).
  python scripts/flicker_axes.py MODEL [MODEL...]     (시드 접미 _s200/_s300 는 자동 풀링: 영상별 오차 평균)
출력: 표 + results/flicker_axes.json ({model: {axis: {vid: err}}})"""
import glob, json, os, pickle, sys
import numpy as np
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from agg_model2 import cat_chunks, hr_for, PF  # noqa
R = f"{PF}/../results"; AXES = ["RTNI", "RTNI17", "RTNI24"]


def per_video(model, ax):
    pk = glob.glob(f"{PF}/runs/ev_{model}_{ax}/*/saved_test_outputs/*.pickle")
    if not pk: return None
    d = pickle.load(open(pk[0], "rb")); lt = d.get("label_type"); out = {}
    for vid in d["predictions"]:
        p = cat_chunks(d["predictions"][vid]); l = cat_chunks(d["labels"][vid])
        out[str(vid)] = float(abs(hr_for(p, lt) - hr_for(l, lt)))
    return out


def pooled(model, ax):
    seeds = [model] + [f"{model}_s{s}" for s in (200, 300)]
    acc = {}
    n = 0
    for m in seeds:
        pv = per_video(m, ax)
        if pv is None: continue
        n += 1
        for v, e in pv.items(): acc.setdefault(v, []).append(e)
    return ({v: float(np.mean(e)) for v, e in acc.items()}, n) if acc else (None, 0)


if __name__ == "__main__":
    models = sys.argv[1:]; res = {}
    print(f"{'모델':22s} " + " ".join(f"{a:>14s}" for a in AXES) + "   (영상 단위 MAE, 시드 풀링 n)")
    for m in models:
        res[m] = {}; cells = []
        for ax in AXES:
            pv, n = pooled(m, ax)
            if pv is None: cells.append(f"{'–':>14s}"); continue
            res[m][ax] = pv; cells.append(f"{np.mean(list(pv.values())):8.2f} ({n}s)")
        print(f"{m:22s} " + " ".join(cells))
    json.dump(res, open(f"{R}/flicker_axes.json", "w"), indent=0)
    print(f"-> {R}/flicker_axes.json")
