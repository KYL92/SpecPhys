#!/usr/bin/env python3
"""[체인 52] MMPD 인트라 fold 모델 정본 집계: runs/finetune_{M}/vipl_u8/saved_test_outputs/*.pickle (VIPL-HR 로더 + mmpd_all_u8 csv 경로) → 영상 단위 HR(toolbox 프로토콜) + 속성(mmpd_trials.json)
  python scripts/agg_mmpd_cv2.py MODEL [MODEL...]   → results/agg_{MODEL}_MMPD.json (agg_mmpd.py 와 같은 행 스키마: vid, subject, light, motion, exercise, skin, gt, pred, err, ratio)
  python scripts/agg_mmpd_cv2.py --cv TAG [TAG...]  → MF1_{TAG}..MF5_{TAG} 각 held-out fold 피험자만 합산(fold 누수 assert) → results/agg_MCV5_{TAG}_MMPD.json
  python scripts/agg_mmpd_cv2.py --lit MODEL        → 문헌 분할 모델(테스트 피험자 27~33) 집계 → agg_{MODEL}_MMPD.json"""
import glob, json, os, pickle, sys
OUTDIR = None
import numpy as np
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from agg_model2 import cat_chunks, toolbox_hr, hr_for, PF
from agg_mmpd import metrics, fmt, LIGHT, MOTION
RES = f"{PF}/../results"; PFA = PF; TRIALS = json.load(open(f"{PFA}/mmpd_trials.json")); FOLDS = {int(k): set(v) for k, v in json.load(open(f"{PFA}/mmpd_folds.json")).items()} if os.path.exists(f"{PFA}/mmpd_folds.json") else {}


def agg(model):
    rows = []
    for pk in sorted(glob.glob(f"{OUTDIR}/*/saved_test_outputs/*.pickle") if OUTDIR else glob.glob(f"{PF}/runs/finetune_{model}/*/saved_test_outputs/*.pickle")):
        d = pickle.load(open(pk, "rb"))
        for vid in d["predictions"]:
            a = TRIALS[str(vid)[7:]]; p = cat_chunks(d["predictions"][vid]); l = cat_chunks(d["labels"][vid]); hp, hl = hr_for(p, d.get("label_type")), hr_for(l, d.get("label_type"))
            rows.append(dict(vid=str(vid), subject=a["subject"], light=a["light"], motion=a["motion"], exercise=a["exercise"], skin=a["skin"], gt=hl, pred=hp, err=abs(hp - hl), ratio=hp / hl))
    return rows


def report(rows, label):
    print(f"--- {label} MMPD (정본 toolbox 프로토콜, n={len(rows)})"); print(fmt(metrics(rows), "all"))
    for v, nm in MOTION.items(): print(fmt(metrics([r for r in rows if r["motion"] == v]), f"motion={nm}"))
    print(fmt(metrics([r for r in rows if r["motion"] != 1]), "motion(non-stationary)"))
    for v, nm in LIGHT.items(): print(fmt(metrics([r for r in rows if r["light"] == v]), f"light={nm}"))
    for v in (3, 4, 5, 6): print(fmt(metrics([r for r in rows if r["skin"] == v]), f"skin={v}"))


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--outdir" in args: i = args.index("--outdir"); OUTDIR = args[i + 1]; del args[i:i + 2]   # 10-21 체인 54: 평가 전용 run(runs/ev_{M}_MMPDZS) 집계
    if args and args[0] == "--cv":
        for tag in args[1:]:
            rows = []
            for k in range(1, 6):
                rk = agg(f"MF{k}_{tag}"); bad = [r for r in rk if r["subject"] not in FOLDS[k]]
                assert not bad, f"fold leak: MF{k}_{tag} has {len(bad)} videos outside fold {k}"; assert rk, f"MF{k}_{tag} empty"; rows += rk
            json.dump(rows, open(f"{RES}/agg_MCV5_{tag}_MMPD.json", "w"), indent=0); report(rows, f"MCV5_{tag}")
    elif args and args[0] == "--lit":
        for m in args[1:]:
            rows = agg(m); assert all(r["subject"] >= 27 for r in rows); json.dump(rows, open(f"{RES}/agg_{m}_MMPD.json", "w"), indent=0); report(rows, m)
    else:
        for m in args:
            rows = agg(m); json.dump(rows, open(f"{RES}/agg_{m}_MMPD.json", "w"), indent=0); report(rows, m)
