#!/usr/bin/env python3
"""MMPD 정본 집계: runs/ev_{model}_MMPD_B*/ pickle → per-video HR(toolbox 프로토콜) + 속성.
출력 _workspace/data/results/agg_{model}_MMPD.json, 표: 전체/held-out(7~33)/학습분할(1~6), 조명·동작·운동·피부톤별.
사용: python agg_mmpd.py MODEL [MODEL...]  (POS/CHROM은 pos_pervideo_MMPD.json에서)"""
import glob, json, pickle, re, sys
import numpy as np
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from agg_model2 import cat_chunks, toolbox_hr, hr_for, PF
LIGHT = {1: "LED-low", 2: "LED-high", 3: "Incandescent", 4: "Nature"}; MOTION = {1: "Stationary", 2: "Rotation", 3: "Talking", 4: "Walking"}
RES = f"{PF}/../results"
def metrics(rows):
    if not rows: return None
    g = np.array([r["gt"] for r in rows]); p = np.array([r["pred"] for r in rows]); e = np.abs(g - p)
    pear = float(np.corrcoef(g, p)[0, 1]) if len(rows) > 2 and g.std() > 0 and p.std() > 0 else float("nan")
    return dict(n=len(rows), MAE=float(e.mean()), RMSE=float(np.sqrt((e ** 2).mean())), MAPE=float((e / g).mean() * 100), r=pear,
                dbl=int(sum(abs(r["pred"] / r["gt"] - 2) < 0.2 for r in rows)), hlv=int(sum(abs(r["pred"] / r["gt"] - 0.5) < 0.1 for r in rows)))
def fmt(m, label):
    return f"  {label:28s} n={m['n']:3d} MAE={m['MAE']:6.2f} RMSE={m['RMSE']:6.2f} MAPE={m['MAPE']:5.1f}% r={m['r']:5.2f} dbl={m['dbl']} hlv={m['hlv']}" if m else f"  {label:28s} (none)"
def report(rows, model):
    print(f"--- {model} MMPD (정본 toolbox 프로토콜)")
    ho = [r for r in rows if r["subject"] >= 7]; tr = [r for r in rows if r["subject"] < 7]
    print(fmt(metrics(rows), "all(1-33)")); print(fmt(metrics(ho), "held-out(7-33)")); print(fmt(metrics(tr), "train-split(1-6)"))
    for key, names in (("light", LIGHT), ("motion", MOTION)):
        for v, nm in names.items(): print(fmt(metrics([r for r in ho if r[key] == v]), f"held-out {key}={nm}"))
    for v in (1, 2): print(fmt(metrics([r for r in ho if r["exercise"] == v]), f"held-out exercise={'yes' if v==1 else 'no'}"))
    for v in (3, 4, 5, 6): print(fmt(metrics([r for r in ho if r["skin"] == v]), f"held-out skin={v}"))
def agg(model):
    rows = []
    for pk in sorted(glob.glob(f"{PF}/runs/ev_{model}_MMPD_B*/*/saved_test_outputs/*.pickle")):
        d = pickle.load(open(pk, "rb")); lt = d.get("label_type")
        for vid in d["predictions"]:
            m = re.match(r"subject(\d+)_L(\d)_MO(\d)_E(\d)_S(\d)_GE(\d)_GL(\d)_H(\d)_MA(\d)", str(vid))
            p = cat_chunks(d["predictions"][vid]); l = cat_chunks(d["labels"][vid])
            hp, hl = hr_for(p, lt), hr_for(l, lt)
            rows.append(dict(vid=str(vid), subject=int(m.group(1)), light=int(m.group(2)), motion=int(m.group(3)), exercise=int(m.group(4)),
                             skin=int(m.group(5)), gt=hl, pred=hp, err=abs(hp - hl), ratio=hp / hl))
    return rows
if __name__ == "__main__":
    for model in sys.argv[1:]:
        if model in ("POS", "CHROM"):
            src = json.load(open(f"{RES}/pos_pervideo_MMPD.json")); L = {v: k for k, v in LIGHT.items()}
            MO = {"Stationary": 1, "Stationary (after exercise)": 1, "Rotation": 2, "Talking": 3, "Walking": 4, "Watching Videos": 4}
            rows = [dict(vid=f"s{r['subject']}_p{r['trial']}", subject=r["subject"], light=L[r["light"]], motion=MO[r["motion"]],
                         exercise=1 if r["exercise"] == "True" else 2, skin=r["skin"], gt=r["gt"], pred=r[model], err=abs(r[model] - r["gt"]), ratio=r[model] / r["gt"]) for r in src]
        else:
            rows = agg(model)
        if not rows: print(f"--- {model}: no MMPD outputs yet"); continue
        report(rows, model); json.dump(rows, open(f"{RES}/agg_{model}_MMPD.json", "w"), indent=0)
