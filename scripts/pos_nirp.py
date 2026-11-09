#!/usr/bin/env python3
"""MR-NIRP 3축(STILL/SMALL/LARGE)에서 POS per-clip HR — 움직임 분할 쌍대 비교용 기준선.
출력: results/agg_POS_NIRP.json  rows {axis, vid, gt, pred, err}
GT/pred 모두 정본 프로토콜(toolbox_hr)로 계산해 agg_*와 비교 가능."""
import glob, importlib.util, json, os, sys
import numpy as np
import cv2

ROOT = "/SPECPHYS_ROOT"
sys.path.insert(0, f"{ROOT}/repos/rPPG-Toolbox")
if not hasattr(np, "mat"): np.mat = np.asmatrix
from unsupervised_methods.methods.POS_WANG import POS_WANG  # noqa
spec = importlib.util.spec_from_file_location("agg2", f"{ROOT}/scripts/agg_model2.py")
agg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(agg2)

def frames_of(avi):
    cap = cv2.VideoCapture(avi); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    cap.release()
    return fr

rows = []
for axis in ("NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE"):
    for d in sorted(glob.glob(f"{ROOT}/_workspace/data/physformer/{axis}/subject*")):
        vid = os.path.basename(d).replace("subject", "")
        try:
            fr = frames_of(f"{d}/vid.avi")
            if len(fr) < 90: continue
            arr = np.stack([cv2.resize(f, (72, 72)) for f in fr]).astype(np.float64)
            bvp = POS_WANG(arr[..., ::-1], 30)  # BGR→RGB
            pred = agg2.toolbox_hr(np.diff(np.asarray(bvp).ravel(), prepend=bvp[0]))
            gt_sig = np.array(open(f"{d}/ground_truth.txt").read().split("\n")[0].split(), float)
            gt = agg2.toolbox_hr(np.diff(gt_sig, prepend=gt_sig[:1]))
            rows.append(dict(axis=axis, vid=int(vid), gt=gt, pred=pred, err=abs(pred - gt)))
            print(f"{axis} s{vid}: gt {gt:.1f} pos {pred:.1f}", flush=True)
        except Exception as e:
            print(f"{axis} s{vid}: FAIL {e}", flush=True)
json.dump(rows, open(f"{ROOT}/_workspace/data/results/agg_POS_NIRP.json", "w"), indent=1)
import collections
by = collections.defaultdict(list)
for r in rows: by[r["axis"]].append(r["err"])
for a, v in by.items(): print(f"{a}: POS MAE {np.mean(v):.2f} (n={len(v)})")
print("POS_NIRP_DONE")
