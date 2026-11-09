#!/usr/bin/env python3
"""VIPL-HR 2,378 영상에서 POS/CHROM HR (uint8 캐시 cache/vipl_u8 의 crop/resize 프레임 사용, eval_classical_mmpd.py 와 동일 프로토콜:
YCrCb 피부마스크 평균 → POS/CHROM → 정본 toolbox HR). GT = 라벨 파형(diff-normalized PPG) 을 동일 프로토콜로.
출력: _workspace/data/results/pos_pervideo_VIPL.json (resumable). 사용: python scripts/eval_classical_vipl.py [N_WORKERS]"""
import glob, json, os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "1")  # 워커당 1스레드(과점유 방지)
import numpy as np, cv2
if not hasattr(np, "mat"): np.mat = np.asmatrix
sys.path.insert(0, "/SPECPHYS_ROOT/repos/rPPG-Toolbox")
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from unsupervised_methods.methods.POS_WANG import POS_WANG
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
from agg_model2 import toolbox_hr
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; CACHE = f"{PFA}/cache/vipl_u8"
OUT = f"{PFA}/../results/pos_pervideo_VIPL.json"; FPS = 30


def load_video(idx):
    ins = sorted(glob.glob(f"{CACHE}/{idx}_input*.npy"), key=lambda p: int(p.rsplit("_input", 1)[1][:-4]))
    fr = np.concatenate([np.load(p)[:160] for p in ins]); lb = np.concatenate([np.load(p.replace("_input", "_label")) for p in ins])
    return fr, lb


def series_from_rgb(frames):
    acc = np.zeros(frames.shape[1:3], np.float32); sub = frames[::40]
    for f in sub:
        y = cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)
        acc += ((y[:, :, 1] >= 133) & (y[:, :, 1] <= 173) & (y[:, :, 2] >= 77) & (y[:, :, 2] <= 127)).astype(np.float32)
    m = (acc / len(sub)) > 0.5
    if m.sum() < 500:
        h, w = frames.shape[1:3]; m = np.zeros((h, w), bool); m[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    return [f.astype(np.float32)[m].mean(0).reshape(1, 1, 3) for f in frames]


def one(idx):
    fr, lb = load_video(idx)
    r = dict(vid=idx, gt=toolbox_hr(lb), n=int(len(fr)))
    ser = series_from_rgb(fr)
    for name, fn in (("POS", POS_WANG), ("CHROM", CHROME_DEHAAN)):
        bvp = np.asarray(fn(ser, FPS)).squeeze().astype(np.float64)
        r[name] = toolbox_hr(np.diff(bvp, prepend=bvp[0]))  # toolbox_hr 는 cumsum 입력 가정 → 미분 신호 전달
    return r


if __name__ == "__main__":
    from multiprocessing import Pool
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {r["vid"] for r in rows}
    idxs = sorted(json.load(open(m))["index"] for m in glob.glob(f"{CACHE}/*_meta.json"))
    todo = [i for i in idxs if i not in done]
    print(f"VIPL classical: total={len(idxs)} done={len(done)} todo={len(todo)}", flush=True)
    with Pool(nw) as pool:
        for k, r in enumerate(pool.imap_unordered(one, todo, chunksize=4)):
            rows.append(r)
            if k % 50 == 0:
                json.dump(rows, open(OUT, "w"), indent=0); print(f"{k}/{len(todo)} {r['vid']} gt={r['gt']:.1f} POS={r['POS']:.1f} CHROM={r['CHROM']:.1f}", flush=True)
    json.dump(rows, open(OUT, "w"), indent=0)
    for k in ("POS", "CHROM"):
        e = np.array([abs(r[k] - r["gt"]) for r in rows]); print(f"{k}: n={len(e)} MAE={e.mean():.2f} RMSE={np.sqrt((e**2).mean()):.2f}")
    print("CLASSICAL_VIPL_DONE")
