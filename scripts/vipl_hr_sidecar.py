#!/usr/bin/env python3
"""VIPL 문헌 GT(gt_HR.csv, 초당 HR) → 프레임 단위 HR 사이드카 cache/vipl_u8/{idx}_hr.npy (float32, 길이 n_out).
트레이너 RPPG_HR_LABEL_FILE=1 이 청크 i 의 HR 라벨을 hr[160i:160i+160].mean() 으로 읽는다(31 문서 트랙 L, GT 차이 해소).
gt_HR 샘플 시각은 영상 길이에 균등 배치(gt_HR n ≈ dur 초)."""
import glob, json, os, re
import numpy as np
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; CACHE = f"{PFA}/cache/vipl_u8"; RAW = "/SPECPHYS_RAW/vipl_hr_raw"


def read_csv_col(path):
    rows = [r.strip() for r in open(path).read().splitlines()]
    return np.asarray([float(r.split(",")[0]) for r in (rows[1:] if rows and not re.fullmatch(r"-?\d+(\.\d+)?", rows[0]) else rows) if r], dtype=np.float64)


if __name__ == "__main__":
    n_ok = n_skip = 0; stats = []
    for m in sorted(glob.glob(f"{CACHE}/*_meta.json")):
        j = json.load(open(m)); idx = j["index"]; out = f"{CACHE}/{idx}_hr.npy"
        if os.path.exists(out): n_skip += 1; continue
        mm = re.match(r"p(\d+)_v(\d+(?:-2)?)_s(\d)", idx); sub, v, src = mm.groups()
        hr = read_csv_col(f"{RAW}/p{sub}/v{v}/source{src}/gt_HR.csv")
        n = int(j["n_out"]); t = np.linspace(0, 1, len(hr)); f = np.linspace(0, 1, n)
        per_frame = np.interp(f, t, hr).astype(np.float32)
        np.save(out, per_frame); n_ok += 1; stats.append((len(hr), n / 30.0))
    print(f"sidecars written={n_ok} existing={n_skip}")
    if stats:
        r = np.array([a / b for a, b in stats]); print(f"gt_HR n / video sec: median {np.median(r):.3f} IQR {np.percentile(r,25):.3f}-{np.percentile(r,75):.3f}")
