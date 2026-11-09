#!/usr/bin/env python3
"""[체인 74, 57 문서] 임의 uint8 캐시(풀·축)의 영상별 전체 mean/std 사이드카 {name}_u8stats.npy = [mean, std] 생성 — Standardized 입력 모델(RhythmFormer)의 u8 즉석 표준화용(BaseLoader patch 29).
vipl_u8_stats.py 는 *_meta.json(VIPL 전처리 산출)을 키로 쓰므로 pool_preprocess_u8 / nirp_pool_stream 캐시에는 쓸 수 없다 → 여기서는 *_input*.npy 이름에서 영상을 모은다.
  python scripts/u8_stats_any.py CACHE_DIR [STEP=3]     (청크 STEP 개 중 1개 표본, float64 누적; 기존 사이드카는 건너뜀)"""
import glob, os, re, sys
import numpy as np

C = sys.argv[1]; STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 3
files = sorted(glob.glob(f"{C}/*_input*.npy")); vids = {}
for f in files:
    m = re.match(r"(.*)_input(\d+)\.npy$", os.path.basename(f)); vids.setdefault(m.group(1), []).append((int(m.group(2)), f))
done = skip = 0
for name, lst in sorted(vids.items()):
    out = f"{C}/{name}_u8stats.npy"
    if os.path.exists(out): skip += 1; continue
    lst.sort(); s1 = s2 = 0.0; n = 0
    for i, (_, f) in enumerate(lst):
        if i % STEP: continue
        x = np.load(f).astype(np.float64); s1 += x.sum(); s2 += (x * x).sum(); n += x.size
    mean = s1 / n; std = float(np.sqrt(max(s2 / n - mean * mean, 1e-12)))
    np.save(out, np.array([mean, std], np.float64)); done += 1
print(f"u8stats: written {done}, skipped {skip}, videos {len(vids)} in {C}")
