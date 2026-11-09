#!/usr/bin/env python3
"""VIPL uint8 캐시 영상별 전체 mean/std 사이드카 {idx}_u8stats.npy = [mean, std] (Standardized 입력 즉석 계산용, BaseLoader patch 29).
원본 standardized_data 는 영상 전체 픽셀 통계. 청크를 3개 중 1개만 표본(IO 절약, 추정 오차 <1%)해 float64 로 누적. 재실행 시 기존 파일 건너뜀.
사용: python scripts/vipl_u8_stats.py [CACHE_DIR=cache/vipl_u8] [STEP=3]"""
import glob, os, sys, numpy as np
C = sys.argv[1] if len(sys.argv) > 1 else "/SPECPHYS_ROOT/_workspace/data/physformer/cache/vipl_u8"
STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 3
metas = sorted(glob.glob(f"{C}/*_meta.json")); done = 0
for m in metas:
    base = m[:-10]; out = base + "_u8stats.npy"
    if os.path.exists(out): continue
    chunks = sorted(glob.glob(base + "_input*.npy"), key=lambda f: int(f.rsplit("_input", 1)[1][:-4]))
    if not chunks: continue
    s1 = s2 = 0.0; n = 0
    for f in chunks[::STEP]:
        x = np.load(f)[:160].astype(np.float64); s1 += x.sum(); s2 += (x * x).sum(); n += x.size
    mean = s1 / n; std = float(np.sqrt(max(s2 / n - mean * mean, 0.0)))
    np.save(out, np.array([mean, std], dtype=np.float64)); done += 1
    if done % 200 == 0: print(f"{done} written", flush=True)
print(f"VIPL_U8_STATS_DONE written={done} total={len(metas)}", flush=True)
