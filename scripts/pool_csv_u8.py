#!/usr/bin/env python3
"""u8 풀 캐시 디렉터리의 청크 목록 csv (VIPL 로더 FILE_LIST_PATH 형식, VIPL 청크 없음 — 풀 단독 학습용).
  python scripts/pool_csv_u8.py CACHE_NAME OUT.csv"""
import glob, sys
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"
cn, out = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(f"{PFA}/cache/{cn}/subject*_input*.npy"))
with open(out, "w") as f:
    f.write(",input_files\n")
    for i, p in enumerate(fs): f.write(f"{i},{p}\n")
print(f"{cn}: {len(fs)} chunks -> {out}")
