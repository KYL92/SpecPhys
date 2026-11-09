#!/usr/bin/env python3
"""MR-NIRP 학습 풀(NIRP_TRAIN) uint8 캐시 커버리지 검사·복구 준비.
  python scripts/nirp_cache_check.py            → 누락 피험자 수 출력, 누락분 심볼릭 풀 NIRP_TRAIN_MISS 갱신, 완비면 exit 0 / 누락 있으면 exit 1
  python scripts/nirp_cache_check.py --csv      → 캐시 전체 청크로 DataFileLists/nirp_train_u8_0.0_1.0.csv 재작성(부분 전처리 후 호출)
10-14 사고: 3 워커 전처리 중 메모리 압박으로 긴(주행) 영상 110/184 피험자 누락 → 단일 워커로 누락분만 재처리."""
import glob, os, sys, collections
import pandas as pd
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"
POOL = f"{PFA}/NIRP_TRAIN"; MISS = f"{PFA}/NIRP_TRAIN_MISS"; CACHE = f"{PFA}/cache/nirp_train_u8"; LISTS = f"{PFA}/cache/DataFileLists"
files = sorted(glob.glob(f"{CACHE}/*_input*.npy"))
if "--csv" in sys.argv:
    os.makedirs(LISTS, exist_ok=True); pd.DataFrame(files, columns=["input_files"]).to_csv(f"{LISTS}/nirp_train_u8_0.0_1.0.csv")
    print(f"csv rewritten n_chunks={len(files)}"); sys.exit(0)
have = collections.Counter(os.path.basename(f).split("_input")[0] for f in files)
subs = [d for d in sorted(os.listdir(POOL)) if os.path.exists(f"{POOL}/{d}/ground_truth.txt")]
missing = []
for d in subs:
    n = int(open(f"{POOL}/{d}/source.txt").read().split("n=")[1].split()[0])
    if have.get(d, 0) < n // 160 - 1: missing.append(d)
print(f"NIRP cache: pool={len(subs)} cached={len(have)} missing={len(missing)} chunks={len(files)}")
if os.path.isdir(MISS):
    for l in os.listdir(MISS): os.unlink(f"{MISS}/{l}")
else: os.makedirs(MISS)
for d in missing: os.symlink(f"{POOL}/{d}", f"{MISS}/{d}")
sys.exit(1 if missing else 0)
