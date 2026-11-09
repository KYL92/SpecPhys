#!/usr/bin/env python3
"""[체인 73 레인 D, 55 문서 후보 ②] MR-NIRP 전용 5-fold 목록 — uint8 캐시 cache/nirp_train_u8 의 청크만으로(VIPL·풀 청크 없이) 2단계 적응용 train/test csv 를 만든다.
  python scripts/nirp_fold_lists.py            → DataFileLists/nirp_only_train{K}.csv (fold K 피험자 제외) / nirp_only_test{K}.csv (fold K 피험자), K=1..5 (nirp_folds.json)
청크 이름 subject{code}_input*.npy, code = 30000 + S*100 + cond*10 + sess → S = (code − 30000) // 100 (nirp_filelist.py 와 동일)."""
import glob, json, os, re
import pandas as pd

PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; CACHE = f"{PFA}/cache/nirp_train_u8"; LISTS = f"{PFA}/cache/DataFileLists"
FOLDS = {int(k): set(v) for k, v in json.load(open(f"{PFA}/nirp_folds.json")).items()}


def subject_of(path):
    code = int(re.search(r"subject(\d+)_input", os.path.basename(path)).group(1)); return (code - 30000) // 100


if __name__ == "__main__":
    files = sorted(glob.glob(f"{CACHE}/*_input*.npy")); assert files, f"no chunks in {CACHE}"
    subs = sorted({subject_of(f) for f in files}); print(f"cache chunks={len(files)} subjects={len(subs)} {subs}")
    os.makedirs(LISTS, exist_ok=True)
    for k, held in FOLDS.items():
        tr = [f for f in files if subject_of(f) not in held]; te = [f for f in files if subject_of(f) in held]
        assert tr and te, (k, len(tr), len(te))
        pd.DataFrame(tr, columns=["input_files"]).to_csv(f"{LISTS}/nirp_only_train{k}.csv"); pd.DataFrame(te, columns=["input_files"]).to_csv(f"{LISTS}/nirp_only_test{k}.csv")
        print(f"fold {k}: held-out {sorted(held)} train chunks={len(tr)} test chunks={len(te)}")
