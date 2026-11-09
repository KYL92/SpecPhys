#!/usr/bin/env python3
"""MR-NIRP in-domain 5-fold(31 문서 트랙 N) 파일 목록.
  python scripts/nirp_filelist.py folds                 → _workspace/data/physformer/nirp_folds.json (피험자 1~19, 시드 0 셔플, 4/4/4/4/3)
  python scripts/nirp_filelist.py train K NIRP_CACHE [POOL_CACHE ...]
        → DataFileLists/nirp_train{K}_{NIRP_CACHE}[_POOL].csv = VIPL fold 2~5 청크(VIPL fold 1 은 항상 held-out) + 풀 청크 + NIRP 캐시 중 fold K 피험자 제외 청크
NIRP 캐시 청크 이름 subject{code}_input*.npy, code=30000+S*100+cond*10+sess → S=(code-30000)//100."""
import glob, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vipl_filelist import PFA, LISTS, fold_subjects, vipl_index, chunks_for, valid_train, write
FOLDS = f"{PFA}/nirp_folds.json"


def folds():
    if os.path.exists(FOLDS): return json.load(open(FOLDS))
    subs = list(range(1, 20)); random.Random(0).shuffle(subs)
    f = {str(k + 1): sorted(subs[k::5]) for k in range(5)}
    json.dump(f, open(FOLDS, "w"), indent=1); return f


def nirp_subject(path):
    return (int(re.search(r"subject(\d+)_input", os.path.basename(path)).group(1)) - 30000) // 100


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "folds":
        print(folds())
    elif mode == "train":
        k = sys.argv[2]; ncache = sys.argv[3]; pools = sys.argv[4:]; held = set(folds()[k])
        idx = vipl_index(); v1 = fold_subjects(1)
        files = chunks_for(sorted(i for i, s in idx.items() if s not in v1)); nv = len(files)
        for p in pools:
            pf = sorted(glob.glob(f"{PFA}/cache/{p}/*_input*.npy")); assert pf, p; files += pf
        npool = len(files) - nv
        nf = [f for f in sorted(glob.glob(f"{PFA}/cache/{ncache}/*_input*.npy")) if nirp_subject(f) not in held]
        assert nf, ncache; files += nf
        print(f"nirp fold{k} held-out subjects={sorted(held)} vipl_chunks={nv} pool_chunks={npool} nirp_chunks={len(nf)}")
        write(valid_train(files), f"nirp_train{k}_{ncache}" + ("_" + "_".join(pools) if pools else ""))
    else:
        raise SystemExit(__doc__)
