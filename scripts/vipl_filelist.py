#!/usr/bin/env python3
"""VIPL-HR uint8 캐시(cache/vipl_u8)용 파일 목록 csv 생성 (툴박스 FILE_LIST_PATH 로 직접 지정, DO_PREPROCESS=false).
  python scripts/vipl_filelist.py all                       → DataFileLists/vipl_all.csv   (2,378 영상 전체: 제로샷 평가)
  python scripts/vipl_filelist.py test  K                   → DataFileLists/vipl_test{K}.csv   (fold K 피험자)
  python scripts/vipl_filelist.py train K [--src S] [--xv] [POOL_CACHE ...]  → DataFileLists/vipl_train{K}[_srcS][_xv][_POOL].csv (fold K 제외 4-fold [소스 S 만] [검증 피험자 제외] [+ 풀 캐시 청크])
  python scripts/vipl_filelist.py valid K                   → DataFileLists/vipl_valid{K}.csv   (32 문서 §5: 학습 4-fold 에서 fold 당 3명 = 12명, vipl_valid_subjects.json 고정)
fold 정의: /SPECPHYS_RAW/vipl_hr_raw/fold/fold{K}.mat (데이터셋 제공 피험자 배타 5-fold)."""
import glob, json, os, sys
import numpy as np, pandas as pd, scipy.io as sio
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"
CACHE = f"{PFA}/cache/vipl_u8"; LISTS = f"{PFA}/cache/DataFileLists"; FOLD = "/SPECPHYS_RAW/vipl_hr_raw/fold"


def fold_subjects(k):
    return set(int(x) for x in np.asarray(sio.loadmat(f"{FOLD}/fold{k}.mat")[f"fold{k}"]).ravel().tolist())


VALID_JSON = f"{PFA}/vipl_valid_subjects.json"


def valid_subjects(k):
    """[32 문서 §5] test fold k 의 검증 피험자 12명: 나머지 4 fold 각각에서 3명(random.Random(0), 생성 후 json 고정·불변)."""
    import random
    if os.path.exists(VALID_JSON):
        return set(json.load(open(VALID_JSON))[str(k)])
    out = {}
    for kk in range(1, 6):
        rng = random.Random(0); ch = []
        for j in range(1, 6):
            if j != kk: ch += rng.sample(sorted(fold_subjects(j)), 3)
        out[str(kk)] = sorted(ch)
    json.dump(out, open(VALID_JSON, "w"), indent=1); print(f"wrote {VALID_JSON}: {out}")
    return set(out[str(k)])


def vipl_index():
    idx = {}
    for m in glob.glob(f"{CACHE}/*_meta.json"):
        j = json.load(open(m)); idx[j["index"]] = j["subject"]
    return idx


def chunks_for(indices):
    files = []
    for i in indices:
        files += glob.glob(f"{CACHE}/{i}_input*.npy")
    return sorted(files)


def valid_train(files):
    """학습 목록 전용 필터: 라벨 청크가 상수(std=0)·비유한·길이 불일치이면 제외 — NegPearson 손실이 NaN 이 되어 가중치를 오염시킴(2026-10-14 fold1 사고)."""
    keep, drop = [], []
    for f in files:
        l = np.load(f.replace("_input", "_label"))
        (keep if (np.isfinite(l).all() and l.std() > 0 and l.shape[0] == 160) else drop).append(f)
    if drop: print(f"  train-filter: dropped {len(drop)} chunks: {[os.path.basename(d) for d in drop[:8]]}")
    return keep


def write(files, name):
    os.makedirs(LISTS, exist_ok=True)
    out = f"{LISTS}/{name}.csv"
    pd.DataFrame(files, columns=["input_files"]).to_csv(out)
    print(f"{out} n_chunks={len(files)}")
    return out


if __name__ == "__main__":
    mode = sys.argv[1]; idx = vipl_index()
    if mode == "all":
        write(chunks_for(sorted(idx)), "vipl_all")
    elif mode == "test":
        k = int(sys.argv[2]); subs = fold_subjects(k)
        sel = [i for i, s in idx.items() if s in subs]
        print(f"fold{k} test subjects={len(subs)} videos={len(sel)}")
        write(chunks_for(sorted(sel)), f"vipl_test{k}")
    elif mode == "valid":
        k = int(sys.argv[2]); vs = valid_subjects(k); assert not (vs & fold_subjects(k))
        sel = [i for i, s in idx.items() if s in vs]
        print(f"fold{k} valid subjects={sorted(vs)} videos={len(sel)}")
        write(valid_train(chunks_for(sorted(sel))), f"vipl_valid{k}")
    elif mode == "train":
        args = sys.argv[2:]; src = None; xv = False
        if "--src" in args: i = args.index("--src"); src = int(args[i + 1]); del args[i:i + 2]   # 소스 필터(31 문서: 휴대폰 = 2)
        if "--xv" in args: xv = True; args.remove("--xv")   # 검증 피험자 제외(32 문서 §5)
        k = int(args[0]); subs = fold_subjects(k); pools = args[1:]
        if xv: subs = subs | valid_subjects(k)
        sel = [i for i, s in idx.items() if s not in subs and (src is None or i.endswith(f"_s{src}"))]
        files = chunks_for(sorted(sel)); nv = len(files)
        for p in pools:
            pf = sorted(glob.glob(f"{PFA}/cache/{p}/*_input*.npy")); assert pf, p; files += pf
        print(f"fold{k} train videos={len(sel)} vipl_chunks={nv} pool_chunks={len(files)-nv}" + (" (valid subjects excluded)" if xv else ""))
        write(valid_train(files), f"vipl_train{k}" + (f"_src{src}" if src else "") + ("_xv" if xv else "") + ("_" + "_".join(pools) if pools else ""))
    else:
        raise SystemExit(__doc__)
