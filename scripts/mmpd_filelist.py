#!/usr/bin/env python3
"""[체인 52, 34 문서] MMPD 피험자 배타 5-fold 목록(uint8 캐시 cache/mmpd_all_u8, 전 33명 660 클립).
  python scripts/mmpd_filelist.py folds                 → mmpd_folds.json (33명 → 7/7/7/6/6, 피부톤 층화 round-robin, 시드 0 고정) + 캐시 커버리지 검사
  python scripts/mmpd_filelist.py test K                → DataFileLists/mmpd_test{K}.csv (held-out 피험자 전 트라이얼)
  python scripts/mmpd_filelist.py train K [POOL ...]    → DataFileLists/mmpd_train{K}[_POOL].csv (나머지 4 fold 전 트라이얼 [+ 풀 u8 캐시])
  python scripts/mmpd_filelist.py lit [POOL ...]        → mmpd_lit_train[_POOL].csv (피험자 1~23) / mmpd_lit_test.csv (27~33): 문헌 7:1:2 순차 분할 참고 행"""
import glob, json, os, random, sys
import numpy as np, pandas as pd
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; CACHE = f"{PFA}/cache/mmpd_all_u8"; LISTS = f"{PFA}/cache/DataFileLists"
FOLDS_JSON = f"{PFA}/mmpd_folds.json"; TRIALS = json.load(open(f"{PFA}/mmpd_trials.json"))


def subject_of(path):
    return TRIALS[os.path.basename(path).split("_input")[0][7:]]["subject"]


def chunks_for(subjects):
    return sorted(f for f in glob.glob(f"{CACHE}/subject7*_input*.npy") if subject_of(f) in subjects)


def valid_train(files):
    keep, drop = [], []
    for f in files:
        l = np.load(f.replace("_input", "_label"))
        (keep if (np.isfinite(l).all() and l.std() > 0.1 and l.shape[0] == 160) else drop).append(f)  # 10-17: std 임계 0.1 (근사 상수 PPG 구간 → NegPearson NaN → 가중치 오염, MF1_BASE_L 사고)
    if drop: print(f"  filter: dropped {len(drop)} chunks: {[os.path.basename(d) for d in drop[:6]]}")
    return keep


def write(files, name):
    os.makedirs(LISTS, exist_ok=True); out = f"{LISTS}/{name}.csv"
    pd.DataFrame(files, columns=["input_files"]).to_csv(out); print(f"{out} n_chunks={len(files)}"); return out


def folds():
    if os.path.exists(FOLDS_JSON): return {int(k): set(v) for k, v in json.load(open(FOLDS_JSON)).items()}
    skin = {}
    for v in TRIALS.values(): skin[v["subject"]] = v["skin"]
    rng = random.Random(0); order = []
    for sk in sorted(set(skin.values())):
        subs = sorted(s for s in skin if skin[s] == sk); rng.shuffle(subs); order += subs
    F = {k: [] for k in range(1, 6)}
    for i, s in enumerate(order): F[i % 5 + 1].append(s)
    json.dump({k: sorted(v) for k, v in F.items()}, open(FOLDS_JSON, "w"), indent=1); print("wrote", FOLDS_JSON, {k: sorted(v) for k, v in F.items()})
    return {k: set(v) for k, v in F.items()}


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "folds":
        F = folds(); have = {os.path.basename(f).split("_input")[0] for f in glob.glob(f"{CACHE}/subject7*_input*.npy")}
        miss = [sid for sid in TRIALS if f"subject{sid}" not in have]
        print(f"cache clips {len(have)}/{len(TRIALS)} missing {len(miss)} {miss[:8]}"); sys.exit(1 if miss else 0)
    elif mode == "test":
        k = int(sys.argv[2]); F = folds(); write(chunks_for(F[k]), f"mmpd_test{k}")
    elif mode == "train":
        k = int(sys.argv[2]); pools = sys.argv[3:]; F = folds(); subs = set().union(*[F[j] for j in F if j != k])
        files = chunks_for(subs); nv = len(files)
        for p in pools:
            pf = sorted(glob.glob(f"{PFA}/cache/{p}/*_input*.npy")); assert pf, p; files += pf
        print(f"fold{k} train subjects={len(subs)} mmpd_chunks={nv} pool_chunks={len(files)-nv}")
        write(valid_train(files), f"mmpd_train{k}" + ("_" + "_".join(pools) if pools else ""))
    elif mode == "lit":
        pools = sys.argv[2:]; files = chunks_for(set(range(1, 24))); nv = len(files)
        for p in pools:
            pf = sorted(glob.glob(f"{PFA}/cache/{p}/*_input*.npy")); assert pf, p; files += pf
        print(f"lit train subjects 1-23 mmpd_chunks={nv} pool_chunks={len(files)-nv}")
        write(valid_train(files), "mmpd_lit_train" + ("_" + "_".join(pools) if pools else "")); write(chunks_for(set(range(27, 34))), "mmpd_lit_test")
    else:
        raise SystemExit(__doc__)
