#!/usr/bin/env python3
"""[45 문서 ②] MMPD fold K 학습 목록의 피험자 부분집합(FRAC %) csv: python scripts/mmpd_subset_csv.py K FRAC → DataFileLists/mmpd_train{K}_sub{FRAC}.csv
학습 피험자(다른 4 fold)를 고정 시드 63 으로 섞어 앞에서 round(FRAC%) 명 선택(25% ⊂ 50% 중첩 보장)."""
import json, sys, random
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; L = f"{PFA}/cache/DataFileLists"
K, FRAC = int(sys.argv[1]), int(sys.argv[2]); folds = json.load(open(f"{PFA}/mmpd_folds.json")); trials = json.load(open(f"{PFA}/mmpd_trials.json"))
train_subj = sorted(s for f, ss in folds.items() if int(f) != K for s in ss); rng = random.Random(63); order = train_subj[:]; rng.shuffle(order)
n = max(2, round(len(order) * FRAC / 100)); keep = set(order[:n])
rows = open(f"{L}/mmpd_train{K}.csv").read().splitlines(); out = [rows[0]]; i = 0
for r in rows[1:]:
    name = r.split("/")[-1].split("_input")[0].replace("subject", ""); subj = trials[name]["subject"]
    if subj in keep: out.append(f"{i},{r.split(',',1)[1]}"); i += 1
open(f"{L}/mmpd_train{K}_sub{FRAC}.csv", "w").write("\n".join(out) + "\n"); print(f"fold {K} {FRAC}%: subjects {sorted(keep)} chunks {i}")
