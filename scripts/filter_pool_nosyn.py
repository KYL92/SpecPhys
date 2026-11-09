#!/usr/bin/env python3
"""[64 문서] 풀 목록에서 생성 얼굴 클립 90개(주입 합성 201~220, 221~226, 231~236, 261~308 + 코덱 사본 601~610)를 뺀 목록을 만든다. 번호 범위로 세지 않고 심볼릭 링크 출처로 확인한다.
  python scripts/filter_pool_nosyn.py → cache/DataFileLists/cotrain_allf_hyb4_nosyn_u8_pool.csv"""
import csv, os, re
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"; L = f"{PF}/cache/DataFileLists"; P = f"{PF}/COTRAIN_ALLF_HYB4"
SYN = {"TRAIN20H_TW", "TRAIN6HI_TW", "TRAIN6LO_TW", "TRAIN24DN", "TRAIN24DN2", "CMP_SYN_TW"}
rm = {s for s in os.listdir(P) if s.startswith("subject") and os.path.realpath(f"{P}/{s}").split("/")[-2] in SYN}; assert len(rm) == 90, len(rm)
rows = list(csv.reader(open(f"{L}/cotrain_allf_hyb4_u8_pool.csv"))); out = [rows[0]]; n = 0
for r in rows[1:]:
    name = re.search(r"(subject\d+)_input", r[1]).group(1)
    if name not in rm: out.append([str(n), r[1]]); n += 1
assert n == 3125, n
tmp = f"{L}/.cotrain_allf_hyb4_nosyn_u8_pool.csv.tmp"; csv.writer(open(tmp, "w", newline="")).writerows(out); os.replace(tmp, f"{L}/cotrain_allf_hyb4_nosyn_u8_pool.csv"); print("nosyn list", n, "chunks;", len({re.search(r"(subject\d+)_input", r[1]).group(1) for r in out[1:]}), "clips")
