#!/bin/bash
# [68 문서 §5] 체인 127 데이터 준비(pro6000 컨테이너): 세 갈래 R(클립 안 심박 변화)·W(같은 장면 짝)·K(정체성–심박 결합 대조)의 합성 80클립 생성 + 코덱 사본 10 →
# 합성 90칸만 u8 캐시 idhr_{r,w,k}_syn_u8 + 부위 분할 → 풀 목록 cotrain_allf_hyb4{r,w,k}_u8_pool.csv(3,665 청크) → 1차원 잡음 정답 noise_gt_hyb2{r,w,k}
set -u; cd /SPECPHYS_ROOT; export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2; PFA=$PWD/_workspace/data/physformer; C=$PFA/configs; L=$PFA/cache/DataFileLists
ts() { date '+%m-%d %H:%M'; }; echo "$(ts) PREP127 start"
for x in r k; do for c in train20h_tw train6hi_tw train6lo_tw train24dn train24dn2; do python scripts/build_parallel.py $C/${c}_$x.json 10 2>&1 | tail -n 1; done; done
python scripts/build_parallel.py $C/trainpair80.json 10 2>&1 | tail -n 1
mk_cmp() { local SRC=$1 D=$2; shift 2; rm -rf $PFA/$D; mkdir -p $PFA/$D; local i=0; for k in "$@"; do i=$((i+1)); local d=$PFA/$D/subject$((600+i)); mkdir -p $d; ffmpeg -v error -y -i $PFA/$SRC/subject$k/vid.avi -c:v mpeg4 -q:v 6 $d/vid.avi; cp $PFA/$SRC/subject$k/ground_truth.txt $d/; done; echo "$D n=$(ls $PFA/$D | wc -l)"; }
mk_cmp TRAIN20H_TW_R CMP_SYN_TW_R 1 3 5 7 9 11 13 15 17 19; mk_cmp TRAIN20H_TW_K CMP_SYN_TW_K 1 3 5 7 9 11 13 15 17 19; mk_cmp TRAINPAIR80 CMP_PAIR_W 1 3 5 7 9 11 13 15 17 19
python - <<'PYEOF'
import os, json
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
slots = list(range(201, 221)) + list(range(221, 227)) + list(range(231, 237)) + list(range(261, 309))
src_rk = {}
for s in range(201, 221): src_rk[s] = ("TRAIN20H_TW", s - 200)
for s in range(221, 227): src_rk[s] = ("TRAIN6HI_TW", s - 220)
for s in range(231, 237): src_rk[s] = ("TRAIN6LO_TW", s - 230)
for s in range(261, 285): src_rk[s] = ("TRAIN24DN", s)
for s in range(285, 309): src_rk[s] = ("TRAIN24DN2", s)
for arm in ("R", "W", "K"):
    P = f"{PF}/SYNC_POOL_{arm}"; os.makedirs(P, exist_ok=True)
    for s in slots:
        tgt = f"{PF}/TRAINPAIR80/subject{slots.index(s) + 1}" if arm == "W" else f"{PF}/{src_rk[s][0]}_{arm}/subject{src_rk[s][1]}"
        assert os.path.exists(tgt), tgt
        if os.path.lexists(f"{P}/subject{s}"): os.remove(f"{P}/subject{s}")
        os.symlink(tgt, f"{P}/subject{s}")
    for i in range(1, 11):
        tgt = f"{PF}/{'CMP_PAIR_W' if arm == 'W' else 'CMP_SYN_TW_' + arm}/subject{600 + i}"; assert os.path.exists(tgt), tgt
        if os.path.lexists(f"{P}/subject{600 + i}"): os.remove(f"{P}/subject{600 + i}")
        os.symlink(tgt, f"{P}/subject{600 + i}")
    print("SYNC_POOL", arm, len(os.listdir(P)))
PYEOF
for x in R W K; do lx=$(echo $x | tr A-Z a-z); [ -f $L/idhr_${lx}_syn_u8_0.0_1.0.csv ] || RPPG_MP_QUOTA=12 python scripts/pool_preprocess_u8.py SYNC_POOL_$x idhr_${lx}_syn_u8 2>&1 | grep -E "POOL_PREPROCESS_DONE|Error" | tail -2; python scripts/face_parse_sidecar.py $PFA/cache/idhr_${lx}_syn_u8 2>&1 | grep -E "^\[parse\] DONE|Error" | tail -1; done
python - <<'PYEOF'
import csv, os, re, glob, shutil, json
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"; L = f"{PF}/cache/DataFileLists"
rows = list(csv.reader(open(f"{L}/cotrain_allf_hyb4_u8_pool.csv")))
for arm in ("r", "w", "k"):
    syn = set(os.listdir(f"{PF}/SYNC_POOL_{arm.upper()}")); out = [rows[0]]; n = rep = 0
    for r in rows[1:]:
        m = re.search(r"(subject\d+)_input(\d+)", r[1]); s, k = m.group(1), m.group(2)
        p = f"{PF}/cache/idhr_{arm}_syn_u8/{s}_input{k}.npy" if s in syn else r[1]; assert os.path.exists(p), p; rep += int(s in syn); out.append([str(n), p]); n += 1
    assert n == 3665, n
    tmp = f"{L}/.cotrain_allf_hyb4{arm}_u8_pool.csv.tmp"; csv.writer(open(tmp, "w", newline="")).writerows(out); os.replace(tmp, f"{L}/cotrain_allf_hyb4{arm}_u8_pool.csv")
    d = f"{PF}/noise_gt_hyb2{arm}"; os.makedirs(d, exist_ok=True); cnt = 0
    for s in sorted(syn):
        f = os.path.realpath(f"{PF}/SYNC_POOL_{arm.upper()}/{s}") + "/noise_gt.npy"
        if os.path.exists(f): shutil.copy(f, f"{d}/{s}.npy"); cnt += 1
    print(f"POOL {arm}: {n} chunks, synthetic chunks replaced {rep}, parse {sum(os.path.exists(r[1].replace('_input', '_parse')) for r in out[1:])}, noise_gt {cnt}")
PYEOF
echo "$(ts) PREP127_DONE"; touch /tmp/prep127.done
