#!/bin/bash
# [67 문서] 체인 126 데이터 준비(pro6000 컨테이너): ① 결정성 점검(세트마다 첫 피험자를 원래 설정으로 다시 렌더 → 원래 vid.avi 와 프레임 비교)
# ② 생성 얼굴 학습 클립 80개를 skin_mask_parser 로 다시 생성(_C) ③ 코덱 사본 10개(mpeg4 q 6, TRAIN20H_TW_C 홀수 1~19 → 601~610) ④ 배경 누출 점검
# ⑤ 풀 COTRAIN_ALLF_HYB4C(실제 클립 링크 그대로, 생성 얼굴 90개만 _C 로) ⑥ 생성 얼굴 90개만 u8 캐시 hyb4c_syn_u8 + 부위 분할 사이드카 ⑦ 풀 목록(3,665 청크) ⑧ 1차원 잡음 정답 noise_gt_hyb2c
set -u; cd /SPECPHYS_ROOT; export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2; PFA=$PWD/_workspace/data/physformer; C=$PFA/configs; L=$PFA/cache/DataFileLists
ts() { date '+%m-%d %H:%M'; }; echo "$(ts) PREP126 start"
for c in train20h_tw train6hi_tw train6lo_tw; do python scripts/build_parallel.py $C/${c}_detchk.json 1 > /dev/null 2>&1; done
python - <<'PYEOF'
import cv2, numpy as np, json
P = "/SPECPHYS_ROOT/_workspace/data/physformer"
def rd(p):
    cap = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    return np.stack(fr)
for st in ("TRAIN20H_TW", "TRAIN6HI_TW", "TRAIN6LO_TW"):
    a, b = rd(f"{P}/{st}/subject1/vid.avi"), rd(f"{P}/{st}_DETCHK/subject1/vid.avi")
    print(f"DETCHK {st}/subject1: same shape {a.shape == b.shape}, identical {bool(a.shape == b.shape and np.array_equal(a, b))}")
PYEOF
for c in train20h_tw train6hi_tw train6lo_tw train24dn train24dn2; do python scripts/build_parallel.py $C/${c}_c.json 10 2>&1 | tail -n 1; done
D=$PFA/CMP_SYN_TW_C; rm -rf $D; mkdir -p $D; i=0; for k in 1 3 5 7 9 11 13 15 17 19; do i=$((i+1)); d=$D/subject$((600+i)); mkdir -p $d; ffmpeg -v error -y -i $PFA/TRAIN20H_TW_C/subject$k/vid.avi -c:v mpeg4 -q:v 6 $d/vid.avi; cp $PFA/TRAIN20H_TW_C/subject$k/ground_truth.txt $d/; done; echo "CMP_SYN_TW_C n=$(ls $D | wc -l)"
python - <<'PYEOF'
import cv2, glob, numpy as np, os
P = "/SPECPHYS_ROOT/_workspace/data/physformer"
for st in ("TRAIN20H_TW", "TRAIN6HI_TW", "TRAIN6LO_TW", "TRAIN24DN", "TRAIN24DN2"):
    out = []
    for suf in ("", "_C"):
        fr = []
        for d in sorted(glob.glob(f"{P}/{st}{suf}/subject*")):
            m = cv2.imread(d + "/amplitude_map.png", 0).astype(float) / 255; w = m.shape[1]; b = int(0.12 * w)
            fr.append((np.concatenate([m[:, :b].ravel(), m[:, -b:].ravel(), m[:b, :].ravel()]) > 0.2).mean())
        out.append(f"{suf or 'orig'} mean {np.mean(fr):.2f} >0.3 {int(np.sum(np.array(fr) > 0.3))}/{len(fr)}")
    print("LEAK", st, " | ".join(out))
PYEOF
python - <<'PYEOF'
import os, csv, re, glob
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"; P0 = f"{PF}/COTRAIN_ALLF_HYB4"; P1 = f"{PF}/COTRAIN_ALLF_HYB4C"; PS = f"{PF}/SYNC_POOL"
SYN = {"TRAIN20H_TW", "TRAIN6HI_TW", "TRAIN6LO_TW", "TRAIN24DN", "TRAIN24DN2", "CMP_SYN_TW"}
os.makedirs(P1, exist_ok=True); os.makedirs(PS, exist_ok=True); syn = []
for s in sorted(os.listdir(P0)):
    if not s.startswith("subject"): continue
    real = os.path.realpath(f"{P0}/{s}"); src, sub = real.split("/")[-2], real.split("/")[-1]
    tgt = f"{PF}/{src}_C/{sub}" if src in SYN else real
    assert os.path.exists(tgt), tgt
    for P in ([P1, PS] if src in SYN else [P1]):
        if os.path.lexists(f"{P}/{s}"): os.remove(f"{P}/{s}")
        os.symlink(tgt, f"{P}/{s}")
    if src in SYN: syn.append(s)
print("POOL HYB4C", len(os.listdir(P1)), "clips, synthetic replaced", len(syn))
PYEOF
[ -f $L/hyb4c_syn_u8_0.0_1.0.csv ] || RPPG_MP_QUOTA=12 python scripts/pool_preprocess_u8.py SYNC_POOL hyb4c_syn_u8 2>&1 | grep -E "POOL_PREPROCESS_DONE|Error" | tail -2
python scripts/face_parse_sidecar.py $PFA/cache/hyb4c_syn_u8 2>&1 | grep -E "^\[parse\]|Error" | tail -2
python - <<'PYEOF'
import csv, os, re, glob, shutil
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"; L = f"{PF}/cache/DataFileLists"
syn = set(os.listdir(f"{PF}/SYNC_POOL")); rows = list(csv.reader(open(f"{L}/cotrain_allf_hyb4_u8_pool.csv"))); out = [rows[0]]; n = rep = 0
for r in rows[1:]:
    name = re.search(r"(subject\d+)_input(\d+)", r[1]); s, k = name.group(1), name.group(2)
    p = f"{PF}/cache/hyb4c_syn_u8/{s}_input{k}.npy" if s in syn else r[1]
    assert os.path.exists(p), p; rep += int(s in syn); out.append([str(n), p]); n += 1
assert n == 3665, n
tmp = f"{L}/.cotrain_allf_hyb4c_u8_pool.csv.tmp"; csv.writer(open(tmp, "w", newline="")).writerows(out); os.replace(tmp, f"{L}/cotrain_allf_hyb4c_u8_pool.csv")
print("POOL CSV", n, "chunks, synthetic chunks replaced", rep, "parse sidecars", sum(os.path.exists(r[1].replace("_input", "_parse")) for r in out[1:]))
d = f"{PF}/noise_gt_hyb2c"; os.makedirs(d, exist_ok=True)
for st in ("TRAIN24DN_C", "TRAIN24DN2_C"):
    for f in glob.glob(f"{PF}/{st}/subject*/noise_gt.npy"): shutil.copy(f, f"{d}/{os.path.basename(os.path.dirname(f))}.npy")
print("noise_gt_hyb2c", len(glob.glob(d + "/*.npy")))
PYEOF
echo "$(ts) PREP126_DONE"; touch /tmp/prep126.done
