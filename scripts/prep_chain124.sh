#!/bin/bash
# [65 문서] 체인 124 데이터 준비(pro6000 컨테이너): 새 이중 렌더 48클립 u8 캐시 → 잡음 영상 사이드카(기존 48 + 새 48, 비트 정렬 검증) → 부위 분할 사이드카 → 풀 목록·1차원 잡음 정답 폴더
set -u; cd /SPECPHYS_ROOT; export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH
PFA=_workspace/data/physformer; L=$PFA/cache/DataFileLists; ts() { date '+%m-%d %H:%M'; }
echo "$(ts) PREP start"
[ -f $L/dr48_u8_0.0_1.0.csv ] || RPPG_MP_QUOTA=12 python scripts/pool_preprocess_u8.py TRAIN48DR dr48_u8 2>&1 | grep -E "POOL_PREPROCESS_DONE|Error|error" | tail -3
python scripts/noisevid_sidecar.py $PFA/TRAIN24DN_NV $PFA/cache/cotrain_allf_hyb4_u8 | tail -2
python scripts/noisevid_sidecar.py $PFA/TRAIN24DN2_NV $PFA/cache/cotrain_allf_hyb4_u8 | tail -2
python scripts/noisevid_sidecar.py $PFA/TRAIN48DR $PFA/cache/dr48_u8 | tail -2
python scripts/face_parse_sidecar.py /SPECPHYS_ROOT/$PFA/cache/dr48_u8 2>&1 | grep -E "^\[parse\]|Error" | tail -3
python - <<'PYEOF'
import pandas as pd, glob, os, shutil, numpy as np
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; L = f"{PFA}/cache/DataFileLists"
a = pd.read_csv(f"{L}/cotrain_allf_hyb4_u8_pool.csv"); b = pd.read_csv(f"{L}/dr48_u8_0.0_1.0.csv")
f = list(a["input_files"]) + [p if p.startswith("/") else os.path.abspath(p) for p in b["input_files"]]
assert len(set(f)) == len(f) and all(os.path.exists(p) for p in f)
pd.DataFrame({"input_files": f}).to_csv(f"{L}/cotrain_allf_hyb4_dr_u8_pool.csv"); print("pool chunks", len(a), "+", len(b), "=", len(f))
nv = sum(os.path.exists(p.replace("_input", "_noisevid")) for p in f); pa = sum(os.path.exists(p.replace("_input", "_parse")) for p in f); print("noisevid sidecars", nv, "parse sidecars", pa)
d = f"{PFA}/noise_gt_dr"; os.makedirs(d, exist_ok=True)
for p in glob.glob(f"{PFA}/noise_gt_hyb2/*.npy"): shutil.copy(p, d)
for s in sorted(glob.glob(f"{PFA}/TRAIN48DR/subject*/noise_gt.npy")): shutil.copy(s, f"{d}/{os.path.basename(os.path.dirname(s))}.npy")
print("1D noise labels", len(glob.glob(d + "/*.npy")))
# 재렌더한 기존 48클립의 1차원 정답이 기존 사이드카와 같은지
same = [np.allclose(np.load(f"{PFA}/TRAIN24DN{'' if int(os.path.basename(os.path.dirname(s))[7:]) <= 284 else '2'}_NV/{os.path.basename(os.path.dirname(s))}/noise_gt.npy"), np.load(f"{PFA}/noise_gt_hyb2/{os.path.basename(os.path.dirname(s))}.npy"), atol=1e-6) for s in glob.glob(f"{PFA}/TRAIN24DN_NV/subject*/noise_gt.npy") + glob.glob(f"{PFA}/TRAIN24DN2_NV/subject*/noise_gt.npy")]
print("re-rendered 1D labels identical", sum(same), "/", len(same))
PYEOF
echo "$(ts) PREP_DONE"; touch /tmp/prep124.done
