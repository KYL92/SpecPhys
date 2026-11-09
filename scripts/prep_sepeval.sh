#!/bin/bash
# [65 문서] 분리 시험 세트 캐시(pro6000 컨테이너): 풀 밖 합성 시험 클립(새 STRESS_SEPX24 24개 + 기존 STRESS_OFFBAND6/PMOT6/INBAND6/NEAR6·TEST_C3 20개)을
# 이름 충돌 없이 SEPEVAL/subjectNNNN 심링크로 모아 u8 캐시 sepeval_u8 + 부위 분할 사이드카 + 목록 csv. 번호: SEPX24 1001~1024, OFFBAND6 1101~, PMOT6 1201~, INBAND6 1301~, NEAR6 1401~, TEST_C3 1501~
set -u; cd /SPECPHYS_ROOT; export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH; PFA=$PWD/_workspace/data/physformer
mkdir -p $PFA/SEPEVAL
for pair in STRESS_SEPX24:1000 STRESS_OFFBAND6:1100 STRESS_PMOT6:1200 STRESS_INBAND6:1300 STRESS_NEAR6:1400 TEST_C3:1500; do S=${pair%%:*}; B=${pair#*:}
  for d in $PFA/$S/subject*; do i=${d##*subject}; ln -sfn $d $PFA/SEPEVAL/subject$((B + i)); done; done
ls $PFA/SEPEVAL | wc -l
[ -f $PFA/cache/DataFileLists/sepeval_u8_0.0_1.0.csv ] || RPPG_MP_QUOTA=8 python scripts/pool_preprocess_u8.py SEPEVAL sepeval_u8 2>&1 | grep -E "POOL_PREPROCESS_DONE|Error" | tail -2
python scripts/face_parse_sidecar.py $PFA/cache/sepeval_u8 2>&1 | grep -E "^\[parse\]|Error" | tail -2
python - <<'PYEOF'
import pandas as pd, os
L = "/SPECPHYS_ROOT/_workspace/data/physformer/cache/DataFileLists"; a = pd.read_csv(f"{L}/sepeval_u8_0.0_1.0.csv")
f = [p if p.startswith("/") else os.path.abspath(p) for p in a["input_files"]]; pd.DataFrame({"input_files": f}).to_csv(f"{L}/sepeval_u8.csv")
print("sepeval chunks", len(f), "parse", sum(os.path.exists(p.replace("_input", "_parse")) for p in f))
PYEOF
touch /tmp/prep_sepeval.done; echo SEPEVAL_DONE
