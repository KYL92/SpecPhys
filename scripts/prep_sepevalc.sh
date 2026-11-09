#!/bin/bash
# [66 문서 §7, 탐색] 깨끗한 배경 분리 시험 세트(pro6000 컨테이너): 44클립 생성(맥파 마스크 = YCrCb ∧ 얼굴 부위 분할 피부) → 배경 누출 점검 → SEPEVALC 심링크 → u8 캐시 sepevalc_u8 + 부위 분할 사이드카 + csv
set -u; cd /SPECPHYS_ROOT; export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2; PFA=$PWD/_workspace/data/physformer; C=$PFA/configs
for c in stress_sepx24 stress_offband6 stress_inband6 stress_near6 stress_pmot6 test_C3; do python scripts/build_parallel.py $C/${c}_clean.json 8 2>&1 | tail -n 1; done
python - <<'PYEOF'
import cv2, glob, numpy as np
P = "/SPECPHYS_ROOT/_workspace/data/physformer"
for st in ("STRESS_SEPX24", "STRESS_OFFBAND6", "STRESS_INBAND6", "STRESS_NEAR6", "STRESS_PMOT6", "TEST_C3"):
    for suf in ("", "_CLEAN"):
        fr = []
        for d in sorted(glob.glob(f"{P}/{st}{suf}/subject*")):
            m = cv2.imread(d + "/amplitude_map.png", 0).astype(float) / 255; w = m.shape[1]; b = int(0.12 * w)
            fr.append((np.concatenate([m[:, :b].ravel(), m[:, -b:].ravel(), m[:b, :].ravel()]) > 0.2).mean())
        print(f"{st}{suf}: border pulse-mask fraction mean {np.mean(fr):.3f} max {np.max(fr):.3f} (n={len(fr)})")
PYEOF
mkdir -p $PFA/SEPEVALC
for pair in STRESS_SEPX24_CLEAN:1000 STRESS_OFFBAND6_CLEAN:1100 STRESS_PMOT6_CLEAN:1200 STRESS_INBAND6_CLEAN:1300 STRESS_NEAR6_CLEAN:1400 TEST_C3_CLEAN:1500; do S=${pair%%:*}; B=${pair#*:}
  for d in $PFA/$S/subject*; do i=${d##*subject}; ln -sfn $d $PFA/SEPEVALC/subject$((B + i)); done; done
[ -f $PFA/cache/DataFileLists/sepevalc_u8_0.0_1.0.csv ] || RPPG_MP_QUOTA=8 python scripts/pool_preprocess_u8.py SEPEVALC sepevalc_u8 2>&1 | grep -E "POOL_PREPROCESS_DONE|Error" | tail -2
python scripts/face_parse_sidecar.py $PFA/cache/sepevalc_u8 2>&1 | grep -E "^\[parse\]|Error" | tail -2
python -c "
import pandas as pd, os; L='$PFA/cache/DataFileLists'; a=pd.read_csv(L+'/sepevalc_u8_0.0_1.0.csv'); f=[p if p.startswith('/') else os.path.abspath(p) for p in a['input_files']]; pd.DataFrame({'input_files': f}).to_csv(L+'/sepevalc_u8.csv'); print('sepevalc chunks', len(f), 'parse', sum(os.path.exists(p.replace('_input','_parse')) for p in f))"
touch /tmp/prep_sepevalc.done; echo SEPEVALC_DONE
