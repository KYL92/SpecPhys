#!/bin/bash
# [68 문서 §4] HRSWEEP40 생성(pro6000 컨테이너) → u8 캐시 hrsweep_u8 + 부위 분할 사이드카 + csv
set -u; cd /SPECPHYS_ROOT; export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2; PFA=$PWD/_workspace/data/physformer
python scripts/build_parallel.py $PFA/configs/hrsweep40.json 10 2>&1 | tail -n 1
[ -f $PFA/cache/DataFileLists/hrsweep_u8_0.0_1.0.csv ] || RPPG_MP_QUOTA=8 python scripts/pool_preprocess_u8.py HRSWEEP40 hrsweep_u8 2>&1 | grep -E "POOL_PREPROCESS_DONE|Error" | tail -2
python scripts/face_parse_sidecar.py $PFA/cache/hrsweep_u8 2>&1 | grep -E "^\[parse\]|Error" | tail -2
python -c "
import pandas as pd, os; L='$PFA/cache/DataFileLists'; a=pd.read_csv(L+'/hrsweep_u8_0.0_1.0.csv'); f=[p if p.startswith('/') else os.path.abspath(p) for p in a['input_files']]; pd.DataFrame({'input_files': f}).to_csv(L+'/hrsweep_u8.csv'); print('hrsweep chunks', len(f), 'parse', sum(os.path.exists(p.replace('_input','_parse')) for p in f))"
python - <<'PYEOF'
import json, numpy as np
P = "/SPECPHYS_ROOT/_workspace/data/physformer"
for s in json.load(open(f"{P}/configs/hrsweep40.json"))["subjects"][:10]:
    m = json.load(open(f"{P}/HRSWEEP40/subject{s['id']}/meta.json")); print("target", s["bpm"], "achieved", m["achieved_bpm"])
PYEOF
touch /tmp/prep_hrsweep.done; echo HRSWEEP_DONE
