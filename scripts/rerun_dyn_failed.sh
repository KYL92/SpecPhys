#!/bin/bash
# 체인 27 보정: dyn 캐시 미존재로 실패한 축(DO_PREPROCESS 패치 후) 재평가 → CoTAF_L_DYN 재집계
set -u; source /SPECPHYS_ROOT/scripts/chain_lib.sh; cd $ROOT; M=CoTAF_L_DYN; Y=$PFA/yaml_$M
until [ -f /tmp/chain27.done ]; do sleep 60; done
for A in DLCN MCD_A MCD_B NIRP_STILL NIRP_SMALL NIRP_LARGE PURE REALTEST RTNI RTNC; do grep -q "${M}_${A}_OK" /tmp/chain24_29.log && continue; run_gated 6000 "$Y/ev_${M}_$A.yaml" /tmp/ev_${M}_$A.log ${M}_$A; drop_cache dyn_$A; done
python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; echo "$(date +%H:%M) AGG_${M}_REDONE"
