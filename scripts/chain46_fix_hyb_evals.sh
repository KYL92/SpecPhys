#!/bin/bash
# 체인 46 보정: conf_head 없는 HYB4 s100~s300 체크포인트 평가 실패(strict load) → 트레이너 패치 후 재평가, 제로샷 집계·판정 재실행
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_CACHE_U8=1 RPPG_NOISE_HEAD=1
Y=$PFA/yaml_VIPL
{
while [ ! -f /tmp/chain46_zs.done ]; do sleep 60; done
cd $ROOT
for M in CoTAM_HYB4_L CoTAM_HYB4_L_s200 CoTAM_HYB4_L_s300; do run_gated 6000 $Y/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL; done
python scripts/agg_vipl.py CoTAM_HYB4_L CoTAM_HYB4_L_s200 CoTAM_HYB4_L_s300 > /tmp/agg_vipl_zeroshot_fix.log 2>&1
python scripts/vipl_judge.py zeroshot > /tmp/chain46_judge_zs.log 2>&1; cat /tmp/chain46_judge_zs.log
echo "$(date '+%m-%d %H:%M') ZEROSHOT_FIX DONE"; touch /tmp/chain46_zsfix.done
} >> /tmp/chain46.log 2>&1
