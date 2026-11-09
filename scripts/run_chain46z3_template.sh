#!/bin/bash
# 체인 46z3 (기여 점검 ③, 10-17): 템플릿 파형 합성 모델 CoTAF_L 시드 200·300 의 VIPL zero-shot 평가 → 파형 뱅크(CoTAF_TW_L) 대비 3시드 쌍대
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_CACHE_U8=1
cd $ROOT
for M in CoTAF_L_s200 CoTAF_L_s300; do
  ck=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  [ -f $ck ] || { echo "missing $ck"; continue; }
  [ -f $ROOT/_workspace/data/results/agg_${M}_VIPL.json ] && { echo "skip $M"; continue; }
  python scripts/vipl_yaml.py ev $M $ck > /dev/null
  echo "$(date '+%m-%d %H:%M') EV $M VIPL start"
  run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL
  python scripts/agg_vipl.py $M 2>&1 | grep -E "^---|all "
done
echo "$(date '+%m-%d %H:%M') CHAIN46Z3 DONE"; touch /tmp/chain46z3.done
