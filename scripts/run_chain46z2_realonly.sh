#!/bin/bash
# 체인 46z2 (기여 점검, 10-17): 실-only 코퍼스 모델 CoTA_L(실 161) · 템플릿 파형 합성 모델 CoTAF_L 의 VIPL zero-shot 평가 → 가상얼굴 42 클립의 zero-shot 기여 대조(vs CoTAF_TW_L 5시드)
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_CACHE_U8=1
cd $ROOT
for M in CoTA_L CoTAF_L; do
  ck=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  [ -f $ck ] || { echo "missing $ck"; continue; }
  [ -f $ROOT/_workspace/data/results/agg_${M}_VIPL.json ] && { echo "skip $M"; continue; }
  python scripts/vipl_yaml.py ev $M $ck > /dev/null
  echo "$(date '+%m-%d %H:%M') EV $M VIPL start"
  run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL
  python scripts/agg_vipl.py $M 2>&1 | grep -E "^---|all "
done
echo "$(date '+%m-%d %H:%M') CHAIN46Z2 DONE"; touch /tmp/chain46z2.done
