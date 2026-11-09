#!/bin/bash
# 체인 46 진단: fold 모델의 epoch별 체크포인트(runs/finetune_$MODEL/epochs_keep/)를 자기 fold held-out(vipl_test{k}.csv)에서 평가해 epoch 곡선을 얻는다.
# 사용: bash scripts/chain46_epoch_curve.sh MODEL TAG FOLD [EPOCH ...]  (예: VF1_TW_L EP1TW 1 0 2 4 6 8) → agg_{TAG}_E{e}_VIPL.json, 로그 /tmp/ev_{TAG}_E*.log
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_CACHE_U8=1
MODEL=$1; TAG=$2; FOLD=$3; shift 3
Y=$PFA/yaml_VIPL; KEEP=$PFA/runs/finetune_$MODEL/epochs_keep
cd $ROOT
for e in ${@:-0 2 4 6 8}; do
  M=${TAG}_E$e; ck=$KEEP/SYNTH_${MODEL}_physformer_Epoch$e.pth
  [ -f $ck ] || { echo "missing $ck"; continue; }
  [ -f $ROOT/_workspace/data/results/agg_${M}_VIPL.json ] && { echo "skip $M"; continue; }
  python scripts/vipl_yaml.py ev $M $ck > /dev/null
  sed -i "s#vipl_all.csv#vipl_test${FOLD}.csv#" $Y/ev_${M}_VIPL.yaml
  run_gated 6000 $Y/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL
  python scripts/agg_vipl.py $M 2>&1 | head -3
done
