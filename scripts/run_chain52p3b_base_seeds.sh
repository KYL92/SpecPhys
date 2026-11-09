#!/bin/bash
# 체인 52 Phase 3b (34 §8 한계 보완): MMPD-only 기준선 셀 MF{k}_BASE_L 시드 $1 (200 또는 300) × 5 fold — 세 셀 모두 3시드로 맞춰 쌍대 비교의 비대칭 제거.
#  mmpd_all_u8 이 없으면 스트리밍 재생성(≈5 분). 로그 /tmp/chain52p3b_s$1.log, 마커 /tmp/chain52p3b_s$1.done
SEED=${1:?seed}; source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_CACHE_U8=1
Y=$PFA/yaml_VIPL; LISTS=$PFA/cache/DataFileLists; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results
cd $ROOT
fail() { echo "$(date '+%m-%d %H:%M') $1 -> abort"; touch /tmp/chain52p3b_s$SEED.fail; exit 1; }
until [ -f /tmp/chain52p3b_prep.done ]; do sleep 120; done
cell() { local M=$1 CSV=$2 TCSV=$3 EP=10
  [ -f $RES/agg_${M}_MMPD.json ] && { echo "skip $M (done)"; return; }
  python scripts/vipl_yaml.py train $M 1 $CSV $EP --test_csv $TCSV > /dev/null
  echo "$(date '+%m-%d %H:%M') TRAIN $M start (hyb=0, seed $SEED)"
  RPPG_SEED=$SEED run_gated $GATE $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN
  [ -f $PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch$((EP-1)).pth ] || fail "${M}_TRAIN_FAIL"
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch$((EP-1)).pth" -delete 2>/dev/null
  python scripts/agg_mmpd_cv2.py $M 2>&1 | head -2
  echo "$(date '+%m-%d %H:%M') CELL_${M}_DONE"; }
echo "$(date '+%m-%d %H:%M') CHAIN52P3B s$SEED START"
for k in 1 2 3 4 5; do cell MF${k}_BASE_L_s$SEED $LISTS/mmpd_train$k.csv $LISTS/mmpd_test$k.csv; done
python scripts/agg_mmpd_cv2.py --cv BASE_L_s$SEED 2>&1 | grep -E "^---|all "
echo "$(date '+%m-%d %H:%M') CHAIN52P3B s$SEED DONE"; touch /tmp/chain52p3b_s$SEED.done
