#!/bin/bash
# 체인 52 Phase 3 (34 문서 §4: 시드 추가 — CLAUDE.md 프로토콜 '쌍대 비교 ≥3 시드' 충족): MMPD 인트라 TW·HYB 셀 × 5 fold, 시드 $1 (200 또는 300).
#  두 인스턴스(시드 200·300)를 병행(게이트 12 GB). 준비 마커 /tmp/chain52p3_prep.done 대기. 로그 /tmp/chain52p3_s$1.log, 마커 /tmp/chain52p3_s$1.done
SEED=${1:?seed}; source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_CACHE_U8=1
Y=$PFA/yaml_VIPL; LISTS=$PFA/cache/DataFileLists; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results
HYB_ENV="RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_DN_ONLY=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"
cd $ROOT
fail() { echo "$(date '+%m-%d %H:%M') $1 -> abort"; touch /tmp/chain52p3_s$SEED.fail; exit 1; }
cell() { local M=$1 CSV=$2 TCSV=$3 HYB=$4 EP=10
  [ -f $RES/agg_${M}_MMPD.json ] && { echo "skip $M (done)"; return; }
  python scripts/vipl_yaml.py train $M 1 $CSV $EP --test_csv $TCSV > /dev/null
  echo "$(date '+%m-%d %H:%M') TRAIN $M start (hyb=$HYB, seed $SEED)"
  if [ $HYB = 1 ]; then ( export $HYB_ENV; RPPG_SEED=$SEED run_gated $GATE $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
  else RPPG_SEED=$SEED run_gated $GATE $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN; fi
  [ -f $PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch$((EP-1)).pth ] || fail "${M}_TRAIN_FAIL"
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch$((EP-1)).pth" -delete 2>/dev/null
  python scripts/agg_mmpd_cv2.py $M 2>&1 | head -2
  echo "$(date '+%m-%d %H:%M') CELL_${M}_DONE"; }
echo "$(date '+%m-%d %H:%M') CHAIN52P3 s$SEED waiting for prep"; until [ -f /tmp/chain52p3_prep.done ]; do sleep 300; done
echo "$(date '+%m-%d %H:%M') CHAIN52P3 s$SEED START"
for k in 1 2 3 4 5; do
  cell MF${k}_TW_L_s$SEED  $LISTS/mmpd_train${k}_cotrain_allf_tw_u8.csv   $LISTS/mmpd_test$k.csv 0
  cell MF${k}_HYB_L_s$SEED $LISTS/mmpd_train${k}_cotrain_allf_hyb4_u8.csv $LISTS/mmpd_test$k.csv 1
done
python scripts/agg_mmpd_cv2.py --cv TW_L_s$SEED HYB_L_s$SEED 2>&1 | grep -E "^---|all "
echo "$(date '+%m-%d %H:%M') CHAIN52P3 s$SEED DONE"; touch /tmp/chain52p3_s$SEED.done
