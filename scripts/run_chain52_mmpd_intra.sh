#!/bin/bash
# 체인 52 (34 문서): MMPD 피험자 배타 5-fold 인트라 — fold 별 BASE(MMPD 만, 기존 방법 in-domain) / TW(+공식 코퍼스) / HYB(+하이브리드) 15 run + 문헌 분할 참고 2 run.
# 준비 마커 /tmp/chain52_prep.done + 체인 50 완료(/tmp/chain50.done) 대기, 단독 GPU(게이트 21 GB). 로그 /tmp/chain52.log, 마커 /tmp/chain52_{p1,p2}.done, /tmp/chain52.done
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_CACHE_U8=1
Y=$PFA/yaml_VIPL; LISTS=$PFA/cache/DataFileLists; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results   # 10-17 /goal: 체인 49·50 과 GPU 병행(게이트 12 GB)
HYB_ENV="RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_DN_ONLY=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"
cd $ROOT
fail() { echo "$(date '+%m-%d %H:%M') $1 -> abort"; touch /tmp/chain52.fail; exit 1; }
# cell NAME TRAIN_CSV TEST_CSV HYB(0/1) [agg mode: cv|lit]
cell() { local M=$1 CSV=$2 TCSV=$3 HYB=$4 EP=10
  [ -f $RES/agg_${M}_MMPD.json ] && { echo "skip $M (done)"; return; }
  python scripts/vipl_yaml.py train $M 1 $CSV $EP --test_csv $TCSV > /dev/null
  echo "$(date '+%m-%d %H:%M') TRAIN $M start (hyb=$HYB)"
  if [ $HYB = 1 ]; then ( export $HYB_ENV; RPPG_SEED=100 run_gated $GATE $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
  else RPPG_SEED=100 run_gated $GATE $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN; fi
  [ -f $PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch$((EP-1)).pth ] || fail "${M}_TRAIN_FAIL"
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch$((EP-1)).pth" -delete 2>/dev/null
  if [ "$5" = lit ]; then python scripts/agg_mmpd_cv2.py --lit $M 2>&1 | head -4; else python scripts/agg_mmpd_cv2.py $M 2>&1 | head -4; fi
  echo "$(date '+%m-%d %H:%M') CELL_${M}_DONE"; }
echo "$(date '+%m-%d %H:%M') CHAIN52 waiting for prep marker (GPU 병행, 체인 50 비대기 — 10-17 /goal 우선순위)"
until [ -f /tmp/chain52_prep.done ]; do sleep 300; done
echo "$(date '+%m-%d %H:%M') CHAIN52 START (Phase 1: 5-fold × BASE/TW/HYB)"
for k in 1 2 3 4 5; do
  cell MF${k}_BASE_L $LISTS/mmpd_train$k.csv $LISTS/mmpd_test$k.csv 0
  cell MF${k}_TW_L   $LISTS/mmpd_train${k}_cotrain_allf_tw_u8.csv   $LISTS/mmpd_test$k.csv 0
  cell MF${k}_HYB_L  $LISTS/mmpd_train${k}_cotrain_allf_hyb4_u8.csv $LISTS/mmpd_test$k.csv 1
done
python scripts/agg_mmpd_cv2.py --cv BASE_L TW_L HYB_L 2>&1 | grep -E "^---|all |non-stationary|Walking"; python scripts/mmpd_judge.py phase1; touch /tmp/chain52_p1.done
echo "$(date '+%m-%d %H:%M') Phase 2: 문헌 분할 참고(학습 1~23, 테스트 27~33)"
cell ML_BASE_L $LISTS/mmpd_lit_train.csv $LISTS/mmpd_lit_test.csv 0 lit
cell ML_HYB_L  $LISTS/mmpd_lit_train_cotrain_allf_hyb4_u8.csv $LISTS/mmpd_lit_test.csv 1 lit
python scripts/mmpd_judge.py lit; touch /tmp/chain52_p2.done
echo "$(date '+%m-%d %H:%M') CHAIN52 DONE"; touch /tmp/chain52.done
