#!/bin/bash
# 체인 58 (43 문서): VIPL 5-fold in-domain — 원시(Standardized) 입력 PhysFormer × 고심박 가중 손실(패치 36 RPPG_HRW). 앵커: TWg(장치 라벨 공동학습)·BASEAV(VIPL 단독 증강+검증).
# Phase 1 fold 1 4셀: RAWg(원시+TWg 레시피) · RAWgW(+HRW 90:4) · TWgW(차분+HRW) · RAWV(원시+VIPL 단독+검증 epoch 선택; 시간 배속 증강은 u8 로더가 DiffNormalized 에서만 지원해 제외) → 게이트 통과 셀만 fold 2~5 → 5-fold 판정.
# 사용: bash scripts/run_chain58_raw5.sh all ; 마커 /tmp/chain58.done ; 로그 /tmp/chain58.log
source /SPECPHYS_ROOT/scripts/chain_lib.sh
cd $ROOT; LISTS=$PFA/cache/DataFileLists; MEM_GATE=${MEM_GATE:-10}
export RPPG_CACHE_U8=1 RPPG_PEARSON_SKIP_RE='^p[0-9]+_v[0-9]' RPPG_HR_LABEL_FILE=1
ts() { date '+%m-%d %H:%M'; }
gate_mem() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$MEM_GATE" ]; do echo "$(ts) MEM_WAIT"; sleep 600; done; wait_disk; }
prep() { cn=cotrain_allf_tw_u8
  if [ ! -d $PFA/cache/$cn ] || [ "$(ls $PFA/cache/$cn | grep -c _input)" -lt 1000 ]; then gate_mem; RPPG_MP_QUOTA=3 nice -n 10 python scripts/pool_preprocess_u8.py COTRAIN_ALLF_TW $cn > /tmp/pool_$cn.log 2>&1; tail -1 /tmp/pool_$cn.log; fi
  n=$(ls $PFA/cache/$cn | grep -c _input); [ "$n" -gt 1500 ] || { echo "$(ts) PREP_FAIL n=$n"; touch /tmp/chain58.fail; exit 1; }; echo "$(ts) CHAIN58 PREP DONE n=$n"; }
cell() {  # cell CELL FOLD  (CELL in RAWg_L RAWgW_L TWgW_L RAWAV_L)
  local C=$1 K=$2; local M=VF${K}_$1 EXTRA="" ARGS="" CSV=$LISTS/vipl_train${K}_cotrain_allf_tw_u8.csv HRW="" TAUG=0 VALID=0
  case $C in RAWg_L) ARGS="--std";; RAWgW_L) ARGS="--std"; HRW="90:4";; TWgW_L) HRW="90:4";; RAWV_L) ARGS="--std --valid_csv $LISTS/vipl_valid$K.csv"; CSV=$LISTS/vipl_train${K}_xv.csv; VALID=1;; *) echo "unknown $C"; exit 1;; esac
  ck=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  if [ ! -f $ck ]; then
    [ $VALID = 1 ] && { [ -f $LISTS/vipl_valid$K.csv ] || python scripts/vipl_filelist.py valid $K > /dev/null; [ -f $LISTS/vipl_train${K}_xv.csv ] || python scripts/vipl_filelist.py train $K --xv > /dev/null; }
    python scripts/vipl_yaml.py train $M $K $CSV 10 $ARGS > /dev/null
    echo "$(ts) TRAIN $M start (fold $K, args: $ARGS, hrw=$HRW, taug=$TAUG, valid=$VALID)"
    ( [ -n "$HRW" ] && export RPPG_HRW=$HRW; [ $TAUG = 1 ] && export RPPG_TAUG=1; [ $VALID = 1 ] && export RPPG_VALID_METRIC=mae; RPPG_SEED=100 run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
    [ -f $ck ] || { echo "$(ts) ${M}_TRAIN_FAIL"; touch /tmp/chain58.fail; exit 1; }
    grep -aq "HRFILE sidecar hits=[1-9]" /tmp/ft_$M.log || { echo "$(ts) ${M}_HRFILE_FAIL"; touch /tmp/chain58.fail; exit 1; }
    [ -n "$HRW" ] && { grep -aq "HRW stats epoch 9" /tmp/ft_$M.log || { echo "$(ts) ${M}_HRW_FAIL"; touch /tmp/chain58.fail; exit 1; }; }
    [ $TAUG = 1 ] && { grep -aq "TAUG stats epoch 0.*x2" /tmp/ft_$M.log || { echo "$(ts) ${M}_TAUG_FAIL"; touch /tmp/chain58.fail; exit 1; }; }
    find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; cp /tmp/ft_$M.log $ROOT/_workspace/data/results/chain58_ft_$M.log; fi
  python scripts/agg_vipl.py $M 2>&1 | grep -E "^---|all \[|v7"; echo "$(ts) CHAIN58 $M DONE"; }
case ${1:-all} in
  cell) cell $2 $3;;
  all) echo "$(ts) CHAIN58 START"; prep
    ( cell RAWg_L 1; cell TWgW_L 1 ) > /tmp/chain58_A.log 2>&1 & ( cell RAWgW_L 1; cell RAWV_L 1 ) > /tmp/chain58_B.log 2>&1 & wait
    [ -f /tmp/chain58.fail ] && { echo "$(ts) CHAIN58 ABORT"; exit 1; }
    python scripts/raw5_judge.py phase1 2>&1 | tee $ROOT/_workspace/data/results/raw5_judge_phase1.txt | tail -30
    GO=$(python scripts/raw5_judge.py gate 2>/dev/null | tail -1); echo "$(ts) PHASE2 gate -> $GO"
    if [ "$GO" != "NONE" ] && [ -n "$GO" ]; then
      for c in $GO; do ( cell $c 2; cell $c 4 ) > /tmp/chain58_${c}_A.log 2>&1 & ( cell $c 3; cell $c 5 ) > /tmp/chain58_${c}_B.log 2>&1 & wait; [ -f /tmp/chain58.fail ] && { echo "$(ts) CHAIN58 ABORT (phase 2)"; exit 1; }
        python scripts/agg_vipl.py --cv $c 2>&1 | grep -E "^---|all \[|v7"; done
      python scripts/raw5_judge.py final 2>&1 | tee $ROOT/_workspace/data/results/raw5_judge_final.txt | tail -40; fi
    drop_cache cotrain_allf_tw_u8; echo "$(ts) CHAIN58 DONE"; touch /tmp/chain58.done;;
esac
