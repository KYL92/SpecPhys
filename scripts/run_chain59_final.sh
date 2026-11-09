#!/bin/bash
# 체인 59 (44 문서): 최종 비교표 실험 — A1 TWgW 시드 200·300 × 5 fold → 게이트 → C zero-shot CoTAF_HW_L 3시드(13축+MMPD+VIPL) → B BASEAVW 5-fold → A2 앵커 TWg 시드 fold 2~5 → E fold 1 민감도.
# 사용: bash scripts/run_chain59_final.sh all ; 로그 /tmp/chain59.log ; 마커 /tmp/chain59_{A1,C,B,A2,E}.done, /tmp/chain59.done
source /SPECPHYS_ROOT/scripts/chain_lib.sh
cd $ROOT; LISTS=$PFA/cache/DataFileLists; MEM_GATE=${MEM_GATE:-10}
ts() { date '+%m-%d %H:%M'; }
gate_mem() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$MEM_GATE" ]; do echo "$(ts) MEM_WAIT"; sleep 600; done; wait_disk; }
prep() { cn=cotrain_allf_tw_u8
  if [ ! -d $PFA/cache/$cn ] || [ "$(ls $PFA/cache/$cn | grep -c _input)" -lt 1000 ]; then gate_mem; RPPG_MP_QUOTA=3 nice -n 10 python scripts/pool_preprocess_u8.py COTRAIN_ALLF_TW $cn > /tmp/pool_$cn.log 2>&1; tail -1 /tmp/pool_$cn.log; fi
  python scripts/pool_csv_u8.py $cn $LISTS/${cn}_pool.csv > /dev/null; n=$(ls $PFA/cache/$cn | grep -c _input); [ "$n" -gt 1500 ] || { echo "$(ts) PREP_FAIL n=$n"; touch /tmp/chain59.fail; exit 1; }; echo "$(ts) CHAIN59 PREP DONE n=$n"; }
fail() { echo "$(ts) $1"; touch /tmp/chain59.fail; exit 1; }
# VIPL fold 셀: vcell CELL FOLD SEED [HRW]  — CELL ∈ TWgW_L(공동학습+가중) TWg_L(앵커) BASEAVW_L(VIPL 단독+증강+검증+가중)
vcell() { local C=$1 K=$2 SEED=$3 HRW=${4:-90:4}; local sfx=""; [ "$SEED" != 100 ] && sfx="_s$SEED"; local M=VF${K}_$C$sfx CSV=$LISTS/vipl_train${K}_cotrain_allf_tw_u8.csv ARGS="" TAUG=0 VALID=0 W=""
  case $C in TWgW_L) W=$HRW;; TWg_L) ;; BASEAVW_L) W=$HRW; CSV=$LISTS/vipl_train${K}_xv.csv; ARGS="--valid_csv $LISTS/vipl_valid$K.csv"; TAUG=1; VALID=1;; *) fail "unknown $C";; esac
  ck=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  if [ ! -f $ck ]; then
    python scripts/vipl_yaml.py train $M $K $CSV 10 $ARGS > /dev/null
    echo "$(ts) TRAIN $M start (fold $K, seed $SEED, hrw=$W, taug=$TAUG, valid=$VALID)"
    ( export RPPG_CACHE_U8=1 RPPG_PEARSON_SKIP_RE='^p[0-9]+_v[0-9]' RPPG_HR_LABEL_FILE=1; [ -n "$W" ] && export RPPG_HRW=$W; [ $TAUG = 1 ] && export RPPG_TAUG=1; [ $VALID = 1 ] && export RPPG_VALID_METRIC=mae; RPPG_SEED=$SEED run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
    [ -f $ck ] || fail "${M}_TRAIN_FAIL"; grep -aq "HRFILE sidecar hits=[1-9]" /tmp/ft_$M.log || fail "${M}_HRFILE_FAIL"
    [ -n "$W" ] && { grep -aq "HRW stats epoch 9" /tmp/ft_$M.log || fail "${M}_HRW_FAIL"; }; [ $TAUG = 1 ] && { grep -aq "TAUG stats epoch 0.*x2" /tmp/ft_$M.log || fail "${M}_TAUG_FAIL"; }
    find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; cp /tmp/ft_$M.log $ROOT/_workspace/data/results/chain59_ft_$M.log; fi
  python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; echo "$(ts) CHAIN59 $M DONE"; }
# zero-shot 가중 셀: zcell SEED — 공식 코퍼스 + HRW(라벨 HR = 누적합), 13축 + VIPL (MMPD 는 zmmpd)
zcell() { local SEED=$1 sfx=""; [ "$SEED" != 100 ] && sfx="_s$SEED"; local M=CoTAF_HW_L$sfx
  ck=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth; Y=$PFA/yaml_$M
  if [ ! -f $ck ]; then python scripts/vipl_yaml.py train $M 1 $LISTS/cotrain_allf_tw_u8_pool.csv 10 --test_csv $LISTS/vipl_test1.csv > /dev/null
    echo "$(ts) TRAIN $M start (zero-shot, seed $SEED, hrw=90:4)"
    ( export RPPG_CACHE_U8=1 RPPG_HRW=90:4; RPPG_SEED=$SEED run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
    [ -f $ck ] || fail "${M}_TRAIN_FAIL"; grep -aq "HRW stats epoch 9" /tmp/ft_$M.log || fail "${M}_HRW_FAIL"
    find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; cp /tmp/ft_$M.log $ROOT/_workspace/data/results/chain59_ft_$M.log; fi
  gen_yaml $M ALLF_TW allf_tw hw > /dev/null
  for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*|VIPL) continue;; esac; [ -d $PFA/runs/ev_${M}_$A ] && continue; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
  if [ ! -d $PFA/runs/ev_${M}_VIPL ]; then python scripts/vipl_yaml.py ev $M $ck > /dev/null; RPPG_CACHE_U8=1 run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL; fi
  python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; echo "$(ts) CHAIN59 $M DONE"; }
zmmpd() { local MODELS="CoTAF_HW_L CoTAF_HW_L_s200 CoTAF_HW_L_s300"
  for k in 1 2 3 4 5 6; do pfdone=0; [ -d $PFA/cache/mmpd_hw_B$k ] && [ "$(ls $PFA/cache/mmpd_hw_B$k | wc -l)" -gt 100 ] && pfdone=1
    for M in $MODELS; do [ -d $PFA/runs/ev_${M}_MMPD_B$k ] && continue; y=$PFA/yaml_$M/ev_${M}_MMPD_B$k.yaml
      if [ $pfdone = 1 ]; then sed -i "s/DO_PREPROCESS: true/DO_PREPROCESS: false/" $y; else sed -i "s/DO_PREPROCESS: false/DO_PREPROCESS: true/" $y; gate_mem; fi
      run_gated 6000 $y /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k; [ -d $PFA/runs/ev_${M}_MMPD_B$k ] && pfdone=1; done; drop_cache mmpd_hw_B$k; done
  for M in $MODELS; do python scripts/agg_mmpd.py $M > /tmp/aggm_$M.log 2>&1; grep -E "held-out\(7-33\)" /tmp/aggm_$M.log | head -1; done; }
case ${1:-all} in
  vcell) vcell $2 $3 $4 $5;; zcell) zcell $2;;
  all) echo "$(ts) CHAIN59 START"; prep
    # A1: TWgW 시드 200·300 × 5 fold (2 병행)
    ( for K in 1 2 3; do vcell TWgW_L $K 200; done; for K in 4 5; do vcell TWgW_L $K 300; done ) > /tmp/chain59_A1a.log 2>&1 & ( for K in 1 2 3; do vcell TWgW_L $K 300; done; for K in 4 5; do vcell TWgW_L $K 200; done ) > /tmp/chain59_A1b.log 2>&1 & wait
    [ -f /tmp/chain59.fail ] && { echo "$(ts) CHAIN59 ABORT (A1)"; exit 1; }
    for s in _s200 _s300; do python scripts/agg_vipl.py --cv TWgW_L$s 2>&1 | grep -E "all \[문헌"; done; touch /tmp/chain59_A1.done
    GO=$(python scripts/final_vipl_judge.py a1 2>/dev/null | tail -1); echo "$(ts) A1 gate -> $GO"; [ "$GO" = GO ] || { echo "$(ts) CHAIN59 STOP after A1"; touch /tmp/chain59.done; exit 0; }
    # C: zero-shot 가중 3시드 (2 병행 + 1) → 13축·VIPL → MMPD
    ( zcell 100; zcell 300 ) > /tmp/chain59_Ca.log 2>&1 & ( zcell 200 ) > /tmp/chain59_Cb.log 2>&1 & wait; [ -f /tmp/chain59.fail ] && { echo "$(ts) CHAIN59 ABORT (C)"; exit 1; }
    zmmpd; touch /tmp/chain59_C.done
    # B: BASEAVW 5-fold (시드 100, 2 병행)
    ( vcell BASEAVW_L 1 100; vcell BASEAVW_L 3 100; vcell BASEAVW_L 5 100 ) > /tmp/chain59_Ba.log 2>&1 & ( vcell BASEAVW_L 2 100; vcell BASEAVW_L 4 100 ) > /tmp/chain59_Bb.log 2>&1 & wait; [ -f /tmp/chain59.fail ] && { echo "$(ts) CHAIN59 ABORT (B)"; exit 1; }
    python scripts/agg_vipl.py --cv BASEAVW_L 2>&1 | grep -E "all \[문헌"; touch /tmp/chain59_B.done
    # A2: 앵커 TWg 시드 fold 2~5
    ( for K in 2 3; do vcell TWg_L $K 200; vcell TWg_L $K 300; done ) > /tmp/chain59_A2a.log 2>&1 & ( for K in 4 5; do vcell TWg_L $K 200; vcell TWg_L $K 300; done ) > /tmp/chain59_A2b.log 2>&1 & wait; [ -f /tmp/chain59.fail ] && { echo "$(ts) CHAIN59 ABORT (A2)"; exit 1; }
    for s in _s200 _s300; do python scripts/agg_vipl.py --cv TWg_L$s 2>&1 | grep -E "all \[문헌"; done; touch /tmp/chain59_A2.done
    # E: fold 1 민감도는 별도 체인(run_chain59e_sens.sh)으로 — 셀 이름에 (THR,W) 를 넣어야 해서 분리
    python scripts/final_vipl_judge.py all 2>&1 | tee $ROOT/_workspace/data/results/final_vipl_judge.txt | tail -40
    drop_cache cotrain_allf_tw_u8; echo "$(ts) CHAIN59 DONE"; touch /tmp/chain59.done;;
esac
