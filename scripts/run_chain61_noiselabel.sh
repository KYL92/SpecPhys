#!/bin/bash
# 체인 61 (45 문서 ⑥): 잡음 라벨 절제 — 하이브리드 풀(366) 에서 잡음 정답을 {관측 trace(noise_gt_tot), 대응 파괴(noise_gt_shuf)} 로 바꾼 셀 3시드 + 분해 없음(CoTAM_L) 시드 200·300.
# 스트림 A 즉시, 스트림 B 는 체인 60(평가 전용) 종료 후(메모리 규칙: 툴박스 2 + 전처리 1). 13축 + VIPL 셀별, MMPD 6묶음 일괄, 판정 noise_label_judge.py. 로그 /tmp/chain61.log, 마커 /tmp/chain61.done
source /SPECPHYS_ROOT/scripts/chain_lib.sh
cd $ROOT; LISTS=$PFA/cache/DataFileLists; MEM_GATE=${MEM_GATE:-10}
ts() { date '+%m-%d %H:%M'; }
gate_mem() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$MEM_GATE" ]; do echo "$(ts) MEM_WAIT"; sleep 600; done; wait_disk; }
fail() { echo "$(ts) $1"; touch /tmp/chain61.fail; exit 1; }
HYB_BASE="RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_DN_ONLY=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25"
prep() { cn=cotrain_allf_hyb4_u8
  if [ ! -d $PFA/cache/$cn ] || [ "$(ls $PFA/cache/$cn | grep -c _input)" -lt 1000 ]; then gate_mem; RPPG_MP_QUOTA=3 nice -n 10 python scripts/pool_preprocess_u8.py COTRAIN_ALLF_HYB4 $cn > /tmp/pool_$cn.log 2>&1; tail -1 /tmp/pool_$cn.log; fi
  python scripts/pool_csv_u8.py $cn $LISTS/${cn}_pool.csv > /dev/null; n=$(grep -c npy $LISTS/${cn}_pool.csv); [ "$n" -gt 3000 ] || fail "PREP_FAIL n=$n"; echo "$(ts) CHAIN61 PREP DONE n=$n"; touch /tmp/chain61_prep.done; }
# cell NAME SEED NGT_DIR|none
cell() { local C=$1 SEED=$2 NGT=$3 sfx=""; [ "$SEED" != 100 ] && sfx="_s$SEED"; local M=$C$sfx HYB=1; [ "$NGT" = none ] && HYB=0
  ck=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth; Y=$PFA/yaml_$M
  if [ ! -f $ck ]; then python scripts/vipl_yaml.py train $M 1 $LISTS/cotrain_allf_hyb4_u8_pool.csv 10 --test_csv $LISTS/vipl_test1.csv > /dev/null
    echo "$(ts) TRAIN $M start (seed $SEED, noise_gt=$NGT)"
    if [ $HYB = 1 ]; then ( export $HYB_BASE HYB_NOISE_GT_DIR=$PFA/$NGT RPPG_CACHE_U8=1; RPPG_SEED=$SEED run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
    else ( unset RPPG_NOISE_HEAD RPPG_HYBRID SINC_PRIOR; export RPPG_CACHE_U8=1; RPPG_SEED=$SEED run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN ); fi
    [ -f $ck ] || fail "${M}_TRAIN_FAIL"; [ $HYB = 1 ] && { grep -aq "noise-GT sidecar 48 loaded" /tmp/ft_$M.log || fail "${M}_NGT_FAIL"; }
    find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; cp /tmp/ft_$M.log $ROOT/_workspace/data/results/chain61_ft_$M.log; fi
  export RPPG_NOISE_HEAD=1   # 평가는 항상 비엄격 적재(잡음 헤드 유무 무관; 03:47 CoTAM_L_s200 strict 적재 실패 교훈 — 해당 체크포인트는 미사용 헤드 포함 상태로 학습됨, 정직 명시)
  gen_yaml $M ALLF_HYB4 allf_hyb4 nl > /dev/null
  for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*|VIPL|TRAINPOOL) continue;; esac; [ -d $PFA/runs/ev_${M}_$A ] && continue; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
  if [ ! -d $PFA/runs/ev_${M}_VIPL ]; then python scripts/vipl_yaml.py ev $M $ck > /dev/null; RPPG_CACHE_U8=1 run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL; fi
  python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; grep -E "^  (RTNI|RTNC|PURE)" /tmp/agg_$M.log | cut -c1-60; echo "$(ts) CHAIN61 $M DONE"; }
mmpd_all() { local MODELS="$1"
  for k in 1 2 3 4 5 6; do pfdone=0; [ -d $PFA/cache/mmpd_nl_B$k ] && [ "$(ls $PFA/cache/mmpd_nl_B$k | wc -l)" -gt 100 ] && pfdone=1
    for M in $MODELS; do [ -d $PFA/runs/ev_${M}_MMPD_B$k ] && continue; y=$PFA/yaml_$M/ev_${M}_MMPD_B$k.yaml
      if [ $pfdone = 1 ]; then sed -i "s/DO_PREPROCESS: true/DO_PREPROCESS: false/" $y; else sed -i "s/DO_PREPROCESS: false/DO_PREPROCESS: true/" $y; gate_mem; fi
      ( export RPPG_NOISE_HEAD=1; run_gated 6000 $y /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k )
      [ -d $PFA/runs/ev_${M}_MMPD_B$k ] && pfdone=1; done; drop_cache mmpd_nl_B$k; done
  for M in $MODELS; do python scripts/agg_mmpd.py $M > /tmp/aggm_$M.log 2>&1; echo "$M $(grep -E 'held-out\(7-33\)' /tmp/aggm_$M.log | head -1 | cut -c1-60)"; done; }
case ${1:-all} in
  cell) cell $2 $3 $4;;
  all) echo "$(ts) CHAIN61 START"; prep
    ( cell HYBX_SHUF_L 100 noise_gt_shuf; cell HYBX_TOT_L 100 noise_gt_tot; cell CoTAM_L 200 none; cell HYBX_SHUF_L 300 noise_gt_shuf ) > /tmp/chain61_A.log 2>&1 &
    ( cell HYBX_SHUF_L 200 noise_gt_shuf; cell HYBX_TOT_L 200 noise_gt_tot; cell CoTAM_L 300 none; cell HYBX_TOT_L 300 noise_gt_tot ) > /tmp/chain61_B.log 2>&1 & wait
    [ -f /tmp/chain61.fail ] && { echo "$(ts) CHAIN61 ABORT"; exit 1; }
    mmpd_all "HYBX_SHUF_L HYBX_SHUF_L_s200 HYBX_SHUF_L_s300 HYBX_TOT_L HYBX_TOT_L_s200 HYBX_TOT_L_s300 CoTAM_L_s200 CoTAM_L_s300"
    python scripts/noise_label_judge.py 2>&1 | tee $ROOT/_workspace/data/results/noise_label_judge.txt | tail -40
    drop_cache cotrain_allf_hyb4_u8; echo "$(ts) CHAIN61 DONE"; touch /tmp/chain61.done;;
esac
