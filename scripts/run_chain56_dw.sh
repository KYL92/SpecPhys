#!/bin/bash
# 체인 56 (41 문서): 도메인 고정 가중 코퍼스 — 체인 55 의 발견(GroupDRO 가 MCD 단독 적합으로 VIPL 7.53 달성, 단 붕괴)을 붕괴 없이 재현하려는 시도.
# 셀(시드 100): DW_MCD2_L / DW_MCD4_L (공식 코퍼스, MCD 가중 2/4) · DW_MCD4_HYB_L (하이브리드 풀 + 분해 손실, MCD 가중 4). 게이트 통과 셀만 시드 200·300.
# 사용: bash scripts/run_chain56_dw.sh all | prep | train CELL SEED | mmpd SEED [CELLS] ; 마커 /tmp/chain56_prep.done, /tmp/chain56_<CELL>_s<SEED>.done, /tmp/chain56.done ; 로그 /tmp/chain56.log
source /SPECPHYS_ROOT/scripts/chain_lib.sh
cd $ROOT; LISTS=$PFA/cache/DataFileLists; MEM_GATE=${MEM_GATE:-10}
HYB_ENV="RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_DN_ONLY=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"
ALL_CELLS="DW_MCD2_L DW_MCD4_L DW_MCD4_HYB_L"
ts() { date '+%m-%d %H:%M'; }
gate_mem() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$MEM_GATE" ]; do echo "$(ts) MEM_WAIT"; sleep 600; done; wait_disk; }
cell_cfg() { case $1 in DW_MCD2_L*) W='{"mcd": 2}'; HYB=0;; DW_MCD4_L*) W='{"mcd": 4}'; HYB=0;; DW_MCD4_HYB_L*) W='{"mcd": 4}'; HYB=1;; *) echo "unknown cell $1"; exit 1;; esac
  if [ $HYB = 1 ]; then CSV=$LISTS/cotrain_allf_hyb4_u8_pool.csv; MAP=$PFA/dg_domains_ALLF_HYB4.json; else CSV=$LISTS/cotrain_allf_tw_u8_pool.csv; MAP=$PFA/dg_domains_ALLF_TW.json; fi; }

prep() {
  echo "$(ts) CHAIN56 PREP START"
  for p in ALLF_TW:cotrain_allf_tw_u8 ALLF_HYB4:cotrain_allf_hyb4_u8; do pool=${p%%:*}; cn=${p##*:}
    if [ ! -d $PFA/cache/$cn ] || [ "$(ls $PFA/cache/$cn | grep -c _input)" -lt 1000 ]; then
      gate_mem; RPPG_MP_QUOTA=3 nice -n 10 python scripts/pool_preprocess_u8.py COTRAIN_$pool $cn > /tmp/pool_$cn.log 2>&1; tail -1 /tmp/pool_$cn.log; fi
    python scripts/pool_csv_u8.py $cn $LISTS/${cn}_pool.csv; python scripts/dg_domain_map.py $pool $PFA/dg_domains_$pool.json; done
  n1=$(grep -c npy $LISTS/cotrain_allf_tw_u8_pool.csv); n2=$(grep -c npy $LISTS/cotrain_allf_hyb4_u8_pool.csv)
  [ "$n1" -gt 1500 ] && [ "$n2" -gt 3000 ] || { echo "$(ts) PREP_FAIL tw=$n1 hyb=$n2"; touch /tmp/chain56.fail; exit 1; }
  echo "$(ts) CHAIN56 PREP DONE tw=$n1 hyb=$n2"; touch /tmp/chain56_prep.done
}

train() {  # train CELL SEED
  local C=$1 SEED=$2 sfx=""; [ "$SEED" != 100 ] && sfx="_s$SEED"; local M=$C$sfx; cell_cfg $C
  [ $HYB = 1 ] && export RPPG_NOISE_HEAD=1
  ck=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth; Y=$PFA/yaml_$M
  if [ ! -f $ck ]; then
    python scripts/vipl_yaml.py train $M 1 $CSV 10 --test_csv $LISTS/vipl_test1.csv > /dev/null
    echo "$(ts) TRAIN $M start (seed $SEED, weights $W, hyb $HYB)"
    if [ $HYB = 1 ]; then ( export $HYB_ENV; RPPG_CACHE_U8=1 RPPG_SEED=$SEED RPPG_DG=fixed RPPG_DG_WEIGHTS="$W" RPPG_DG_DOMAIN_MAP=$MAP run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
    else RPPG_CACHE_U8=1 RPPG_SEED=$SEED RPPG_DG=fixed RPPG_DG_WEIGHTS="$W" RPPG_DG_DOMAIN_MAP=$MAP run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN; fi
    [ -f $ck ] || { echo "$(ts) ${M}_TRAIN_FAIL -> abort"; touch /tmp/chain56.fail; exit 1; }
    grep -q "DG stats epoch 9: method=fixed" /tmp/ft_$M.log && ! grep -q "miss=[1-9]" /tmp/ft_$M.log || { echo "$(ts) ${M}_DG_CHECK_FAIL"; touch /tmp/chain56.fail; exit 1; }
    grep "DG stats epoch 9" /tmp/ft_$M.log | cut -c1-200
    find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; cp /tmp/ft_$M.log $ROOT/_workspace/data/results/chain56_ft_$M.log
  else echo "skip train $M"; fi
  gen_yaml $M ALLF_TW allf_tw dw > /dev/null
  for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*|VIPL) continue;; esac
    [ -d $PFA/runs/ev_${M}_$A ] && continue; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
  if [ ! -d $PFA/runs/ev_${M}_VIPL ]; then python scripts/vipl_yaml.py ev $M $ck > /dev/null; RPPG_CACHE_U8=1 run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL; fi
  python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; python scripts/agg_vipl.py $M 2>&1 | grep -E "^---|all "; grep -E "^(PURE|RTNI|RTNC|MCD_A|NIRP_SMALL)" /tmp/agg_$M.log | head -5
  echo "$(ts) CHAIN56 $M DONE"; touch /tmp/chain56_${C}_s$SEED.done
}

mmpd() {  # mmpd SEED [CELLS]
  local SEED=$1 CELLS=${2:-$ALL_CELLS} sfx=""; [ "$SEED" != 100 ] && sfx="_s$SEED"; local MODELS=""
  for c in $CELLS; do MODELS="$MODELS $c$sfx"; done
  for k in 1 2 3 4 5 6; do
    pfdone=0; [ -d $PFA/cache/mmpd_dw_B$k ] && [ "$(ls $PFA/cache/mmpd_dw_B$k | wc -l)" -gt 100 ] && pfdone=1
    for M in $MODELS; do [ -d $PFA/runs/ev_${M}_MMPD_B$k ] && continue; y=$PFA/yaml_$M/ev_${M}_MMPD_B$k.yaml
      if [ $pfdone = 1 ]; then sed -i "s/DO_PREPROCESS: true/DO_PREPROCESS: false/" $y; else sed -i "s/DO_PREPROCESS: false/DO_PREPROCESS: true/" $y; gate_mem; fi
      case $M in *_HYB_*) ( export RPPG_NOISE_HEAD=1; run_gated 6000 $y /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k );; *) run_gated 6000 $y /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k;; esac
      [ -d $PFA/runs/ev_${M}_MMPD_B$k ] && pfdone=1; done
    drop_cache mmpd_dw_B$k
  done
  for M in $MODELS; do python scripts/agg_mmpd.py $M > /tmp/aggm_$M.log 2>&1; grep -E "held-out\(7-33\)" /tmp/aggm_$M.log | head -1; done
  echo "$(ts) CHAIN56 MMPD s$SEED DONE"; touch /tmp/chain56_mmpd_s$SEED.done
}

case ${1:-all} in
  prep) prep;; train) train $2 $3;; mmpd) mmpd $2 "$3";;
  all)
    echo "$(ts) CHAIN56 START"; [ -f /tmp/chain56_prep.done ] || prep
    ( bash $0 train DW_MCD2_L 100; bash $0 train DW_MCD4_L 100 ) > /tmp/chain56_A_s100.log 2>&1 & ( bash $0 train DW_MCD4_HYB_L 100 ) > /tmp/chain56_B_s100.log 2>&1 & wait
    [ -f /tmp/chain56.fail ] && { echo "$(ts) CHAIN56 ABORT"; exit 1; }
    mmpd 100; python scripts/dw_judge.py phase1 2>&1 | tee $ROOT/_workspace/data/results/dw_judge_phase1.txt | tail -25
    GO=$(python scripts/dw_judge.py gate 2>/dev/null | tail -1); echo "$(ts) PHASE2 gate -> $GO"
    if [ "$GO" != "NONE" ] && [ -n "$GO" ]; then
      for S in 200 300; do
        i=0; for c in $GO; do ( bash $0 train $c $S ) > /tmp/chain56_${c}_s$S.log 2>&1 & i=$((i+1)); [ $i -ge 2 ] && { wait; i=0; }; done; wait
        [ -f /tmp/chain56.fail ] && { echo "$(ts) CHAIN56 ABORT (phase 2)"; exit 1; }
        mmpd $S "$GO"; done
      python scripts/dw_judge.py final 2>&1 | tee $ROOT/_workspace/data/results/dw_judge_final.txt | tail -30
    fi
    drop_cache cotrain_allf_tw_u8; drop_cache cotrain_allf_hyb4_u8
    echo "$(ts) CHAIN56 DONE"; touch /tmp/chain56.done;;
esac
