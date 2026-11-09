#!/bin/bash
# 체인 55 (40 문서): 도메인 일반화 기준선(GroupDRO·VREx·DANN, 툴박스 패치 #35 RPPG_DG) × {공식 코퍼스 TW 풀, 하이브리드 풀 + 분해 손실} —
# "최신 일반화 기법 + 최강 백본과 같은 학습 데이터로 비교" 요구(38 문서 평가) 대응. 평가 13축 + MMPD held-out + VIPL zero-shot 전부.
# 사용: bash scripts/run_chain55_dg.sh all            # 52p3b 마커 대기 → prep → Phase 1(시드 100, TW·HYB 스트림 병행) → MMPD → 판정 → 게이트 통과 방법만 Phase 2(시드 200·300)
#       bash scripts/run_chain55_dg.sh prep | train TW|HYB SEED [METHODS] | mmpd SEED | judge MODE
# 마커: /tmp/chain55_prep.done, /tmp/chain55_{TW,HYB}_s{SEED}.done, /tmp/chain55_mmpd_s{SEED}.done, /tmp/chain55.done; 로그 /tmp/chain55.log
source /SPECPHYS_ROOT/scripts/chain_lib.sh
cd $ROOT; LISTS=$PFA/cache/DataFileLists; MEM_GATE=${MEM_GATE:-10}
HYB_ENV="RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_DN_ONLY=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"
ALL_METHODS="GDRO VREX DANN"
ts() { date '+%m-%d %H:%M'; }
gate_mem() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$MEM_GATE" ]; do echo "$(ts) MEM_WAIT"; sleep 600; done; wait_disk; }

prep() {
  echo "$(ts) CHAIN55 PREP START"
  for p in ALLF_TW:cotrain_allf_tw_u8 ALLF_HYB4:cotrain_allf_hyb4_u8; do pool=${p%%:*}; cn=${p##*:}
    if [ ! -d $PFA/cache/$cn ] || [ "$(ls $PFA/cache/$cn | grep -c _input)" -lt 1000 ]; then
      gate_mem; RPPG_MP_QUOTA=3 nice -n 10 python scripts/pool_preprocess_u8.py COTRAIN_$pool $cn > /tmp/pool_$cn.log 2>&1; tail -1 /tmp/pool_$cn.log; fi
    python scripts/pool_csv_u8.py $cn $LISTS/${cn}_pool.csv; python scripts/dg_domain_map.py $pool $PFA/dg_domains_$pool.json
  done
  n1=$(grep -c npy $LISTS/cotrain_allf_tw_u8_pool.csv); n2=$(grep -c npy $LISTS/cotrain_allf_hyb4_u8_pool.csv)
  [ "$n1" -gt 1500 ] && [ "$n2" -gt 3000 ] || { echo "$(ts) PREP_FAIL tw=$n1 hyb=$n2"; touch /tmp/chain55.fail; exit 1; }
  echo "$(ts) CHAIN55 PREP DONE tw=$n1 hyb=$n2"; touch /tmp/chain55_prep.done
}

train() {  # train STREAM SEED [METHODS]
  local ST=$1 SEED=$2 METHODS=${3:-$ALL_METHODS} sfx=""; [ "$SEED" != 100 ] && sfx="_s$SEED"
  if [ $ST = TW ]; then CSV=$LISTS/cotrain_allf_tw_u8_pool.csv; MAP=$PFA/dg_domains_ALLF_TW.json; HYB=0; else CSV=$LISTS/cotrain_allf_hyb4_u8_pool.csv; MAP=$PFA/dg_domains_ALLF_HYB4.json; HYB=1; export RPPG_NOISE_HEAD=1; fi   # 잡음 헤드 체크포인트 평가(strict 적재)
  for m in $METHODS; do
    [ $HYB = 1 ] && M=DG_${m}_HYB_L$sfx || M=DG_${m}_L$sfx
    ck=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth; Y=$PFA/yaml_$M
    if [ ! -f $ck ]; then
      python scripts/vipl_yaml.py train $M 1 $CSV 10 --test_csv $LISTS/vipl_test1.csv > /dev/null
      echo "$(ts) TRAIN $M start (stream $ST, seed $SEED, dg=$m)"
      dg=$(echo $m | tr A-Z a-z)
      if [ $HYB = 1 ]; then ( export $HYB_ENV; RPPG_CACHE_U8=1 RPPG_SEED=$SEED RPPG_DG=$dg RPPG_DG_DOMAIN_MAP=$MAP run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
      else RPPG_CACHE_U8=1 RPPG_SEED=$SEED RPPG_DG=$dg RPPG_DG_DOMAIN_MAP=$MAP run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN; fi
      [ -f $ck ] || { echo "$(ts) ${M}_TRAIN_FAIL -> abort"; touch /tmp/chain55.fail; exit 1; }
      grep -q "DG stats epoch 9: method=$dg" /tmp/ft_$M.log && ! grep -q "miss=[1-9]" /tmp/ft_$M.log || { echo "$(ts) ${M}_DG_CHECK_FAIL (DG stats 없음 또는 도메인 미매핑)"; touch /tmp/chain55.fail; exit 1; }
      grep "DG stats epoch 9" /tmp/ft_$M.log | cut -c1-200; grep "NAN_SKIPPED" /tmp/ft_$M.log | tail -1
      find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
      cp /tmp/ft_$M.log $ROOT/_workspace/data/results/chain55_ft_$M.log
    else echo "skip train $M"; fi
    gen_yaml $M ALLF_TW allf_tw dg > /dev/null
    for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*|VIPL) continue;; esac
      [ -d $PFA/runs/ev_${M}_$A ] && continue; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
    if [ ! -d $PFA/runs/ev_${M}_VIPL ]; then python scripts/vipl_yaml.py ev $M $ck > /dev/null; RPPG_CACHE_U8=1 run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL; fi
    python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; python scripts/agg_vipl.py $M 2>&1 | grep -E "^---|all " ; grep -E "^(MCD|PURE|DLCN|NIRP)" /tmp/agg_$M.log | head -8
  done
  echo "$(ts) CHAIN55 $ST s$SEED DONE"; touch /tmp/chain55_${ST}_s$SEED.done
}

mmpd() {  # mmpd SEED [METHODS]
  local SEED=$1 METHODS=${2:-$ALL_METHODS} sfx=""; [ "$SEED" != 100 ] && sfx="_s$SEED"; local MODELS=""
  for m in $METHODS; do MODELS="$MODELS DG_${m}_L$sfx DG_${m}_HYB_L$sfx"; done
  for k in 1 2 3 4 5 6; do   # 묶음 캐시 mmpd_dg_B$k 를 모델 간 공유: 첫 모델만 전처리(체인 53 방식)
    pfdone=0; [ -d $PFA/cache/mmpd_dg_B$k ] && [ "$(ls $PFA/cache/mmpd_dg_B$k | wc -l)" -gt 100 ] && pfdone=1
    for M in $MODELS; do [ -d $PFA/runs/ev_${M}_MMPD_B$k ] && continue; y=$PFA/yaml_$M/ev_${M}_MMPD_B$k.yaml
      if [ $pfdone = 1 ]; then sed -i "s/DO_PREPROCESS: true/DO_PREPROCESS: false/" $y; else sed -i "s/DO_PREPROCESS: false/DO_PREPROCESS: true/" $y; gate_mem; fi
      case $M in *_HYB_*) ( export RPPG_NOISE_HEAD=1; run_gated 6000 $y /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k );; *) run_gated 6000 $y /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k;; esac
      [ -d $PFA/runs/ev_${M}_MMPD_B$k ] && pfdone=1; done
    drop_cache mmpd_dg_B$k
  done
  for M in $MODELS; do python scripts/agg_mmpd.py $M > /tmp/aggm_$M.log 2>&1; grep -E "^MMPD|held" /tmp/aggm_$M.log | head -3; done
  echo "$(ts) CHAIN55 MMPD s$SEED DONE"; touch /tmp/chain55_mmpd_s$SEED.done
}

case ${1:-all} in
  prep) prep;; train) train $2 $3 "$4";; mmpd) mmpd $2 "$3";;
  judge) python scripts/dg_judge.py ${2:-final} 2>&1 | tee $ROOT/_workspace/data/results/dg_judge_${2:-final}.txt | tail -30;;
  all)
    echo "$(ts) CHAIN55 START (waiting chain52p3b markers)"
    while [ ! -f /tmp/chain52p3b_s200.done ] || [ ! -f /tmp/chain52p3b_s300.done ]; do sleep 300; done
    [ -f /tmp/chain55_prep.done ] || prep
    ( bash $0 train TW 100 ) > /tmp/chain55_TW_s100.log 2>&1 & ( bash $0 train HYB 100 ) > /tmp/chain55_HYB_s100.log 2>&1 & wait
    [ -f /tmp/chain55.fail ] && { echo "$(ts) CHAIN55 ABORT (fail marker)"; exit 1; }
    mmpd 100; python scripts/dg_judge.py phase1 2>&1 | tee $ROOT/_workspace/data/results/dg_judge_phase1.txt | tail -25
    GO=$(python scripts/dg_judge.py gate 2>/dev/null | tail -1); echo "$(ts) PHASE2 gate -> $GO"
    if [ "$GO" != "NONE" ] && [ -n "$GO" ]; then
      for S in 200 300; do
        ( bash $0 train TW $S "$GO" ) > /tmp/chain55_TW_s$S.log 2>&1 & ( bash $0 train HYB $S "$GO" ) > /tmp/chain55_HYB_s$S.log 2>&1 & wait
        [ -f /tmp/chain55.fail ] && { echo "$(ts) CHAIN55 ABORT (fail marker, phase 2)"; exit 1; }
        mmpd $S "$GO"; done
      python scripts/dg_judge.py final 2>&1 | tee $ROOT/_workspace/data/results/dg_judge_final.txt | tail -30
    fi
    drop_cache cotrain_allf_tw_u8; drop_cache cotrain_allf_hyb4_u8
    echo "$(ts) CHAIN55 DONE"; touch /tmp/chain55.done;;
esac
