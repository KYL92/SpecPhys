#!/bin/bash
# 체인 62~64 (45 문서 §3, 체인 61 종료 후 자동): ① 2단계 사전학습→미세조정(MMPD·VIPL 5-fold) → 게이트 → ② 데이터 효율(MMPD fold 1·2 × 25/50% × 출발 2 × 3시드) → ③ 실 42 클립 교환 코퍼스 CoTSW_L 3시드.
# 로그 /tmp/chain62.log ; 마커 /tmp/chain62.done /tmp/chain63.done /tmp/chain64.done
source /SPECPHYS_ROOT/scripts/chain_lib.sh
cd $ROOT; LISTS=$PFA/cache/DataFileLists; MEM_GATE=${MEM_GATE:-10}; CK_PT=$PFA/runs/finetune_CoTAF_TW_L/PreTrainedModels/SYNTH_CoTAF_TW_L_physformer_Epoch9.pth
ts() { date '+%m-%d %H:%M'; }; fail() { echo "$(ts) $1"; touch /tmp/chain62.fail; exit 1; }
gate_mem() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$MEM_GATE" ]; do echo "$(ts) MEM_WAIT"; sleep 600; done; wait_disk; }
prep_mmpd() { if [ ! -f $LISTS/mmpd_all_u8_0.0_1.0.csv ] || [ ! -d $PFA/cache/mmpd_all_u8 ] || [ "$(ls $PFA/cache/mmpd_all_u8 | wc -l)" -lt 7000 ]; then gate_mem; nice -n 10 python scripts/nirp_pool_stream.py $PFA/COTRAIN_MMPD_ALL $PFA/cache/mmpd_all_u8 --workers 3 2>&1 | tail -1; fi
  n=$(ls $PFA/cache/mmpd_all_u8 | wc -l); [ "$n" -gt 7000 ] || fail "PREP_MMPD_FAIL n=$n"; echo "$(ts) mmpd_all_u8 ok n=$n"; }
# mcell NAME FOLD SEED TRAIN_CSV RESUME(ckpt|'')  — MMPD fold 셀(VIPL 로더, mmpd_all_u8)
mcell() { local M=$1 K=$2 SEED=$3 CSV=$4 RES=$5; ck=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  if [ ! -f $ck ]; then python scripts/vipl_yaml.py train $M 1 $CSV 10 --test_csv $LISTS/mmpd_test$K.csv ${RES:+--resume $RES} > /dev/null
    echo "$(ts) TRAIN $M start (fold $K, seed $SEED, resume=${RES:-public})"; ( export RPPG_CACHE_U8=1; RPPG_SEED=$SEED run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
    [ -f $ck ] || fail "${M}_TRAIN_FAIL"; find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; cp /tmp/ft_$M.log $ROOT/_workspace/data/results/chain62_ft_$M.log; fi
  echo "$(ts) CHAIN62 $M DONE"; }
# vcellpt FOLD — VIPL 단독 증강+검증 레시피를 코퍼스 체크포인트에서 출발
vcellpt() { local K=$1; local M=VF${K}_PTAV_L; ck=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  if [ ! -f $ck ]; then [ -f $LISTS/vipl_valid$K.csv ] || python scripts/vipl_filelist.py valid $K > /dev/null; [ -f $LISTS/vipl_train${K}_xv.csv ] || python scripts/vipl_filelist.py train $K --xv > /dev/null
    python scripts/vipl_yaml.py train $M $K $LISTS/vipl_train${K}_xv.csv 10 --valid_csv $LISTS/vipl_valid$K.csv --resume $CK_PT > /dev/null
    echo "$(ts) TRAIN $M start (fold $K, BASEAV recipe from CoTAF_TW_L)"
    ( export RPPG_CACHE_U8=1 RPPG_PEARSON_SKIP_RE='^p[0-9]+_v[0-9]' RPPG_HR_LABEL_FILE=1 RPPG_TAUG=1 RPPG_VALID_METRIC=mae; RPPG_SEED=100 run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
    [ -f $ck ] || fail "${M}_TRAIN_FAIL"; grep -aq "TAUG stats epoch 0.*x2" /tmp/ft_$M.log || fail "${M}_TAUG_FAIL"; grep -aq "HRFILE sidecar hits=[1-9]" /tmp/ft_$M.log || fail "${M}_HRFILE_FAIL"
    find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; cp /tmp/ft_$M.log $ROOT/_workspace/data/results/chain62_ft_$M.log; fi
  python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; echo "$(ts) CHAIN62 $M DONE"; }
# 64: 교환 코퍼스
swap_pool() { rm -rf $PFA/COTRAIN_SWAP; mkdir -p $PFA/COTRAIN_SWAP; for s in $PFA/COTRAIN_REALONLY/subject*; do ln -sfn "$(readlink -f $s)" $PFA/COTRAIN_SWAP/$(basename $s); done
  for S in 1 2 3 4 5 6; do for T in 00 01 02 03 04 08 15; do d=$PFA/MMPD_TRAIN/subject700${S}${T}; [ -d $d ] && ln -sfn "$(readlink -f $d)" $PFA/COTRAIN_SWAP/subject700${S}${T}; done; done
  echo "$(ts) POOL_SWAP n=$(ls $PFA/COTRAIN_SWAP | wc -l)"; [ "$(ls $PFA/COTRAIN_SWAP | wc -l)" = 203 ] || fail "SWAP_POOL_SIZE"; }
scell() { local SEED=$1 sfx=""; [ "$SEED" != 100 ] && sfx="_s$SEED"; local M=CoTSW_L$sfx; ck=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth; Y=$PFA/yaml_$M
  if [ ! -f $ck ]; then python scripts/vipl_yaml.py train $M 1 $LISTS/cotrain_swap_u8_pool.csv 10 --test_csv $LISTS/vipl_test1.csv > /dev/null
    echo "$(ts) TRAIN $M start (seed $SEED, swap corpus)"; ( export RPPG_CACHE_U8=1; RPPG_SEED=$SEED run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
    [ -f $ck ] || fail "${M}_TRAIN_FAIL"; find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; cp /tmp/ft_$M.log $ROOT/_workspace/data/results/chain64_ft_$M.log; fi
  gen_yaml $M ALLF_TW allf_tw sw > /dev/null
  for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*|VIPL|TRAINPOOL) continue;; esac; [ -d $PFA/runs/ev_${M}_$A ] && continue; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
  if [ ! -d $PFA/runs/ev_${M}_VIPL ]; then python scripts/vipl_yaml.py ev $M $ck > /dev/null; RPPG_CACHE_U8=1 run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL; fi
  python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; echo "$(ts) CHAIN64 $M DONE"; }
smmpd() { local MODELS="CoTSW_L CoTSW_L_s200 CoTSW_L_s300"
  for k in 1 2 3 4 5 6; do pfdone=0; [ -d $PFA/cache/mmpd_sw_B$k ] && [ "$(ls $PFA/cache/mmpd_sw_B$k | wc -l)" -gt 100 ] && pfdone=1
    for M in $MODELS; do [ -d $PFA/runs/ev_${M}_MMPD_B$k ] && continue; y=$PFA/yaml_$M/ev_${M}_MMPD_B$k.yaml
      if [ $pfdone = 1 ]; then sed -i "s/DO_PREPROCESS: true/DO_PREPROCESS: false/" $y; else sed -i "s/DO_PREPROCESS: false/DO_PREPROCESS: true/" $y; gate_mem; fi
      run_gated 6000 $y /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k; [ -d $PFA/runs/ev_${M}_MMPD_B$k ] && pfdone=1; done; drop_cache mmpd_sw_B$k; done
  for M in $MODELS; do python scripts/agg_mmpd.py $M > /tmp/aggm_$M.log 2>&1; echo "$M $(grep -E 'held-out\(7-33\)' /tmp/aggm_$M.log | head -1 | cut -c1-60)"; done; }
case ${1:-all} in
  all) echo "$(ts) CHAIN62 waiting chain61"; while [ ! -f /tmp/chain61.done ]; do sleep 300; done; echo "$(ts) CHAIN62 START"; prep_mmpd
    # 62-① MMPD 5-fold PT (2 병행) 와 VIPL PTAV 5-fold (2 병행) — 순차 묶음
    ( mcell MF1_PT_L 1 100 $LISTS/mmpd_train1.csv $CK_PT; mcell MF3_PT_L 3 100 $LISTS/mmpd_train3.csv $CK_PT; mcell MF5_PT_L 5 100 $LISTS/mmpd_train5.csv $CK_PT ) > /tmp/chain62_Ma.log 2>&1 & ( mcell MF2_PT_L 2 100 $LISTS/mmpd_train2.csv $CK_PT; mcell MF4_PT_L 4 100 $LISTS/mmpd_train4.csv $CK_PT ) > /tmp/chain62_Mb.log 2>&1 & wait
    [ -f /tmp/chain62.fail ] && { echo "$(ts) CHAIN62 ABORT (MMPD PT)"; exit 1; }; for k in 1 2 3 4 5; do python scripts/agg_mmpd_cv2.py MF${k}_PT_L > /dev/null 2>&1; done; python scripts/agg_mmpd_cv2.py --cv PT_L 2>&1 | grep -E "^---|all "
    ( vcellpt 1; vcellpt 3; vcellpt 5 ) > /tmp/chain62_Va.log 2>&1 & ( vcellpt 2; vcellpt 4 ) > /tmp/chain62_Vb.log 2>&1 & wait
    [ -f /tmp/chain62.fail ] && { echo "$(ts) CHAIN62 ABORT (VIPL PTAV)"; exit 1; }; python scripts/agg_vipl.py --cv PTAV_L 2>&1 | grep -E "all \[문헌"
    python scripts/pretrain_judge.py all 2>&1 | tee $ROOT/_workspace/data/results/pretrain_judge_62.txt | tail -12; touch /tmp/chain62.done
    GO=$(python scripts/pretrain_judge.py gate62 2>/dev/null | tail -1); echo "$(ts) 63 gate -> $GO"
    if [ "$GO" = GO ]; then  # 63-② 데이터 효율
      for K in 1 2; do for F in 25 50; do python scripts/mmpd_subset_csv.py $K $F; done; done
      ( for K in 1 2; do for F in 25 50; do for S in 100 200 300; do sfx=""; [ $S != 100 ] && sfx="_s$S"; mcell MF${K}_SUB${F}_PUB_L$sfx $K $S $LISTS/mmpd_train${K}_sub$F.csv ""; done; done; done ) > /tmp/chain63_a.log 2>&1 &
      ( for K in 1 2; do for F in 25 50; do for S in 100 200 300; do sfx=""; [ $S != 100 ] && sfx="_s$S"; mcell MF${K}_SUB${F}_PT_L$sfx $K $S $LISTS/mmpd_train${K}_sub$F.csv $CK_PT; done; done; done ) > /tmp/chain63_b.log 2>&1 & wait
      [ -f /tmp/chain62.fail ] && { echo "$(ts) CHAIN63 ABORT"; exit 1; }
      for K in 1 2; do for F in 25 50; do for T in PUB PT; do for sfx in "" _s200 _s300; do python scripts/agg_mmpd_cv2.py MF${K}_SUB${F}_${T}_L$sfx > /dev/null 2>&1; done; done; done; done
      python scripts/pretrain_judge.py all 2>&1 | tee $ROOT/_workspace/data/results/pretrain_judge_63.txt | tail -12; touch /tmp/chain63.done
    else echo "$(ts) CHAIN63 skipped (gate)"; fi
    drop_cache mmpd_all_u8
    # 64-③ 교환 코퍼스
    echo "$(ts) CHAIN64 START"; swap_pool; cn=cotrain_swap_u8
    if [ ! -d $PFA/cache/$cn ] || [ "$(ls $PFA/cache/$cn | grep -c _input)" -lt 1000 ]; then gate_mem; RPPG_MP_QUOTA=3 nice -n 10 python scripts/pool_preprocess_u8.py COTRAIN_SWAP $cn > /tmp/pool_$cn.log 2>&1; tail -1 /tmp/pool_$cn.log; fi
    python scripts/pool_csv_u8.py $cn $LISTS/${cn}_pool.csv > /dev/null
    ( scell 100; scell 300 ) > /tmp/chain64_a.log 2>&1 & ( scell 200 ) > /tmp/chain64_b.log 2>&1 & wait; [ -f /tmp/chain62.fail ] && { echo "$(ts) CHAIN64 ABORT"; exit 1; }
    smmpd; python scripts/realonly_judge.py CoTSW_L CoTAF_TW_L 2>&1 | tee $ROOT/_workspace/data/results/swap_judge_vs_official.txt | tail -20; python scripts/realonly_judge.py CoTReal_L CoTSW_L 2>&1 | tee $ROOT/_workspace/data/results/swap_judge_vs_realonly.txt | tail -20
    drop_cache $cn; echo "$(ts) CHAIN64 DONE"; touch /tmp/chain64.done;;
esac
