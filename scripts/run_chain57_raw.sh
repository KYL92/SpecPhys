#!/bin/bash
# 체인 57 (42 문서): 원시(Standardized) 입력 PhysFormer — 고심박(운동 후) 실패가 입력 표현(프레임 차분) 때문인지 검증. VIPL zero-shot 만 평가(13축 Standardized 캐시 없음).
# 셀(시드 100): RAW_TW_L (공식 코퍼스, UBFC 체크포인트 이어학습 10 ep lr 5e-5) · RAWS_TW_L (공식 코퍼스, scratch 25 ep lr 1e-4, PF_LOSS_A=0.1 문헌 레시피)
# 사용: bash scripts/run_chain57_raw.sh all ; 마커 /tmp/chain57.done ; 로그 /tmp/chain57.log
source /SPECPHYS_ROOT/scripts/chain_lib.sh
cd $ROOT; LISTS=$PFA/cache/DataFileLists; MEM_GATE=${MEM_GATE:-10}
ts() { date '+%m-%d %H:%M'; }
gate_mem() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$MEM_GATE" ]; do echo "$(ts) MEM_WAIT"; sleep 600; done; wait_disk; }
prep() { cn=cotrain_allf_tw_u8
  if [ ! -d $PFA/cache/$cn ] || [ "$(ls $PFA/cache/$cn | grep -c _input)" -lt 1000 ]; then gate_mem; RPPG_MP_QUOTA=3 nice -n 10 python scripts/pool_preprocess_u8.py COTRAIN_ALLF_TW $cn > /tmp/pool_$cn.log 2>&1; tail -1 /tmp/pool_$cn.log; fi
  python scripts/pool_csv_u8.py $cn $LISTS/${cn}_pool.csv; n=$(grep -c npy $LISTS/${cn}_pool.csv); [ "$n" -gt 1500 ] || { echo "$(ts) PREP_FAIL n=$n"; touch /tmp/chain57.fail; exit 1; }; echo "$(ts) CHAIN57 PREP DONE n=$n"; }
cell() {  # cell NAME SEED EPOCHS EXTRA_ARGS... ; env PF_LOSS_A 등은 호출 측
  local C=$1 SEED=$2 EP=$3; shift 3; local sfx=""; [ "$SEED" != 100 ] && sfx="_s$SEED"; local M=$C$sfx
  ck=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch$((EP-1)).pth
  if [ ! -f $ck ]; then
    python scripts/vipl_yaml.py train $M 1 $LISTS/cotrain_allf_tw_u8_pool.csv $EP --test_csv $LISTS/vipl_test1.csv --std "$@" > /dev/null
    echo "$(ts) TRAIN $M start (seed $SEED, ep $EP, std input, args: $*)"
    RPPG_CACHE_U8=1 RPPG_SEED=$SEED run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN
    [ -f $ck ] || { echo "$(ts) ${M}_TRAIN_FAIL"; touch /tmp/chain57.fail; exit 1; }
    find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch$((EP-1)).pth" -delete 2>/dev/null; cp /tmp/ft_$M.log $ROOT/_workspace/data/results/chain57_ft_$M.log; fi
  if [ ! -d $PFA/runs/ev_${M}_VIPL ]; then python scripts/vipl_yaml.py ev $M $ck > /dev/null
    python - "$PFA/yaml_VIPL/ev_${M}_VIPL.yaml" <<'PY'
import sys, yaml; p = sys.argv[1]; c = yaml.safe_load(open(p)); c["TEST"]["DATA"]["PREPROCESS"]["DATA_TYPE"] = ["Standardized"]; yaml.safe_dump(c, open(p, "w"), sort_keys=False)
PY
    RPPG_CACHE_U8=1 run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL; fi
  python scripts/agg_vipl.py $M 2>&1 | grep -E "^---|all \[|v7"; echo "$(ts) CHAIN57 $M DONE"; }
case ${1:-all} in
  all) echo "$(ts) CHAIN57 START"; prep
    ( cell RAW_TW_L 100 10 ) > /tmp/chain57_A.log 2>&1 & ( export PF_LOSS_A=0.1; cell RAWS_TW_L 100 25 --lr 1e-4 --scratch ) > /tmp/chain57_B.log 2>&1 & wait
    [ -f /tmp/chain57.fail ] && { echo "$(ts) CHAIN57 ABORT"; exit 1; }
    python scripts/raw_judge.py phase1 2>&1 | tee $ROOT/_workspace/data/results/raw_judge_phase1.txt | tail -30
    GO=$(python scripts/raw_judge.py gate 2>/dev/null | tail -1); echo "$(ts) PHASE2 gate -> $GO"
    if [ "$GO" != "NONE" ] && [ -n "$GO" ]; then for S in 200 300; do for c in $GO; do
      if [ $c = RAWS_TW_L ]; then ( export PF_LOSS_A=0.1; cell $c $S 25 --lr 1e-4 --scratch ) > /tmp/chain57_${c}_s$S.log 2>&1; else ( cell $c $S 10 ) > /tmp/chain57_${c}_s$S.log 2>&1; fi; done; done
      python scripts/raw_judge.py final 2>&1 | tee $ROOT/_workspace/data/results/raw_judge_final.txt | tail -30; fi
    drop_cache cotrain_allf_tw_u8; echo "$(ts) CHAIN57 DONE"; touch /tmp/chain57.done;;
esac
