#!/bin/bash
# 체인 115 (61 문서 §14, 2026-11-04; 사용자 지시 "설정 C 에 대하여 MMPD in-domain 별도 축"): 설정 A·B 의 MMPD 피험자 배타 5-fold 2단계 적응 — 설정 C in-domain 의 구성 요소
#  MF{k}_SKA_L_s{S} (설정 A: NF_SKREF_REP_L_s{S} 출발) · MF{k}_SKB_L_s{S} (설정 B: NF_SKREF_X15_L_s{S} 출발), 시드 600·700·800, lr 5e-5, 10 에폭, + NF_HR_TB=1.
#  큐(머신별 작업 목록, 작업별 flock 로 레인 간 중복 없음). 사용: nohup bash scripts/run_chain115_mmpd_indomain.sh LANE QUEUE(alpha|pro6000) >> /tmp/chain115_LANE.log 2>&1 &
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
LANE=${1:?LANE}; Q=${2:?QUEUE}; exec 9>/tmp/chain115_$LANE.lock; flock -n 9 || { echo "chain115 $LANE already running (lock) -> exit"; exit 0; }
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; TB=$ROOT/repos/rPPG-Toolbox
NFYAML=$PFA/yaml_NF_HYB_L/finetune_NF_HYB_L.yaml; NF_ENV="RPPG_NF=1 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) [$LANE] $1 -> abort"; touch /tmp/chain115_$LANE.fail; exit 1; }
wait_ram() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$1" ]; do echo "$(ts) RAM_WAIT avail<$1G"; sleep 120; done; }
prune() { find $PFA/runs/finetune_$1/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; }
ckpt() { echo "$PFA/runs/finetune_$1/PreTrainedModels/SYNTH_${1}_factorizephys_Epoch9.pth"; }
grep -q "patch 50" $TB/dataset/data_loader/BaseLoader.py && grep -q "patch 52" $TB/neural_methods/trainer/FactorizePhysTrainer.py && grep -q "patch 54" $TB/dataset/data_loader/BaseLoader.py || fail "PATCH_MISSING"
for k in 1 2 3 4 5; do [ -f $LISTS/mmpd_train$k.csv ] && [ -f $LISTS/mmpd_test$k.csv ] || fail "LIST_MISSING fold $k"; done
[ "$(ls $PFA/cache/mmpd_all_u8 | grep -c '_parse[0-9]')" -ge 7260 ] || fail "MMPD_PARSE_MISSING"; [ -f $NFYAML ] || fail "NFYAML_MISSING"
AENV="$NF_ENV RPPG_SKINW=0.2 NF_SS_TRACE=1 NF_SS_REF=1 NF_W_DBL=0.5 RPPG_TAUG=1 RPPG_TAUG_P=0.25 RPPG_TAUG_HR_SPLIT=300 NF_HR_TB=1"
fold_train() { # fold_train NAME FOLD RESUME SEED X15(0/1)
  local M=$1 K=$2 RCK=$3 SEED=$4 X15=$5 ck y; ck=$(ckpt $M)
  [ -f "$ck" ] && { echo "$(ts) skip train $M"; return 0; }
  [ -f "$RCK" ] || fail "${M}_RESUME_MISSING $RCK"
  python scripts/vipl_yaml.py train $M $K $LISTS/mmpd_train$K.csv 10 --test_csv $LISTS/mmpd_test$K.csv --template $NFYAML --resume "$RCK" --lr 5e-5 > /dev/null; y=$PFA/yaml_VIPL/finetune_$M.yaml
  grep -q "RESUME: $RCK" $y && grep -q "NAME: FactorizePhys" $y && grep -q "FILE_LIST_PATH: $LISTS/mmpd_train$K.csv" $y || fail "${M}_YAML"
  echo "$(ts) TRAIN $M start (MMPD fold $K in-domain, 2-stage from $(basename $RCK), seed $SEED, lr 5e-5, x15=$X15)"
  wait_ram 14
  ( export $AENV RPPG_CACHE_U8=1; if [ "$X15" = 1 ]; then export RPPG_TAUG_X15_RE='^subject7[0-9]{5}$' RPPG_TAUG_X15_P=0.25; else unset RPPG_TAUG_X15_RE RPPG_TAUG_X15_P; fi; RPPG_SEED=$SEED run_gated $GATE $y /tmp/ft_$M.log ${M}_TRAIN )
  [ -f "$ck" ] || fail "${M}_TRAIN_FAIL"; prune $M
  grep -aq "^NF stats epoch 0" /tmp/ft_$M.log || fail "${M}_NF_INACTIVE"; grep -a "\[nf\] active" /tmp/ft_$M.log | grep -q "hr_tb=1" || fail "${M}_HRTB_INACTIVE"
  grep -aq "\[skinbg\] active" /tmp/ft_$M.log || fail "${M}_SKINBG_INACTIVE"; grep -a "NF-REF stats epoch 9" /tmp/ft_$M.log | grep -q "bg=[1-9]" || fail "${M}_NFREF_INACTIVE"
  grep -a "TAUG stats epoch 9" /tmp/ft_$M.log | grep -q "x2" || fail "${M}_TAUG_INACTIVE"
  if [ "$X15" = 1 ]; then grep -a "TAUG stats epoch 9" /tmp/ft_$M.log | grep -q "x1.5" || fail "${M}_X15_INACTIVE"; else grep -a "TAUG stats epoch 9" /tmp/ft_$M.log | grep -q "x1.5" && fail "${M}_X15_LEAK"; fi
  grep -a "TAUG stats epoch 9" /tmp/ft_$M.log; cp /tmp/ft_$M.log $RES/chain115_ft_$M.log
  python scripts/agg_mmpd_cv2.py $M > /tmp/agg_${M}_MMPD.log 2>&1 || fail "${M}_AGG"; echo "$(ts) $M MMPD(fold $K): $(sed -n 2p /tmp/agg_${M}_MMPD.log | cut -c1-110)"; }
case $Q in
  alpha) JOBS="SKA:1:600 SKB:1:600 SKA:2:600 SKB:2:600 SKA:3:600 SKB:3:600 SKA:4:600 SKB:4:600";;
  pro6000) JOBS="SKA:5:600 SKB:5:600"; for S in 700 800; do for K in 1 2 3 4 5; do JOBS="$JOBS SKA:$K:$S SKB:$K:$S"; done; done;;
  *) fail "unknown queue $Q";;
esac
echo "$(ts) [$LANE] start queue=$Q jobs=$(echo $JOBS | wc -w)"
for J in $JOBS; do C=${J%%:*}; R=${J#*:}; K=${R%%:*}; S=${R#*:}; M=MF${K}_${C}_L_s$S
  [ -f /tmp/chain115_job_$M.done ] && continue
  exec 8>/tmp/chain115_job_$M.lock; flock -n 8 || { exec 8>&-; continue; }
  [ -f /tmp/chain115_job_$M.done ] && { exec 8>&-; continue; }
  if [ $C = SKA ]; then fold_train $M $K "$(ckpt NF_SKREF_REP_L_s$S)" $S 0; else fold_train $M $K "$(ckpt NF_SKREF_X15_L_s$S)" $S 1; fi
  touch /tmp/chain115_job_$M.done; echo "$(ts) [$LANE] $M DONE"; exec 8>&-
done
touch /tmp/chain115_$LANE.done; echo "$(ts) [$LANE] LANE_DONE"
