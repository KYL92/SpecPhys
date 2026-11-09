#!/bin/bash
# 체인 67 (48 §6, 2026-10-28 06:00, 사용자 지시 '절제 셀 1개'): NF_HYB_ORIG_L = 원본 FactorizePhys FSAM(rank-1, 원 T_KAB view) + 스펙트럼 손실(출력 NLL·반감)만, 하이브리드 풀 366, 시드 100.
#  목적: NF-Phys 이득(5.82 vs 공개 6.88)의 원천을 (a) 스펙트럼 손실 vs (b) FSAM 시간 축 정렬로 분리. 판정 규칙 48 §6(H-B4b). 로그 /tmp/chain67.log, 마커 /tmp/chain67.done. flock.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
exec 9>/tmp/chain66.lock; flock -n 9 || { echo "chain66 already running (lock) -> exit"; exit 0; }   # 2중 기동 방지(10-27 사고)
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; FPLR=1e-4; TPLN=CoTAM_HYB4_L
NF_ENV="RPPG_NF=1 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) $1 -> abort"; touch /tmp/chain67.fail; exit 1; }
ckpt() { echo "$PFA/runs/finetune_$1/PreTrainedModels/SYNTH_${1}_factorizephys_Epoch9.pth"; }
wait_ram() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$1" ]; do echo "$(ts) RAM_WAIT avail<$1G"; sleep 120; done; }
wait_load() { while [ "$(cut -d. -f1 /proc/loadavg)" -gt "$1" ]; do echo "$(ts) LOAD_WAIT 1min>$1"; sleep 300; done; }
cell_env() { case $1 in *NOSUP*) echo "NF_SUP=0";; *_R1_*) echo "NF_RANK1=1";; *_ORIG_*) echo "NF_ORIGLAYOUT=1";; *) echo "";; esac; }
train_nf() { local M=$1 SEED=$2
  [ -f "$(ckpt $M)" ] && { echo "skip train $M"; return; }
  python scripts/fp_yaml.py train $M $TPLN $FPLR > /dev/null
  echo "$(ts) TRAIN $M start (seed $SEED, lr $FPLR, pool $TPLN, ablation env: '$(cell_env $M)')"
  wait_ram 14; ( export $NF_ENV $(cell_env $M); RPPG_SEED=$SEED run_gated $GATE $PFA/yaml_$M/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
  [ -f "$(ckpt $M)" ] || fail "${M}_TRAIN_FAIL"
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
  grep -aq "^NF stats epoch 0" /tmp/ft_$M.log || fail "${M}_NF_INACTIVE"
  case $M in *NOSUP*|*_R1_*|*_ORIG_*) ;; *) grep -a "NF stats epoch 9" /tmp/ft_$M.log | grep -q "gate=[1-9]" || fail "${M}_NF_GATE0"; grep -a "NF stats epoch 0" /tmp/ft_$M.log | grep -q "ngt_hit=[1-9]" || fail "${M}_NGT0";; esac
  grep -a "NF stats epoch" /tmp/ft_$M.log | tail -n 1; cp /tmp/ft_$M.log $RES/chain66_ft_$M.log; }
eval_axes() { local M=$1
  for f in $PFA/yaml_$TPLN/ev_${TPLN}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${TPLN}_//"); case $A in MMPD_B*|VIPL) continue;; esac
    [ -d $PFA/runs/ev_${M}_$A ] && continue
    EXP=$(grep -m1 "EXP_DATA_NAME" "$f" | awk '{print $2}'); pre=1; [ -d $PFA/cache/fp_$EXP ] && pre=0
    python scripts/fp_yaml.py ev $M $TPLN $A --preprocess $pre > /dev/null
    ( export $NF_ENV $(cell_env $M); run_gated 3000 $PFA/yaml_$M/ev_${M}_$A.yaml /tmp/ev_${M}_$A.log ${M}_$A ); done; }
eval_mmpd_float() { local b M CACHE fpdone   # 54b 방식: 묶음마다 float 전처리(워커 quota) → 모든 셀 평가 → 캐시 삭제
  for b in 1 2 3 4 5 6; do CACHE=fp_mmpd_nf_B$b; fpdone=0; [ -f $PFA/cache/DataFileLists/${CACHE}_0.0_1.0.csv ] && [ -d $PFA/cache/$CACHE ] && fpdone=1
    for M in "$@"; do [ -d $PFA/runs/ev_${M}_MMPD_B$b ] && continue
      [ $fpdone = 0 ] && { wait_ram 16; wait_load 30; }
      python scripts/fp_yaml.py ev $M $TPLN MMPD_B$b --preprocess $((1-fpdone)) > /dev/null; y=$PFA/yaml_$M/ev_${M}_MMPD_B$b.yaml
      sed -i "s#EXP_DATA_NAME: fp_mmpd_[a-z0-9]*_B$b#EXP_DATA_NAME: $CACHE#" $y; [ $fpdone = 1 ] && sed -i "s/DO_PREPROCESS: true/DO_PREPROCESS: false/" $y
      ( export $NF_ENV $(cell_env $M); run_gated 3000 $y /tmp/ev_${M}_MMPD_B$b.log ${M}_MMPD_B$b ); fpdone=1; done
    drop_cache $CACHE; done; }
eval_vipl() { local M=$1; [ -f $RES/agg_${M}_VIPL.json ] && return
  python scripts/vipl_yaml.py ev $M "$(ckpt $M)" --template $PFA/yaml_$M/ev_${M}_DLCN.yaml > /dev/null
  ( export $NF_ENV $(cell_env $M) RPPG_CACHE_U8=1; run_gated 3000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL )
  python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; }
agg_cell() { local M=$1; python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; python scripts/agg_mmpd.py $M >> /tmp/agg_$M.log 2>&1; echo "$(ts) AGG_${M}_DONE $(grep -E 'RTNI|NIRP_SMALL' /tmp/agg_$M.log | head -2 | tr '\n' ' ' | cut -c1-120)"; }

exec 9>/tmp/chain67.lock; flock -n 9 || { echo "chain67 already running (lock) -> exit"; exit 0; }
echo "$(ts) CHAIN67 START (quota ${RPPG_MP_QUOTA:-?}, workers ${RPPG_NUM_WORKERS:-?})"
[ -d $PFA/COTRAIN_ALLF_HYB4 ] && [ "$(ls $PFA/COTRAIN_ALLF_HYB4 | wc -l)" = 366 ] || fail "POOL_HYB4"
M=NF_HYB_ORIG_L
train_nf $M 100
grep -a "NF stats epoch 9" /tmp/ft_$M.log | grep -q "ps=0.000" || fail "${M}_PS_NOT_ZERO"   # 슬롯 항이 꺼졌는지(원본 배치) 확인
eval_axes $M; eval_vipl $M; eval_mmpd_float $M; agg_cell $M
drop_cache fp_cotrain_allf_hyb4
python scripts/nf_judge.py --phase2 2>&1 | grep -E "NF_HYB_ORIG_L vs (NF_HYB_L|공개)" | head -n 30
echo "$(ts) CHAIN67 DONE"; touch /tmp/chain67.done
