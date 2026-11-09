#!/bin/bash
# 체인 53 (36 문서): UBFC 기반 요인 실험 — 백본 {PhysFormer(PF), FactorizePhys(FP)} × 셀 {U0 UBFC 34명만, U1 +가상얼굴 42, U2 +잡음 클립 48+분해 손실, U3 +실 노이즈·압축 사본 44}
#  Phase 0: FP 학습률 탐색(U0S 학습 / U0VAL 검증 손실) → Phase 1: 8셀 시드 100 + 13축·MMPD(캐시 공유)·VIPL 평가 → 판정 → Phase 2: 시드 200·300(규칙 충족 시).
#  체인 50 Phase 1 완료(/tmp/chain50_p1.done) 대기(10-19 변경: Phase 2b 와 병행), 체인 52 와 GPU 병행(게이트 12 GB). 로그 /tmp/chain53.log, 마커 /tmp/chain53_{p0,p1,p2}.done, /tmp/chain53.done
source /SPECPHYS_ROOT/scripts/chain_lib.sh
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; REL=$ROOT/repos/rPPG-Toolbox/final_model_release
HYB_ENV="RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_DN_ONLY=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"
fail() { echo "$(date '+%m-%d %H:%M') $1 -> abort"; touch /tmp/chain53.fail; exit 1; }
ckpt() { local M=$1; [ "${M:0:2}" = FP ] && echo "$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_factorizephys_Epoch9.pth" || echo "$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth"; }
is_hyb() { case $1 in *U2*|*U3*) return 0;; *) return 1;; esac; }
train_pf() { local M=$1 SEED=$2   # yaml_$M 은 gen_yaml 로 준비됨
  [ -f "$(ckpt $M)" ] && { echo "skip train $M"; return; }
  echo "$(date '+%m-%d %H:%M') TRAIN $M start (seed $SEED)"
  if is_hyb $M; then ( export $HYB_ENV; RPPG_SEED=$SEED run_gated $GATE $PFA/yaml_$M/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN ); else RPPG_SEED=$SEED run_gated $GATE $PFA/yaml_$M/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN; fi
  [ -f "$(ckpt $M)" ] || fail "${M}_TRAIN_FAIL"; find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; }
train_fp() { local M=$1 PFN=$2 LR=$3 SEED=$4
  [ -f "$(ckpt $M)" ] && { echo "skip train $M"; return; }
  python scripts/fp_yaml.py train $M $PFN $LR > /dev/null
  echo "$(date '+%m-%d %H:%M') TRAIN $M start (seed $SEED, lr $LR)"
  if is_hyb $M; then ( export $HYB_ENV; RPPG_SEED=$SEED run_gated $GATE $PFA/yaml_$M/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN ); else RPPG_SEED=$SEED run_gated $GATE $PFA/yaml_$M/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN; fi
  [ -f "$(ckpt $M)" ] || fail "${M}_TRAIN_FAIL"; find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; }
eval_axes() { local M=$1 PFN=$2   # 13축(MMPD 제외): PF 는 기존 캐시, FP 는 fp_* 캐시(첫 모델만 전처리)
  for f in $PFA/yaml_$PFN/ev_${PFN}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${PFN}_//"); case $A in MMPD_B*|VIPL) continue;; esac
    [ -d $PFA/runs/ev_${M}_$A ] && continue
    if [ "${M:0:2}" = FP ]; then EXP=$(grep -m1 "EXP_DATA_NAME" "$f" | awk '{print $2}'); pre=1; [ -d $PFA/cache/fp_$EXP ] && pre=0
      python scripts/fp_yaml.py ev $M $PFN $A --preprocess $pre > /dev/null; y=$PFA/yaml_$M/ev_${M}_$A.yaml
    else y=$f; fi
    if is_hyb $M; then ( export RPPG_NOISE_HEAD=1; run_gated 6000 $y /tmp/ev_${M}_$A.log ${M}_$A ); else run_gated 6000 $y /tmp/ev_${M}_$A.log ${M}_$A; fi
  done; }
eval_mmpd_all() { local MODELS="$1"   # 배치별로 캐시(mmpd_u_B*, fp_mmpd_u_B*) 공유: 백본별 첫 모델만 전처리
  for b in 1 2 3 4 5 6; do local pfdone=0 fpdone=0
    for M in $MODELS; do [ -d $PFA/runs/ev_${M}_MMPD_B$b ] && continue; PFN=$(echo $M | sed 's/^FP_/PF_/')
      if [ "${M:0:2}" = FP ]; then python scripts/fp_yaml.py ev $M $PFN MMPD_B$b --preprocess $((1-fpdone)) > /dev/null; y=$PFA/yaml_$M/ev_${M}_MMPD_B$b.yaml; fpdone=1
      else y=$PFA/yaml_$M/ev_${M}_MMPD_B$b.yaml; if [ $pfdone = 1 ]; then sed -i "s/DO_PREPROCESS: true/DO_PREPROCESS: false/" $y; else sed -i "s/DO_PREPROCESS: false/DO_PREPROCESS: true/" $y; fi; pfdone=1; fi
      if is_hyb $M; then ( export RPPG_NOISE_HEAD=1; run_gated 6000 $y /tmp/ev_${M}_MMPD_B$b.log ${M}_MMPD_B$b ); else run_gated 6000 $y /tmp/ev_${M}_MMPD_B$b.log ${M}_MMPD_B$b; fi
    done
    drop_cache mmpd_u_B$b; drop_cache fp_mmpd_u_B$b; done; }
eval_vipl() { local M=$1; [ -f $RES/agg_${M}_VIPL.json ] && return
  if [ "${M:0:2}" = FP ]; then python scripts/vipl_yaml.py ev $M "$(ckpt $M)" --template $PFA/yaml_$M/ev_${M}_DLCN.yaml > /dev/null; else python scripts/vipl_yaml.py ev $M "$(ckpt $M)" > /dev/null; fi
  if is_hyb $M; then ( export RPPG_CACHE_U8=1 RPPG_NOISE_HEAD=1; run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL ); else ( export RPPG_CACHE_U8=1; run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL ); fi
  python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; }
agg_all() { for M in "$@"; do python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; python scripts/agg_mmpd.py $M >> /tmp/agg_$M.log 2>&1; echo "$(date +%H:%M) AGG_${M}_DONE $(grep -E 'held-out\(7-33\)' /tmp/agg_$M.log | head -1 | cut -c1-60)"; done; }
run_phase() { local SEED=$1 SUF=$2 FPLR=$3; local MODELS=""
  for k in 0 1 2 3; do
    PFN=PF_U${k}$SUF; [ -d $PFA/yaml_$PFN ] || gen_yaml $PFN U$k u$k u > /dev/null
    train_pf $PFN $SEED; train_fp FP_U${k}$SUF $PFN $FPLR $SEED; MODELS="$MODELS $PFN FP_U${k}$SUF"; done
  for M in $MODELS; do eval_axes $M $(echo $M | sed 's/^FP_/PF_/'); done
  eval_mmpd_all "$MODELS"
  for M in $MODELS; do eval_vipl $M; done
  agg_all $MODELS; }
echo "$(date '+%m-%d %H:%M') CHAIN53 waiting for chain50 Phase 1 marker (chain50_p1.done; Phase 2b BASEA 5-fold 와 GPU 게이트 병행)"; until [ -f /tmp/chain50_p1.done ]; do sleep 600; done
echo "$(date '+%m-%d %H:%M') CHAIN53 START"
# ---- Phase 0: FactorizePhys 학습률 탐색 (U0S 학습 30명, U0VAL 검증 4명, 검증 손실 최소) ----
BEST=""; BESTV=""
for LR in 1e-3 1e-4 1e-5; do M=FPLR_$LR
  if [ ! -f /tmp/ft_$M.log ] || ! grep -aq "min_val_loss\|Validation" /tmp/ft_$M.log; then
    python scripts/fp_yaml.py train $M PF_U0 $LR --valid_dir $PFA/COTRAIN_U0VAL > /dev/null; sed -i "s#COTRAIN_U0\b#COTRAIN_U0S#; s#fp_cotrain_u0\b#fp_cotrain_u0s#" $PFA/yaml_$M/finetune_$M.yaml
    echo "$(date '+%m-%d %H:%M') SWEEP $M start"; RPPG_SEED=100 run_gated $GATE $PFA/yaml_$M/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN; fi
  V=$(grep -ao "min_val_loss: [0-9.eE+-]*" /tmp/ft_$M.log | tail -1 | awk '{print $2}'); [ -z "$V" ] && V=$(tr '\r' '\n' < /tmp/ft_$M.log | grep -ao "Validation RMSE:[0-9.]*\|valid loss: *[0-9.]*" | grep -oE "[0-9.]+$" | sort -g | head -1)
  echo "  $M val=${V:-none}"; if [ -n "$V" ] && { [ -z "$BESTV" ] || awk "BEGIN{exit !($V < $BESTV)}"; }; then BEST=$LR; BESTV=$V; fi
done
FPLR=${BEST:-1e-4}; echo "$(date '+%m-%d %H:%M') PHASE0 FP lr=$FPLR (val=$BESTV)"; echo $FPLR > $RES/chain53_fp_lr.txt; touch /tmp/chain53_p0.done
# ---- Phase 1 ----
run_phase 100 "" $FPLR; python scripts/ubfc_judge.py phase1 | tee /tmp/ubfc_judge_p1.txt | tail -3; touch /tmp/chain53_p1.done
if grep -q "PHASE2=1" /tmp/ubfc_judge_p1.txt; then for S in 200 300; do run_phase $S _s$S $FPLR; done; python scripts/ubfc_judge.py phase2 | tail -3; fi
touch /tmp/chain53_p2.done; echo "$(date '+%m-%d %H:%M') CHAIN53 DONE"; touch /tmp/chain53.done
