#!/bin/bash
# 체인 54 (39 문서, 2026-10-21): FactorizePhys + 공식 코퍼스 (+ 하이브리드 분해) zero-shot, 시드 100·200·300.
#  셀 FP_OFF_TW(COTRAIN_ALLF_TW 203, 템플릿 yaml_CoTAF_TW_L) / FP_OFF_HYB(COTRAIN_ALLF_HYB4 366 + 하이브리드 손실, 템플릿 yaml_CoTAM_HYB4_L).
#  체인 52 Phase 3(다른 세션)과 GPU 게이트 12 GB 병행. 로그 /tmp/chain54.log, 마커 /tmp/chain54.done. 판정 scripts/fp_official_judge.py.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; FPLR=1e-4
HYB_ENV="RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_DN_ONLY=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"
fail() { echo "$(date '+%m-%d %H:%M') $1 -> abort"; touch /tmp/chain54.fail; exit 1; }
ckpt() { echo "$PFA/runs/finetune_$1/PreTrainedModels/SYNTH_${1}_factorizephys_Epoch9.pth"; }
is_hyb() { case $1 in *HYB*) return 0;; *) return 1;; esac; }
tpl() { case $1 in *HYB*) echo CoTAM_HYB4_L;; *) echo CoTAF_TW_L;; esac; }
train_fp() { local M=$1 SEED=$2 PFN=$(tpl $1)
  [ -f "$(ckpt $M)" ] && { echo "skip train $M"; return; }
  python scripts/fp_yaml.py train $M $PFN $FPLR > /dev/null
  echo "$(date '+%m-%d %H:%M') TRAIN $M start (seed $SEED, lr $FPLR, pool $PFN)"
  if is_hyb $M; then ( export $HYB_ENV; RPPG_SEED=$SEED run_gated $GATE $PFA/yaml_$M/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN ); else RPPG_SEED=$SEED run_gated $GATE $PFA/yaml_$M/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN; fi
  [ -f "$(ckpt $M)" ] || fail "${M}_TRAIN_FAIL"; find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
  if is_hyb $M; then grep -aq "hybrid-fp\] dual loss active" /tmp/ft_$M.log || fail "${M}_HYB_INACTIVE"; fi; }
eval_axes() { local M=$1 PFN=$(tpl $1)
  for f in $PFA/yaml_$PFN/ev_${PFN}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${PFN}_//"); case $A in MMPD_B*|VIPL) continue;; esac
    [ -d $PFA/runs/ev_${M}_$A ] && continue
    EXP=$(grep -m1 "EXP_DATA_NAME" "$f" | awk '{print $2}'); pre=1; [ -d $PFA/cache/fp_$EXP ] && pre=0
    python scripts/fp_yaml.py ev $M $PFN $A --preprocess $pre > /dev/null; y=$PFA/yaml_$M/ev_${M}_$A.yaml
    if is_hyb $M; then ( export RPPG_NOISE_HEAD=1; run_gated 3000 $y /tmp/ev_${M}_$A.log ${M}_$A ); else run_gated 3000 $y /tmp/ev_${M}_$A.log ${M}_$A; fi
  done; }
eval_mmpd_u8() { local M=$1   # 10-21: float 전처리 대신 mmpd_all_u8(uint8, 660편) + 목록 mmpd_zs_7_33.csv 로 held-out 7~33 평가(VIPL 로더, 패치 32 Raw 72x72), 전처리 없음
  [ -f $RES/agg_${M}_MMPD.json ] && return
  python scripts/vipl_yaml.py ev $M "$(ckpt $M)" --template $PFA/yaml_$M/ev_${M}_DLCN.yaml > /dev/null
  local y=$PFA/yaml_VIPL/ev_${M}_MMPDZS.yaml; sed -e "s#vipl_all.csv#mmpd_zs_7_33.csv#; s#EXP_DATA_NAME: vipl_u8#EXP_DATA_NAME: mmpd_all_u8#; s#ev_${M}_VIPL#ev_${M}_MMPDZS#g" $PFA/yaml_VIPL/ev_${M}_VIPL.yaml > $y
  grep -q "mmpd_zs_7_33.csv" $y && grep -q "mmpd_all_u8" $y || fail "${M}_MMPDZS_YAML"
  if is_hyb $M; then ( export RPPG_CACHE_U8=1 RPPG_NOISE_HEAD=1; run_gated 3000 $y /tmp/ev_${M}_MMPDZS.log ${M}_MMPDZS ); else ( export RPPG_CACHE_U8=1; run_gated 3000 $y /tmp/ev_${M}_MMPDZS.log ${M}_MMPDZS ); fi
  python scripts/agg_mmpd_cv2.py --outdir $PFA/runs/ev_${M}_MMPDZS $M 2>&1 | grep -E "all |non-stat" | head -2; }
eval_vipl() { local M=$1; [ -f $RES/agg_${M}_VIPL.json ] && return
  python scripts/vipl_yaml.py ev $M "$(ckpt $M)" --template $PFA/yaml_$M/ev_${M}_DLCN.yaml > /dev/null
  if is_hyb $M; then ( export RPPG_CACHE_U8=1 RPPG_NOISE_HEAD=1; run_gated 3000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL ); else ( export RPPG_CACHE_U8=1; run_gated 3000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL ); fi
  python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; }
agg_all() { for M in "$@"; do python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; echo "$(date +%H:%M) AGG_${M}_DONE"; done; }
echo "$(date '+%m-%d %H:%M') CHAIN54 START"
MODELS=""
for S in 100 200 300; do SUF=""; [ $S != 100 ] && SUF=_s$S
  for C in FP_OFF_TW FP_OFF_HYB; do train_fp ${C}${SUF} $S; MODELS="$MODELS ${C}${SUF}"; done; done
drop_cache fp_cotrain_allf_tw; drop_cache fp_cotrain_allf_hyb4
for M in $MODELS; do eval_axes $M; done
for M in $MODELS; do eval_mmpd_u8 $M; done
for M in $MODELS; do eval_vipl $M; done
agg_all $MODELS
python scripts/fp_official_judge.py | tee /tmp/fp_official_judge.txt | tail -n 6
echo "$(date '+%m-%d %H:%M') CHAIN54 DONE"; touch /tmp/chain54.done
