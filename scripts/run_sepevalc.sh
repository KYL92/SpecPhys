#!/bin/bash
# [66 문서 §7, 탐색] 깨끗한 배경 분리 시험(sepevalc_u8)으로 기존 체크포인트 재평가(학습 없음): 기준·게이트 보정(학습/평가만)·학습된 빼기(정답 있음/없음)·현재 최종 모델, 각 5시드.
# 사용: nohup bash scripts/run_sepevalc.sh LANE >> /tmp/sepevalc_LANE.log 2>&1 &
source /SPECPHYS_ROOT/scripts/chain_lib.sh
LANE=${1:?LANE}; exec 9>/tmp/sepevalc_$LANE.lock; flock -n 9 || exit 0
cd $ROOT; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; SEPCSV=$LISTS/sepevalc_u8.csv
B0="RPPG_NF=1 RPPG_SKINW=0.2 NF_SS_TRACE=1 NF_SS_REF=1 NF_ARCH=nsp NF_NSP_LAM=0.01"
ts() { date '+%m-%d %H:%M'; }
ckpt() { ls $PFA/runs/finetune_$1/PreTrainedModels/*_Epoch9.pth 2>/dev/null | head -1; }
has_out() { ls $1/*/saved_test_outputs/*.pickle > /dev/null 2>&1; }
[ -f $SEPCSV ] || { echo "no csv"; exit 1; }
JOBS=""; for S in 2600 2700 2800 2900 3000; do for C in NF_NSP2DR_L NF_BGREF_L NF_NSP2DR_BGT_L NF_LSUB_L NF_LSUB0_L; do JOBS="$JOBS $C:$S"; done; done; for S in 2100 2200 2300 2400 2500; do JOBS="$JOBS NF_NSP2_L:$S"; done
for J in $JOBS; do C=${J%%:*}; S=${J#*:}; M=${C}_s$S
  case $C in NF_BGREF_L|NF_NSP2DR_BGT_L) E="$B0 NF_BGREF=1 NF_BGREF_MODE=gate RPPG_PARSE_CH=1";; NF_LSUB_L|NF_LSUB0_L) E="$B0 NF_LSUB=1";; *) E="$B0";; esac
  [ -f /tmp/sepevalc_job_$M.done ] && continue; exec 8>/tmp/sepevalc_job_$M.lock; flock -n 8 || { exec 8>&-; continue; }; [ -f /tmp/sepevalc_job_$M.done ] && { exec 8>&-; continue; }
  ck=$(ckpt $M); [ -n "$ck" ] || { echo "$(ts) [$LANE] $M no ckpt -> skip"; exec 8>&-; continue; }
  if ! has_out $PFA/runs/ev_${M}_SEPEVALC; then
    python scripts/vipl_yaml.py ev ${M}_SCTMP "$ck" --template $PFA/yaml_NF_HYB_L/ev_NF_HYB_L_DLCN.yaml > /dev/null; y=$PFA/yaml_VIPL/ev_${M}_SEPEVALC.yaml
    sed -e "s#$LISTS/vipl_all.csv#$SEPCSV#" -e "s#ev_${M}_SCTMP_VIPL#ev_${M}_SEPEVALC#g" $PFA/yaml_VIPL/ev_${M}_SCTMP_VIPL.yaml > $y; rm -f $PFA/yaml_VIPL/ev_${M}_SCTMP_VIPL.yaml
    grep -q "FILE_LIST_PATH: $SEPCSV" $y || { echo "$(ts) [$LANE] $M yaml fail"; exit 1; }
    ( export $E RPPG_CACHE_U8=1; run_gated 6000 $y /tmp/ev_${M}_SEPEVALC.log ${M}_SEPEVALC )
    case $C in NF_BGREF_L|NF_NSP2DR_BGT_L) grep -aq "\[bgref\] module active: mode=gate" /tmp/ev_${M}_SEPEVALC.log || { echo "$(ts) [$LANE] $M EVAL_WITHOUT_BGREF"; exit 1; };; NF_LSUB_L|NF_LSUB0_L) grep -aq "\[lsub\] module active" /tmp/ev_${M}_SEPEVALC.log || { echo "$(ts) [$LANE] $M EVAL_WITHOUT_LSUB"; exit 1; };; esac
  fi
  SEPX_SET=C python scripts/sepx_metrics.py $M > /dev/null 2>&1 && [ -f $RES/sepxc_$M.json ] || { echo "$(ts) [$LANE] $M metrics fail"; exit 1; }
  touch /tmp/sepevalc_job_$M.done; echo "$(ts) [$LANE] $M DONE"; exec 8>&-
done
touch /tmp/sepevalc_$LANE.done; echo "$(ts) [$LANE] LANE_DONE"
