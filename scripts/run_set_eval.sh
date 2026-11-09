#!/bin/bash
# [68 문서] 임의 u8 시험 세트로 기존 체크포인트 평가(학습 없음). 사용: nohup bash scripts/run_set_eval.sh LANE TAG CSV "CELL:SEED:ENVKIND ..." >> /tmp/seteval_TAG_LANE.log 2>&1 &
#   ENVKIND: base(최종 레시피) | lsub | bgref.  출력 runs/ev_{CELL}_s{SEED}_{TAG}
source /SPECPHYS_ROOT/scripts/chain_lib.sh
LANE=${1:?LANE}; TAG=${2:?TAG}; CSV=${3:?CSV}; JOBS=${4:?JOBS}; exec 9>/tmp/seteval_${TAG}_$LANE.lock; flock -n 9 || exit 0
cd $ROOT; LISTS=$PFA/cache/DataFileLists; B0="RPPG_NF=1 RPPG_SKINW=0.2 NF_SS_TRACE=1 NF_SS_REF=1 NF_ARCH=nsp NF_NSP_LAM=0.01"
ts() { date '+%m-%d %H:%M'; }; ckpt() { ls $PFA/runs/finetune_$1/PreTrainedModels/*_Epoch9.pth 2>/dev/null | head -1; }; has_out() { ls $1/*/saved_test_outputs/*.pickle > /dev/null 2>&1; }
for J in $JOBS; do C=${J%%:*}; R2=${J#*:}; S=${R2%%:*}; K=${R2#*:}; M=${C}_s$S
  case $K in bgref) E="$B0 NF_BGREF=1 NF_BGREF_MODE=gate RPPG_PARSE_CH=1";; lsub) E="$B0 NF_LSUB=1";; *) E="$B0";; esac
  [ -f /tmp/seteval_${TAG}_job_$M.done ] && continue; exec 8>/tmp/seteval_${TAG}_job_$M.lock; flock -n 8 || { exec 8>&-; continue; }; [ -f /tmp/seteval_${TAG}_job_$M.done ] && { exec 8>&-; continue; }
  ck=$(ckpt $M); [ -n "$ck" ] || { echo "$(ts) [$LANE] $M no ckpt -> skip"; exec 8>&-; continue; }
  if ! has_out $PFA/runs/ev_${M}_$TAG; then
    python scripts/vipl_yaml.py ev ${M}_${TAG}TMP "$ck" --template $PFA/yaml_NF_HYB_L/ev_NF_HYB_L_DLCN.yaml > /dev/null; y=$PFA/yaml_VIPL/ev_${M}_$TAG.yaml
    sed -e "s#$LISTS/vipl_all.csv#$CSV#" -e "s#ev_${M}_${TAG}TMP_VIPL#ev_${M}_$TAG#g" $PFA/yaml_VIPL/ev_${M}_${TAG}TMP_VIPL.yaml > $y; rm -f $PFA/yaml_VIPL/ev_${M}_${TAG}TMP_VIPL.yaml
    grep -q "FILE_LIST_PATH: $CSV" $y || { echo "$(ts) [$LANE] $M yaml fail"; exit 1; }
    ( export $E RPPG_CACHE_U8=1; run_gated 6000 $y /tmp/ev_${M}_$TAG.log ${M}_$TAG ); has_out $PFA/runs/ev_${M}_$TAG || { echo "$(ts) [$LANE] $M eval fail"; exit 1; }
  fi
  touch /tmp/seteval_${TAG}_job_$M.done; echo "$(ts) [$LANE] $M DONE"; exec 8>&-
done
touch /tmp/seteval_${TAG}_$LANE.done; echo "$(ts) [$LANE] LANE_DONE"
