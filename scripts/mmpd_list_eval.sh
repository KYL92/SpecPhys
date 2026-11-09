#!/bin/bash
# [60 문서 §30] 기존 체크포인트를 지정한 MMPD 목록(csv)으로만 재평가(학습 없음): bash scripts/mmpd_list_eval.sh NAME BASE CSV "EXTRA_ENV"
#   BASE 의 VIPL 평가 yaml(yaml_VIPL/ev_${BASE}_VIPL.yaml)을 틀로 FILE_LIST_PATH 만 바꿔 runs/finetune_${NAME} 에 출력 → agg_mmpd_cv2.py NAME
source /SPECPHYS_ROOT/scripts/chain_lib.sh
N=$1; B=$2; CSV=$3; X=$4; cd $ROOT; ts() { date '+%m-%d %H:%M'; }
[ -f $PFA/yaml_VIPL/ev_${B}_VIPL.yaml ] || { echo "$(ts) MLE_FAIL $N no base yaml"; exit 1; }
python - <<PY || { echo "$(ts) MLE_FAIL $N yaml"; exit 1; }
import yaml
c = yaml.safe_load(open("$PFA/yaml_VIPL/ev_${B}_VIPL.yaml")); run = "$PFA/runs/finetune_$N"
c["LOG"]["PATH"] = run; c["TEST"]["OUTPUT_SAVE_DIR"] = run + "/outputs"; c["TEST"]["DATA"]["FILE_LIST_PATH"] = "$CSV"
yaml.safe_dump(c, open("$PFA/yaml_VIPL/ev_${N}_MMPDL.yaml", "w"), sort_keys=False)
PY
ls $PFA/runs/finetune_$N/*/saved_test_outputs/*.pickle >/dev/null 2>&1 || { echo "$(ts) MLE $N start"; ( export RPPG_CACHE_U8=1 $X; run_gated ${GATE:-6000} $PFA/yaml_VIPL/ev_${N}_MMPDL.yaml /tmp/ev_${N}_MMPDL.log ${N}_MMPDL ); }
ls $PFA/runs/finetune_$N/*/saved_test_outputs/*.pickle >/dev/null 2>&1 || { echo "$(ts) MLE_FAIL $N no output"; exit 1; }
python scripts/agg_mmpd_cv2.py $N > /tmp/agg_${N}_MMPD.log 2>&1 || { echo "$(ts) MLE_FAIL $N agg"; exit 1; }
echo "$(ts) MLE_DONE $N $(sed -n 2p /tmp/agg_${N}_MMPD.log | grep -o 'MAE= *[0-9.]*')"
