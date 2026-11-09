#!/bin/bash
# [60 문서 §22, patch 48] 추론 시 반속 TTA 평가: 기존 모델 체크포인트를 VIPL·MMPD 7~33 에 RPPG_TTA_SLOW=1 로 재평가 → 결합(tta_combine.py)
#   bash scripts/tta_eval.sh SLOWNAME BASEMODEL "EXTRA_ENV"   (예: RF_TTS_D_L_s200 RF_SPEC_D_L_s200 "" | NF_TTS_D_L NF_HYB_D_L "RPPG_NF=1 HYB_NOISE_GT_DIR=...")
source /SPECPHYS_ROOT/scripts/chain_lib.sh
S=$1; B=$2; X=$3; cd $ROOT; LISTS=$PFA/cache/DataFileLists; RES=$ROOT/_workspace/data/results
ts() { date '+%m-%d %H:%M'; }
[ -f $PFA/yaml_VIPL/ev_${B}_VIPL.yaml ] || { echo "$(ts) TTA_FAIL $S no base yaml"; exit 1; }
python - <<PY || { echo "$(ts) TTA_FAIL $S yaml"; exit 1; }
import yaml
PFA = "$PFA"; c = yaml.safe_load(open(f"{PFA}/yaml_VIPL/ev_${B}_VIPL.yaml"))
for ax, run, csv in (("VIPL", f"{PFA}/runs/ev_${S}_VIPL", c["TEST"]["DATA"]["FILE_LIST_PATH"]), ("MMPD", f"{PFA}/runs/finetune_${S}", "$LISTS/mmpd_zs_7_33.csv")):
    d = yaml.safe_load(yaml.safe_dump(c)); d["LOG"]["PATH"] = run; d["TEST"]["OUTPUT_SAVE_DIR"] = run + "/outputs"; d["TEST"]["DATA"]["FILE_LIST_PATH"] = csv
    yaml.safe_dump(d, open(f"{PFA}/yaml_VIPL/ev_${S}_{ax}.yaml", "w"), sort_keys=False)
print("yaml ok", c["INFERENCE"]["MODEL_PATH"])
PY
for ax in VIPL MMPD; do
  out=$([ $ax = VIPL ] && echo $PFA/runs/ev_${S}_VIPL || echo $PFA/runs/finetune_${S})
  ls $out/*/saved_test_outputs/*.pickle >/dev/null 2>&1 && continue
  echo "$(ts) TTA $S $ax start"; ( export RPPG_CACHE_U8=1 RPPG_TTA_SLOW=1 $X; run_gated ${GATE:-6000} $PFA/yaml_VIPL/ev_${S}_${ax}.yaml /tmp/ev_${S}_${ax}.log ${S}_${ax} )
  grep -aq "\[tta\] slow active" /tmp/ev_${S}_${ax}.log || { echo "$(ts) TTA_FAIL $S $ax inactive"; exit 1; }
  ls $out/*/saved_test_outputs/*.pickle >/dev/null 2>&1 || { echo "$(ts) TTA_FAIL $S $ax no output"; exit 1; }
done
python scripts/agg_vipl.py $S > /tmp/agg_${S}_VIPL.log 2>&1 && python scripts/agg_mmpd_cv2.py $S > /tmp/agg_${S}_MMPD.log 2>&1 || { echo "$(ts) TTA_FAIL $S agg"; exit 1; }
echo "$(ts) TTA_DONE $S VIPL $(grep -E 'all \[문헌' /tmp/agg_${S}_VIPL.log | head -1 | grep -o 'MAE= *[0-9.]*') MMPD $(sed -n 2p /tmp/agg_${S}_MMPD.log | grep -o 'MAE= *[0-9.]*')"
