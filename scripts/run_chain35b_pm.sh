#!/bin/bash
# 체인 35b: 체인 35의 A 파트 재개 — PhysMamba 백본 비교 (yaml_PM_* 은 이미 생성됨)
source /SPECPHYS_ROOT/scripts/chain_lib.sh
{
echo "$(date '+%m-%d %H:%M') CHAIN35B START"
python -c "from mamba_ssm import Mamba; print('mamba import OK')" || echo MAMBA_IMPORT_FAIL
for M in PM_CoTAF_L PM_CoTAF_TW_L; do
  c=allf_pm; [ "$M" = PM_CoTAF_TW_L ] && c=official_pm
  drop_cache cotrain_$c
  run_gated 21000 "$PFA/yaml_$M/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN
  drop_cache cotrain_$c
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
done
for M in PM_CoTAF_L PM_CoTAF_TW_L; do
  for f in $PFA/yaml_$M/ev_${M}_*.yaml; do
    A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*) continue;; esac
    run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A
  done
done
for k in 1 2 3 4 5 6; do
  run_gated 6000 "$PFA/yaml_PM_CoTAF_L/ev_PM_CoTAF_L_MMPD_B$k.yaml" /tmp/ev_PM_CoTAF_L_MMPD_B$k.log PM_CoTAF_L_MMPD_B$k
  run_gated 6000 "$PFA/yaml_PM_CoTAF_TW_L/ev_PM_CoTAF_TW_L_MMPD_B$k.yaml" /tmp/ev_PM_CoTAF_TW_L_MMPD_B$k.log PM_CoTAF_TW_L_MMPD_B$k
  drop_cache mmpd_pm_B$k
done
for M in PM_CoTAF_L PM_CoTAF_TW_L; do
  ( cd $ROOT && python scripts/agg_model2.py $M > /dev/null 2>&1; python scripts/agg_mmpd.py $M > /dev/null 2>&1 )
done
echo "== cmp PM_CoTAF_TW_L vs PM_CoTAF_L (백본 내 TW 효과)"
( cd $ROOT && python scripts/cmp_model.py PM_CoTAF_TW_L PM_CoTAF_L 2>/dev/null )
echo "== cmp PM_CoTAF_TW_L vs CoTAF_TW_L (백본 간)"
( cd $ROOT && python scripts/cmp_model.py PM_CoTAF_TW_L CoTAF_TW_L 2>/dev/null )
rm -rf $PFA/cache/pm_* ; rm -f $PFA/cache/DataFileLists/pm_*.csv
echo "$(date '+%m-%d %H:%M') CHAIN35B DONE"
} >> /tmp/chain35.log 2>&1
