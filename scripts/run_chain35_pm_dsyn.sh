#!/bin/bash
# 체인 35: (B) 대규모 저SNR 합성 쌍 refiner DSYN/DSYNG → (A) PhysMamba 백본 비교 (설계서 §16, 16문서 §19 예정)
source /SPECPHYS_ROOT/scripts/chain_lib.sh
SCR=$ROOT/scripts/lowsnr_param.py
agg_cmp() { local M=$1; for S in "" _s200 _s300; do
  ( cd $ROOT && python scripts/agg_model2.py $M$S > /dev/null 2>&1; python scripts/agg_mmpd.py $M$S > /dev/null 2>&1 )
  echo "== cmp $M$S vs CoTAF_TW_L$S"; ( cd $ROOT && python scripts/cmp_model.py $M$S CoTAF_TW_L$S 2>/dev/null )
done; }

# gen_pm M POOL cache_tag PRE(공유 평가 캐시 전처리 여부) — yaml24 기반 PhysFormer→PhysMamba 변환
gen_pm() { local M=$1 P=$2 c=$3 PRE=$4
  local Y=$(gen_yaml $M $P $c pm)
  sed -i -E -e "s/NAME: PhysFormer/NAME: PhysMamba/" \
    -e "s#UBFC-rPPG_PhysFormer_DiffNormalized.pth#UBFC-rPPG_PhysMamba_DiffNormalized.pth#" \
    -e "s/LR: 5e-5/LR: 1e-4/" \
    -e "s/CHUNK_LENGTH: 160/CHUNK_LENGTH: 128/" \
    -e "s/EXP_DATA_NAME: real_/EXP_DATA_NAME: pm_/" \
    -e "s/EXP_DATA_NAME: synth_/EXP_DATA_NAME: pm_s_/" $Y/*.yaml
  for f in $Y/ev_${M}_*.yaml; do sed -i -E "s/DO_PREPROCESS: (true|false)/DO_PREPROCESS: $PRE/" "$f"; done
  # finetune: train 섹션은 항상 true(자기 풀), valid/test는 PRE 따름(공유 pm_s_* 캐시)
  python - "$Y/finetune_$M.yaml" "$PRE" <<'PYEOF'
import re, sys
p, pre = sys.argv[1], sys.argv[2]
s = open(p).read()
parts = re.split(r"(DO_PREPROCESS: \w+)", s)
n = 0
for i, seg in enumerate(parts):
    if seg.startswith("DO_PREPROCESS:"):
        n += 1
        parts[i] = "DO_PREPROCESS: " + ("true" if n == 1 else pre)
open(p, "w").write("".join(parts))
PYEOF
  echo "GENPM $M pool=$P cache=$c pre=$PRE"; }

{
echo "$(date '+%m-%d %H:%M') CHAIN35 START"

# ---------- B: 대규모 저SNR 합성 쌍 (DSYN) ----------
python $SCR pairs_syn --k 12 && echo PAIRSSYN_OK || echo PAIRSSYN_FAIL
wait_gpu 8000
python $SCR train_syn --steps 40000 && echo TRAINSYN_OK || echo TRAINSYN_FAIL
for S in "" _s200 _s300; do
  wait_gpu 6000
  python $SCR apply_syn --suffix "$S" && echo APPLYSYN${S:-_s100}_OK || echo APPLYSYN${S:-_s100}_FAIL
done
python $SCR gate --model CoTAF_TW_DSYN_L --out gate_syn.json && echo GATESYN_OK || echo GATESYN_FAIL
for S in "" _s200 _s300; do
  python $SCR applygate --suffix "$S" --model CoTAF_TW_DSYN_L --gate gate_syn.json --name CoTAF_TW_DSYNG_L --domain MMPD
done
agg_cmp CoTAF_TW_DSYN_L
agg_cmp CoTAF_TW_DSYNG_L
echo "$(date '+%m-%d %H:%M') B_BRANCH_DONE"

# ---------- A: PhysMamba 백본 비교 ----------
python -c "from mamba_ssm import Mamba; print('mamba import OK')" || { echo MAMBA_IMPORT_FAIL; }
gen_pm PM_CoTAF_L ALLF allf_pm true
gen_pm PM_CoTAF_TW_L OFFICIAL official_pm false
for M in PM_CoTAF_L PM_CoTAF_TW_L; do
  c=allf_pm; [ "$M" = PM_CoTAF_TW_L ] && c=official_pm
  drop_cache cotrain_$c
  run_gated 21000 "$PFA/yaml_$M/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN
  drop_cache cotrain_$c
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
done
# 13축: 모델1이 pm_* 캐시 전처리, 모델2 재사용
for M in PM_CoTAF_L PM_CoTAF_TW_L; do
  for f in $PFA/yaml_$M/ev_${M}_*.yaml; do
    A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*) continue;; esac
    run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A
  done
done
# MMPD 6배치: 배치별 모델1(전처리)→모델2(재사용)→캐시 드롭
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
echo "$(date '+%m-%d %H:%M') CHAIN35 DONE"
} >> /tmp/chain35.log 2>&1
