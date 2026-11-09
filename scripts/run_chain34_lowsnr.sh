#!/bin/bash
# 체인 34 (18단계 §14): 저SNR 쌍 재학습 DLS + 파라메트릭 PARM + 도메인 조건부 게이트(DLSG/PARMG)
source /SPECPHYS_ROOT/scripts/chain_lib.sh
SCR=$ROOT/scripts/lowsnr_param.py
agg_cmp() { local M=$1; for S in "" _s200 _s300; do
  ( cd $ROOT && python scripts/agg_model2.py $M$S > /dev/null 2>&1; python scripts/agg_mmpd.py $M$S > /dev/null 2>&1 )
  echo "== cmp $M$S vs CoTAF_TW_L$S"; ( cd $ROOT && python scripts/cmp_model.py $M$S CoTAF_TW_L$S 2>/dev/null )
done; }
{
echo "$(date '+%m-%d %H:%M') CHAIN34 START"

# --- PARM (CPU, 학습 무) ---
python $SCR bank && echo BANK_OK || echo BANK_FAIL
for S in "" _s200 _s300; do
  python $SCR parm --suffix "$S" && echo PARM${S:-_s100}_OK || echo PARM${S:-_s100}_FAIL
done
python $SCR gate --model CoTAF_TW_PARM_L --out gate_parm.json && echo GATEPARM_OK || echo GATEPARM_FAIL
for S in "" _s200 _s300; do
  python $SCR applygate --suffix "$S" --model CoTAF_TW_PARM_L --gate gate_parm.json --name CoTAF_TW_PARMG_L --domain MMPD
done
agg_cmp CoTAF_TW_PARM_L
agg_cmp CoTAF_TW_PARMG_L
echo "$(date '+%m-%d %H:%M') PARM_BRANCH_DONE"

# --- DLS (GPU: 재학습 + DDIM 적용) ---
python $SCR pairs && echo PAIRS_OK || echo PAIRS_FAIL
wait_gpu 8000
python $SCR train --steps 20000 && echo TRAINLS_OK || echo TRAINLS_FAIL
for S in "" _s200 _s300; do
  wait_gpu 6000
  python $SCR apply --suffix "$S" && echo APPLYDLS${S:-_s100}_OK || echo APPLYDLS${S:-_s100}_FAIL
done
python $SCR gate --model CoTAF_TW_DLS_L --out gate_ls.json && echo GATELS_OK || echo GATELS_FAIL
for S in "" _s200 _s300; do
  python $SCR applygate --suffix "$S" --model CoTAF_TW_DLS_L --gate gate_ls.json --name CoTAF_TW_DLSG_L --domain MMPD
done
agg_cmp CoTAF_TW_DLS_L
agg_cmp CoTAF_TW_DLSG_L
echo "$(date '+%m-%d %H:%M') CHAIN34 DONE"
} >> /tmp/chain34.log 2>&1
