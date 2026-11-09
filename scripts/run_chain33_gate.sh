#!/bin/bash
# 체인 33 (18단계): confidence-gated selective refinement — 학습 분할 τ 선택 후 per-video 게이트 결합
source /SPECPHYS_ROOT/scripts/chain_lib.sh
{
echo "$(date '+%m-%d %H:%M') CHAIN33 START"
wait_gpu 6000
python $ROOT/scripts/selective_refine.py calibrate && echo CALIB_OK || echo CALIB_FAIL
for S in "" _s200 _s300; do
  python $ROOT/scripts/selective_refine.py apply --suffix "$S" && echo APPLY${S:-_s100}_OK || echo APPLY${S:-_s100}_FAIL
  ( cd $ROOT && python scripts/agg_model2.py CoTAF_TW_GATE_L$S && python scripts/agg_mmpd.py CoTAF_TW_GATE_L$S )
  echo "== cmp CoTAF_TW_GATE_L$S vs CoTAF_TW_L$S"
  ( cd $ROOT && python scripts/cmp_model.py CoTAF_TW_GATE_L$S CoTAF_TW_L$S )
done
echo "$(date '+%m-%d %H:%M') CHAIN33 DONE"
} >> /tmp/chain33.log 2>&1
