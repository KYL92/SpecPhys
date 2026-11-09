#!/bin/bash
# 체인 41b: conf 덤프 버그(수정 완료) 여파 평가 재실행 — 학습 3시드 재사용, 평가+집계+분석만
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_NOISE_HEAD=1 RPPG_CONF_DUMP=1
{
echo "$(date '+%m-%d %H:%M') CHAIN41B START (eval 재실행)"
for SEED in 100 200 300; do
  S=""; [ "$SEED" != "100" ] && S=_s$SEED
  M=CoTAM_HYB5_L$S
  eval_only $M allf_hyb5 hyb5
  echo "== cmp $M vs CoTAF_TW_L$S"
  ( cd $ROOT && python scripts/cmp_model.py $M CoTAF_TW_L$S 2>/dev/null )
  ( cd $ROOT && python scripts/motion_split_cmp.py $M CoTAF_TW_L$S 2>/dev/null )
done
( cd $ROOT && python scripts/conf_routing.py CoTAM_HYB5_L )
echo "$(date '+%m-%d %H:%M') CHAIN41B DONE"
} >> /tmp/chain41.log 2>&1
