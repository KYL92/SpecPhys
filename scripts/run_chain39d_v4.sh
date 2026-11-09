#!/bin/bash
# 체인 39d (V4): V3 + MMPD 지정 학습분할(1~6, 120클립) 결합 — in-domain 최강 지렛대의 하이브리드 증폭 검정
# CoTAM 계열 표기(MMPD train-split 사용). 판정: 움직임 분할 + held-out 7~33.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_DN_ONLY=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25
export HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2
{
echo "$(date '+%m-%d %H:%M') CHAIN39D WAIT chain39c"
for i in $(seq 1 720); do grep -q "CHAIN39C DONE" /tmp/chain39.log 2>/dev/null && break; sleep 60; done
echo "$(date '+%m-%d %H:%M') CHAIN39D START (V4: +MMPD 1~6, CoTAM 계열)"
make_pool ALLF_HYB4 $PFA/COTRAIN_OFFICIAL $PFA/TRAIN24DN $PFA/TRAIN24DN2 $PFA/MMPD_TRAIN
for SEED in 100 200 300; do
  S=""; [ "$SEED" != "100" ] && S=_s$SEED
  M=CoTAM_HYB4_L$S
  gen_yaml $M ALLF_HYB4 allf_hyb4 hyb4 > /dev/null
  train_eval $M allf_hyb4 hyb4 $SEED
  echo "== cmp $M vs CoTAF_TW_L$S"
  ( cd $ROOT && python scripts/cmp_model.py $M CoTAF_TW_L$S 2>/dev/null )
  ( cd $ROOT && python scripts/motion_split_cmp.py $M CoTAF_TW_L$S 2>/dev/null )
done
echo "$(date '+%m-%d %H:%M') CHAIN39D DONE"
} >> /tmp/chain39.log 2>&1
