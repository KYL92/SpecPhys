#!/bin/bash
# 체인 39b (V2): V1의 깨끗한 축 손상 원인 제거 — SiNC 전역 사전 OFF, 탈상관 0.25로 완화.
# 노이즈 GT 이중 감독(사용자 설계 핵심)은 유지. 코퍼스·사이드카 재사용(재빌드 없음).
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=0 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25
export HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb
{
echo "$(date '+%m-%d %H:%M') CHAIN39B WAIT chain39"
for i in $(seq 1 720); do grep -q "CHAIN39 DONE" /tmp/chain39.log 2>/dev/null && break; sleep 60; done
echo "$(date '+%m-%d %H:%M') CHAIN39B START (V2: SiNC off, decor 0.25)"
for SEED in 100 200 300; do
  S=""; [ "$SEED" != "100" ] && S=_s$SEED
  M=CoTAF_HYB2_L$S
  gen_yaml $M ALLF_HYB allf_hyb hyb2 > /dev/null
  train_eval $M allf_hyb hyb2 $SEED
  echo "== cmp $M vs CoTAF_TW_L$S"
  ( cd $ROOT && python scripts/cmp_model.py $M CoTAF_TW_L$S 2>/dev/null )
done
echo "$(date '+%m-%d %H:%M') CHAIN39B DONE"
} >> /tmp/chain39.log 2>&1
