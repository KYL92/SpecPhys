#!/bin/bash
# 체인 32 (18단계 확장): diffusion 변형 4종 스윕 — REG(회귀 ablation)/X0/DIT/W600. 체인 31 완료 대기 후 시작.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
{
echo "$(date '+%m-%d %H:%M') CHAIN32 WAIT (chain31)"
until grep -q "CHAIN31 DONE" /tmp/chain31.log 2>/dev/null; do sleep 300; done
echo "$(date '+%m-%d %H:%M') CHAIN32 START"
python $ROOT/scripts/diffusion_refiner_v2.py pairs --win 600 && echo PAIRS600_OK || echo PAIRS600_FAIL
for spec in "REG reg unet eps 300" "X0 ddpm unet x0 300" "DIT ddpm dit eps 300" "W600 ddpm unet eps 600"; do
  set -- $spec; TAG=$1; MODE=$2; ARCH=$3; PARAM=$4; WIN=$5
  wait_gpu 6000
  python $ROOT/scripts/diffusion_refiner_v2.py train --tag $TAG --mode $MODE --arch $ARCH --param $PARAM --win $WIN --steps 20000 && echo TRAIN_${TAG}_OK || { echo TRAIN_${TAG}_FAIL; continue; }
  wait_gpu 6000
  python $ROOT/scripts/diffusion_refiner_v2.py apply --tag $TAG --src CoTAF_TW_L --dst CoTAF_TW_D${TAG}_L && echo APPLY_${TAG}_OK || { echo APPLY_${TAG}_FAIL; continue; }
  ( cd $ROOT && python scripts/agg_model2.py CoTAF_TW_D${TAG}_L && python scripts/agg_mmpd.py CoTAF_TW_D${TAG}_L )
  echo "== cmp CoTAF_TW_D${TAG}_L vs CoTAF_TW_L"
  ( cd $ROOT && python scripts/cmp_model.py CoTAF_TW_D${TAG}_L CoTAF_TW_L )
done
echo "== cmp 기본 head(CoTAF_TW_DIFF_L) vs 변형들 참고: agg JSON 비교는 cmp_model로 개별 수행"
echo "$(date '+%m-%d %H:%M') CHAIN32 DONE"
} >> /tmp/chain32.log 2>&1
