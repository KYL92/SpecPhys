#!/bin/bash
# 체인 31 (18단계): diffusion waveform refinement head — 설계서 _workspace/18_diffusion_refiner_design.md
source /SPECPHYS_ROOT/scripts/chain_lib.sh
{
echo "$(date '+%m-%d %H:%M') CHAIN31 START"
# S1: 학습 풀(공식 코퍼스 203) backbone 덤프 — 평가 게이트 6000MB
run_gated 6000 $PFA/yaml_CoTAF_TW_L/ev_CoTAF_TW_L_TRAINPOOL.yaml /tmp/ev_TW_TRAINPOOL.log TW_TRAINPOOL
# S2: pulse-space 창 쌍 생성 → 캐시 즉시 삭제
python $ROOT/scripts/diffusion_refiner.py pairs && echo PAIRS_OK || echo PAIRS_FAIL
drop_cache cotrain_dump_tw
# S3: head 학습 (게이트)
wait_gpu 6000
python $ROOT/scripts/diffusion_refiner.py train --steps 20000 && echo TRAIN_OK || echo TRAIN_FAIL
# S4: 시드 3종 backbone pickle에 적용
wait_gpu 6000
for S in "" _s200 _s300; do
  python $ROOT/scripts/diffusion_refiner.py apply --src CoTAF_TW_L$S --dst CoTAF_TW_DIFF_L$S && echo APPLY${S:-_s100}_OK || echo APPLY${S:-_s100}_FAIL
done
# S5: 집계·비교 (정본 프로토콜 그대로)
cd $ROOT
for M in CoTAF_TW_DIFF_L CoTAF_TW_DIFF_L_s200 CoTAF_TW_DIFF_L_s300; do
  python scripts/agg_model2.py $M; python scripts/agg_mmpd.py $M
done
echo "== cmp vs CoTAF_TW_L (seed100)"; python scripts/cmp_model.py CoTAF_TW_DIFF_L CoTAF_TW_L
echo "== cmp vs CoTAF_TW_L_s200"; python scripts/cmp_model.py CoTAF_TW_DIFF_L_s200 CoTAF_TW_L_s200
echo "== cmp vs CoTAF_TW_L_s300"; python scripts/cmp_model.py CoTAF_TW_DIFF_L_s300 CoTAF_TW_L_s300
echo "$(date '+%m-%d %H:%M') CHAIN31 DONE"
} >> /tmp/chain31.log 2>&1
