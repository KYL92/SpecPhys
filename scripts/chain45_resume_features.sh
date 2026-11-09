#!/bin/bash
# 체인 45 보조: MMPD 소프 평가(메모리 피크) 종료 후 관측 특징 추출 재개(3워커) → 마커 → 체인이 라우터 단계로 진행
until grep -q "AGG_CoTAM_HYB4_SOUP_L_DONE" /tmp/chain45.log; do sleep 120; done
cd /SPECPHYS_ROOT
/SPECPHYS_CONDA/envs/vface-rppg/bin/python scripts/mmpd_obs_features.py 3 >> /tmp/chain45_feat.log 2>&1 && touch /tmp/chain45_feat.done
