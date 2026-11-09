#!/bin/bash
# [체인 74, 57 문서] pro6000 → Alpha 결과 회수(작은 파일만: agg json·학습 로그·판정·체인 로그). Alpha 에서 실행: bash scripts/pull_chain74.sh
set -e; H=user@train-host; RES=/SPECPHYS_ROOT/_workspace/data/results; mkdir -p $RES/chain74_logs
rsync -az --include='agg_RF_POOL_L*' --include='agg_PM_POOL_L*' --include='agg_PMbase*' --include='agg_ML_NF_L*' --include='agg_ML_NFO_L*' --include='backbone_judge.*' --include='chain74_*' --exclude='*' $H:$RES/ $RES/
rsync -az "$H:/SPECPHYS_TMP/container/chain74_*.log" $RES/chain74_logs/ 2>/dev/null || true
rsync -az "$H:/SPECPHYS_TMP/container/ft_RF_POOL_L*.log" "$H:/SPECPHYS_TMP/container/ft_PM_POOL_L*.log" "$H:/SPECPHYS_TMP/container/ft_ML_NF*.log" $RES/chain74_logs/ 2>/dev/null || true
ls $RES | grep -E "^agg_(RF_POOL|PM_POOL|PMbase|ML_NF)" ; echo PULLED
