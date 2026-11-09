#!/bin/bash
# [체인 75, 59 문서] pro6000 → Alpha 결과 회수(작은 파일만). Alpha 에서: bash scripts/pull_chain75.sh
set -e; H=user@train-host; RES=/SPECPHYS_ROOT/_workspace/data/results; mkdir -p $RES/chain75_logs
rsync -az --include='agg_RF_SPEC_L*' --include='agg_PM_SPEC_L*' --include='agg_PM_POOL_EP_L*' --include='agg_PM_POOL_LR5_L*' --include='crossbb_judge.*' --include='chain75_*' --exclude='*' $H:$RES/ $RES/
rsync -az "$H:/SPECPHYS_TMP/container/chain75_*.log" $RES/chain75_logs/ 2>/dev/null || true
rsync -az "$H:/SPECPHYS_TMP/container/ft_RF_SPEC_L*.log" "$H:/SPECPHYS_TMP/container/ft_PM_SPEC_L*.log" "$H:/SPECPHYS_TMP/container/ft_PM_POOL_EP_L.log" "$H:/SPECPHYS_TMP/container/ft_PM_POOL_LR5_L.log" $RES/chain75_logs/ 2>/dev/null || true
ls $RES | grep -E "^agg_(RF_SPEC|PM_SPEC|PM_POOL_EP|PM_POOL_LR5)"; echo PULLED
