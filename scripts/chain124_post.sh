#!/bin/bash
# [65 문서] 체인 124 후처리(Alpha): pro6000 집계·분리 시험 지표·로그·체크포인트 회수 → 판정
set -e; ROOT=/SPECPHYS_ROOT; PFA=$ROOT/_workspace/data/physformer; RES=$ROOT/_workspace/data/results; H=user@train-host
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH; cd $ROOT
rsync -az --include='agg_NF_NSP2DR_L_s*' --include='agg_NF_LSUB_L_s*' --include='agg_NF_LSUB0_L_s*' --include='sepx_NF_*' --include='chain124_*' --exclude='*' $H:$RES/ $RES/
mkdir -p $RES/chain124_logs; rsync -az "$H:/SPECPHYS_TMP/container/chain124_P*.log" $RES/chain124_logs/ 2>/dev/null || true
for c in NF_NSP2DR_L NF_LSUB_L NF_LSUB0_L; do for s in 2600 2700 2800 2900 3000; do mkdir -p $PFA/runs/finetune_${c}_s$s/PreTrainedModels; rsync -az "$H:$PFA/runs/finetune_${c}_s$s/PreTrainedModels/*Epoch9.pth" $PFA/runs/finetune_${c}_s$s/PreTrainedModels/ || true; done; done
python scripts/lsub_judge.py
