#!/bin/bash
# [64 문서] 체인 123 후처리(Alpha): pro6000 집계·로그·체크포인트 회수 → 판정
set -e; ROOT=/SPECPHYS_ROOT; PFA=$ROOT/_workspace/data/physformer; RES=$ROOT/_workspace/data/results; H=user@train-host
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH; cd $ROOT
rsync -az --include='agg_NF_NSP2NL_L_s*' --include='agg_NF_NSP2NS_L_s*' --include='chain123_*' --exclude='*' $H:$RES/ $RES/
rsync -az "$H:/SPECPHYS_TMP/container/chain123_P*.log" $RES/chain87_logs/ 2>/dev/null || true
for c in NF_NSP2NL_L NF_NSP2NS_L; do for s in 2100 2200 2300 2400 2500; do mkdir -p $PFA/runs/finetune_${c}_s$s/PreTrainedModels; rsync -az "$H:$PFA/runs/finetune_${c}_s$s/PreTrainedModels/*Epoch9.pth" $PFA/runs/finetune_${c}_s$s/PreTrainedModels/; done; done
python scripts/nsp_contrib_judge.py
