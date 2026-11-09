#!/bin/bash
# [68 문서] 체인 127 후처리(Alpha): 집계·통제 시험 지표·로그·체크포인트 회수 → 판정
set -e; ROOT=/SPECPHYS_ROOT; PFA=$ROOT/_workspace/data/physformer; RES=$ROOT/_workspace/data/results; H=user@train-host
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH; cd $ROOT
rsync -az --include='agg_NF_NSP2K_L_s*' --include='agg_NF_NSP2R_L_s*' --include='agg_NF_NSP2W_L_s*' --include='hrsweep*_NF_*' --include='chain127_*' --exclude='*' $H:$RES/ $RES/
mkdir -p $RES/chain127_logs; rsync -az "$H:/SPECPHYS_TMP/container/chain127_P*.log" $RES/chain127_logs/ 2>/dev/null || true
for c in NF_NSP2K_L NF_NSP2R_L NF_NSP2W_L; do for s in 2100 2200 2300 2400 2500; do mkdir -p $PFA/runs/finetune_${c}_s$s/PreTrainedModels; rsync -az "$H:$PFA/runs/finetune_${c}_s$s/PreTrainedModels/*Epoch9.pth" $PFA/runs/finetune_${c}_s$s/PreTrainedModels/ || true; done; done
python scripts/idhr_judge.py
