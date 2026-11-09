#!/bin/bash
# [66 문서] 체인 125 후처리(Alpha): pro6000 에서 지연 표 갱신 → 집계·분리 지표·로그·체크포인트 회수 → 판정
set -e; ROOT=/SPECPHYS_ROOT; PFA=$ROOT/_workspace/data/physformer; RES=$ROOT/_workspace/data/results; H=user@train-host
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH; cd $ROOT
ssh $H 'docker exec -u user vface_train bash -lc "cd /SPECPHYS_ROOT && /SPECPHYS_CONDA/envs/vface-rppg/bin/python scripts/sepx_lag.py | tail -1"'
rsync -az --include='agg_NF_BGREF_L_s*' --include='agg_NF_NSP2DR_BGT_L_s*' --include='sepx_NF_BGREF_L_s*' --include='sepx_NF_NSP2DR_BGT_L_s*' --include='sepx_lag.json' --include='chain125_*' --exclude='*' $H:$RES/ $RES/
mkdir -p $RES/chain125_logs; rsync -az "$H:/SPECPHYS_TMP/container/chain125_P*.log" $RES/chain125_logs/ 2>/dev/null || true
for s in 2600 2700 2800 2900 3000; do mkdir -p $PFA/runs/finetune_NF_BGREF_L_s$s/PreTrainedModels; rsync -az "$H:$PFA/runs/finetune_NF_BGREF_L_s$s/PreTrainedModels/*Epoch9.pth" $PFA/runs/finetune_NF_BGREF_L_s$s/PreTrainedModels/ || true; done
python scripts/bgref_judge.py
