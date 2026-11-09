#!/bin/bash
# [67 문서] 체인 126 후처리(Alpha): pro6000 에서 깨끗한 분리 시험 지연 표 갱신 → 집계·지표·로그·체크포인트 회수 → 판정
set -e; ROOT=/SPECPHYS_ROOT; PFA=$ROOT/_workspace/data/physformer; RES=$ROOT/_workspace/data/results; H=user@train-host
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH; cd $ROOT
ssh $H 'docker exec -u user vface_train bash -lc "cd /SPECPHYS_ROOT && SEPX_SET=C /SPECPHYS_CONDA/envs/vface-rppg/bin/python scripts/sepx_lag.py | tail -1"'
rsync -az --include='agg_NF_NSP2C_L_s*' --include='sepxc_NF_*' --include='sepx_lag_c.json' --include='chain126_*' --exclude='*' $H:$RES/ $RES/
mkdir -p $RES/chain126_logs; rsync -az "$H:/SPECPHYS_TMP/container/chain126_P*.log" $RES/chain126_logs/ 2>/dev/null || true
for s in 2100 2200 2300 2400 2500; do mkdir -p $PFA/runs/finetune_NF_NSP2C_L_s$s/PreTrainedModels; rsync -az "$H:$PFA/runs/finetune_NF_NSP2C_L_s$s/PreTrainedModels/*Epoch9.pth" $PFA/runs/finetune_NF_NSP2C_L_s$s/PreTrainedModels/ || true; done
python scripts/cleanmask_judge.py
