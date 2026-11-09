#!/bin/bash
# [69 문서] 체인 128 후처리(Alpha): 집계·통제 시험 지표·로그·체크포인트 회수 → 판정·그림
set -e; ROOT=/SPECPHYS_ROOT; PFA=$ROOT/_workspace/data/physformer; RES=$ROOT/_workspace/data/results; H=user@train-host
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH; cd $ROOT
rsync -az --include='agg_NF_NSP2Q_L_s*' --include='agg_NF_NSP2QN_L_s*' --include='agg_NF_NSP2SO_L_s*' --include='hrsweep*' --include='chain128_*' --exclude='*' $H:$RES/ $RES/
mkdir -p $RES/chain128_logs; rsync -az "$H:/SPECPHYS_TMP/container/chain128_P*.log" $RES/chain128_logs/ 2>/dev/null || true
for c in NF_NSP2Q_L NF_NSP2QN_L NF_NSP2SO_L; do for s in 2100 2200 2300 2400 2500; do mkdir -p $PFA/runs/finetune_${c}_s$s/PreTrainedModels; rsync -az "$H:$PFA/runs/finetune_${c}_s$s/PreTrainedModels/*Epoch9.pth" $PFA/runs/finetune_${c}_s$s/PreTrainedModels/ || true; done; done
python scripts/contrib_judge.py > /dev/null; python scripts/contrib_figs.py; sed -n '/가족 D/,$p' $RES/contrib_judge.txt
