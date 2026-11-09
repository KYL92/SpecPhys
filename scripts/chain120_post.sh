#!/bin/bash
# [62 §10] 체인 120 후처리(Alpha): pro6000 집계·출력 회수 → 시드마다 라우팅(70/105) NF_ENS_N_L → 판정
set -e; ROOT=/SPECPHYS_ROOT; PFA=$ROOT/_workspace/data/physformer; RES=$ROOT/_workspace/data/results; H=user@train-host
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH; cd $ROOT
rsync -az --include='agg_NF_NSP2_L*' --include='agg_NF_NSP2H_L*' --include='agg_NF_NSP2P_L*' --include='chain120_*' --exclude='*' $H:$RES/ $RES/
rsync -az --prune-empty-dirs --include='finetune_NF_NSP2*_L_s*/' --include='finetune_NF_NSP2*_L_s*/vipl_u8/' --include='finetune_NF_NSP2*_L_s*/vipl_u8/saved_test_outputs/' --include='finetune_NF_NSP2*_L_s*/vipl_u8/saved_test_outputs/*.pickle' --include='ev_NF_NSP2*_L_s*/' --include='ev_NF_NSP2*_L_s*/*/' --include='ev_NF_NSP2*_L_s*/*/saved_test_outputs/' --include='ev_NF_NSP2*_L_s*/*/saved_test_outputs/*.pickle' --exclude='*' $H:$PFA/runs/ $PFA/runs/
for S in 600 700 800 900 1000; do python scripts/ensemble_cells.py NF_ENS_N_L_s$S NF_NSP2_L_s$S NF_NSP2H_L_s$S --gate 70 105 | grep -E "ENSEMBLE_DONE|skip|Error" | cut -c1-170; done
python scripts/nsp_final_judge.py
