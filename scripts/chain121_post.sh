#!/bin/bash
# [62 §12] 체인 121 후처리(Alpha): pro6000 집계·출력 회수 → 시드마다 현재 최종 설정 라우팅(70/105) NF_ENS_AB_L → 판정
set -e; ROOT=/SPECPHYS_ROOT; PFA=$ROOT/_workspace/data/physformer; RES=$ROOT/_workspace/data/results; H=user@train-host
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH; cd $ROOT
rsync -az --include='agg_NF_NSP2_L_s2[1-5]00*' --include='agg_NF_ORIGA_L_s2[1-5]00*' --include='agg_NF_SKREF_REP_L_s2[1-5]00*' --include='agg_NF_SKREF_X15_L_s2[1-5]00*' --include='chain121_*' --exclude='*' $H:$RES/ $RES/
rsync -az --prune-empty-dirs --include='finetune_NF_SKREF_*_L_s2[1-5]00/' --include='finetune_NF_SKREF_*_L_s2[1-5]00/vipl_u8/' --include='finetune_NF_SKREF_*_L_s2[1-5]00/vipl_u8/saved_test_outputs/' --include='finetune_NF_SKREF_*_L_s2[1-5]00/vipl_u8/saved_test_outputs/*.pickle' --include='ev_NF_SKREF_*_L_s2[1-5]00_*/' --include='ev_NF_SKREF_*_L_s2[1-5]00_*/*/' --include='ev_NF_SKREF_*_L_s2[1-5]00_*/*/saved_test_outputs/' --include='ev_NF_SKREF_*_L_s2[1-5]00_*/*/saved_test_outputs/*.pickle' --exclude='*' $H:$PFA/runs/ $PFA/runs/
for S in 2100 2200 2300 2400 2500; do python scripts/ensemble_cells.py NF_ENS_AB_L_s$S NF_SKREF_REP_L_s$S NF_SKREF_X15_L_s$S --gate 70 105 | grep -E "ENSEMBLE_DONE|skip|Error" | cut -c1-170; done
python scripts/nsp_decisive_judge.py
