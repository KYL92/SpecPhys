#!/bin/bash
# [61 §14] 체인 115 후처리(Alpha): pro6000 fold 출력 회수 → fold 마다 설정 C 라우팅(70/105) → 5-fold CV 집계(A·B·C × 시드) → in-domain 판정
set -e; ROOT=/SPECPHYS_ROOT; PFA=$ROOT/_workspace/data/physformer; RES=$ROOT/_workspace/data/results; H=user@train-host
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH; cd $ROOT
rsync -az --include='finetune_MF[1-5]_SK[AB]_L_s*/' --include='finetune_MF[1-5]_SK[AB]_L_s*/vipl_u8/' --include='finetune_MF[1-5]_SK[AB]_L_s*/vipl_u8/saved_test_outputs/' --include='finetune_MF[1-5]_SK[AB]_L_s*/vipl_u8/saved_test_outputs/*.pickle' --exclude='*' $H:$PFA/runs/ $PFA/runs/
rsync -az --include='agg_MF[1-5]_SK[AB]_L_s*' --include='chain115_*' --exclude='*' $H:$RES/ $RES/
rsync -az "$H:/SPECPHYS_TMP/container/chain115_P*.log" $RES/chain87_logs/ 2>/dev/null || true; cp /tmp/chain115_A*.log $RES/chain87_logs/ 2>/dev/null || true
for S in 600 700 800; do for K in 1 2 3 4 5; do
  for C in SKA SKB; do ls $PFA/runs/finetune_MF${K}_${C}_L_s$S/vipl_u8/saved_test_outputs/*.pickle > /dev/null 2>&1 || { echo "MISSING MF${K}_${C}_L_s$S"; exit 1; }; done
  python scripts/ensemble_cells.py MF${K}_SKC_L_s$S MF${K}_SKA_L_s$S MF${K}_SKB_L_s$S --gate 70 105 | grep -E "ENSEMBLE_DONE|skip|Error" | cut -c1-160
done; python scripts/agg_mmpd_cv2.py --cv SKA_L_s$S SKB_L_s$S SKC_L_s$S 2>&1 | grep -E "^--- |all " | cut -c1-140; done
python scripts/mmpd_indomain_judge.py
