#!/bin/bash
# [62 §14] 체인 122 후처리(Alpha): pro6000 집계 회수 → MMPD in-domain 5-fold CV 집계 → 판정
set -e; ROOT=/SPECPHYS_ROOT; PFA=$ROOT/_workspace/data/physformer; RES=$ROOT/_workspace/data/results; H=user@train-host
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH; cd $ROOT
rsync -az --include='agg_MF[1-5]_SKN_L_s*' --include='chain122_*' --exclude='*' $H:$RES/ $RES/
rsync -az "$H:/SPECPHYS_TMP/container/chain122_P*.log" $RES/chain87_logs/ 2>/dev/null || true; cp /tmp/chain122_[AB].log $RES/chain87_logs/ 2>/dev/null || true
# --cv 는 run 의 pickle 을 읽으므로 pickle 이 있는 pro6000 컨테이너에서 돌리고 결과만 회수한다(Alpha 에서 돌리면 "MF1_SKN_L_s600 empty")
ssh $H "docker exec -u user vface_train bash -lc 'cd $ROOT && /SPECPHYS_CONDA/envs/vface-rppg/bin/python scripts/agg_mmpd_cv2.py --cv SKN_L_s600 SKN_L_s700 SKN_L_s800'" 2>&1 | grep -E "^--- |^ *all |Traceback|Error" | cut -c1-140
rsync -az --include='agg_MCV5_SKN_L_s*' --exclude='*' $H:$RES/ $RES/
python scripts/nsp_abl_judge.py
