#!/bin/bash
# [65 문서] 패치 56 1에폭 시험 학습(판정에 쓰지 않음): 변환 일치 점검 → 학습된 잡음 빼기 + 분리 감독 1에폭(시드 1, 시험은 분리 세트 264청크)
source /SPECPHYS_ROOT/scripts/chain_lib.sh; cd $ROOT; L=$PFA/cache/DataFileLists
( cd repos/rPPG-Toolbox && RPPG_CACHE_U8=1 RPPG_SKINW=0.2 python ../../scripts/lsub_smoke.py 2>&1 | grep -E "chunks|max|LSUB_SMOKE|Error|assert" )
M=NF_LSUB_SMOKE; python scripts/vipl_yaml.py train $M 1 $L/cotrain_allf_hyb4_dr_u8_pool.csv 1 --test_csv $L/sepeval_u8.csv --template $PFA/yaml_NF_HYB_L/finetune_NF_HYB_L.yaml --lr 1e-4 > /dev/null
export RPPG_NF=1 RPPG_SKINW=0.2 NF_SS_TRACE=1 NF_SS_REF=1 NF_ARCH=nsp NF_NSP_LAM=0.01 HYB_NOISE_GT_DIR=$PFA/noise_gt_dr NF_LSUB=1 NF_LSUB_SUP=1 RPPG_CACHE_U8=1 NF_W_DBL=0.5 RPPG_TAUG=1 RPPG_TAUG_P=0.25 RPPG_TAUG_HR_SPLIT=300 RPPG_SEED=1
run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN
grep -aE "\[lsub\]|LSUB stats|NF stats epoch 0|nan|Error|Traceback" /tmp/ft_$M.log | cut -c1-220 | head -20; touch /tmp/lsub_smoke.done
