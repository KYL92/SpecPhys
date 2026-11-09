#!/bin/bash
# [65 문서] 분리 손실 가중 공학 점검(판정에 쓰지 않음, 시험 출력은 보지 않는다): 2에폭, 시드 1, 학습 손실(sep·r0·spec)만 비교
source /SPECPHYS_ROOT/scripts/chain_lib.sh; cd $ROOT; L=$PFA/cache/DataFileLists; W=$1; M=NF_LSUB_SMOKE_W$W
python scripts/vipl_yaml.py train $M 1 $L/cotrain_allf_hyb4_dr_u8_pool.csv 2 --test_csv $L/sepeval_u8.csv --template $PFA/yaml_NF_HYB_L/finetune_NF_HYB_L.yaml --lr 1e-4 > /dev/null
export RPPG_NF=1 RPPG_SKINW=0.2 NF_SS_TRACE=1 NF_SS_REF=1 NF_ARCH=nsp NF_NSP_LAM=0.01 HYB_NOISE_GT_DIR=$PFA/noise_gt_dr NF_LSUB=1 NF_LSUB_SUP=1 NF_LSUB_W=$W RPPG_CACHE_U8=1 NF_W_DBL=0.5 RPPG_TAUG=1 RPPG_TAUG_P=0.25 RPPG_TAUG_HR_SPLIT=300 RPPG_SEED=1
run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN; grep -aE "LSUB stats|NF stats epoch" /tmp/ft_$M.log | cut -c1-120; touch /tmp/lsub_smoke_w$W.done
