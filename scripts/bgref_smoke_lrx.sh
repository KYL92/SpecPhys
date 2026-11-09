#!/bin/bash
# [66 문서] 패치 57 2에폭 공학 점검(게이트 학습률 배수 $1)(판정에 쓰지 않음, 시험 출력은 보지 않고 지운다): 입력 단계 배경 기준 보정, 시드 1
source /SPECPHYS_ROOT/scripts/chain_lib.sh; cd $ROOT; L=$PFA/cache/DataFileLists; M=NF_BGREF_SMOKE_X$1
python scripts/vipl_yaml.py train $M 1 $L/cotrain_allf_hyb4_dr_u8_pool.csv 2 --test_csv $L/sepeval_u8.csv --template $PFA/yaml_NF_HYB_L/finetune_NF_HYB_L.yaml --lr 1e-4 > /dev/null
export RPPG_NF=1 RPPG_SKINW=0.2 NF_SS_TRACE=1 NF_SS_REF=1 NF_ARCH=nsp NF_NSP_LAM=0.01 HYB_NOISE_GT_DIR=$PFA/noise_gt_dr NF_BGREF=1 RPPG_PARSE_CH=1 NF_BGREF_LRX=$1 RPPG_CACHE_U8=1 NF_W_DBL=0.5 RPPG_TAUG=1 RPPG_TAUG_P=0.25 RPPG_TAUG_HR_SPLIT=300 RPPG_SEED=1
run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN
grep -aE "\[bgref\]|\[parse-ch\]|BGREF stats|NF stats epoch 0|Mean train loss|nan|Error|Traceback" /tmp/ft_$M.log | cut -c1-220 | head -12
rm -rf $PFA/runs/finetune_$M $PFA/yaml_VIPL/finetune_$M.yaml; touch /tmp/bgref_smoke_x$1.done
