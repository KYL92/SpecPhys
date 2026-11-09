#!/bin/bash
# [69 문서] 체인 128 준비(pro6000 컨테이너): 실제 사람 1/4 풀 3개, 통제 시험 캐시의 u8 통계(RhythmFormer 표준화 입력용), 이미 평가된 셀의 대역 통과 민감도 판독.
set -e; ROOT=/SPECPHYS_ROOT; PFA=$ROOT/_workspace/data/physformer; cd $ROOT
python scripts/make_scarce_pools.py --build
python scripts/u8_stats_any.py $PFA/cache/hrsweep_u8 1; python scripts/u8_stats_any.py $PFA/cache/hrsweeplo_u8 1
M=""; for c in NF_NSP2_L NF_NSP2C_L NF_NSP2NS_L; do for s in 2100 2200 2300 2400 2500; do M="$M ${c}_s$s"; done; done
HRSWEEP_SET=LO HRSWEEP_BP=1 python scripts/hrsweep_metrics.py $M | tail -2; HRSWEEP_BP=1 python scripts/hrsweep_metrics.py $M | tail -2
for s in 1600 1700 1800 1900 2000; do ls $PFA/runs/finetune_NF_NSP2_L_s$s/PreTrainedModels/*Epoch9.pth > /dev/null || { echo "CKPT_MISSING $s"; exit 1; }; done
echo "$(date '+%m-%d %H:%M') PREP128_DONE"
