#!/bin/bash
# 체인 49 (31 문서 D 단계): 통합 트랙 시드·fold 확장 + 조건 셀. 체인 46·46b·47 완료 마커 대기 후 단독 GPU 사용.
#  ①LITS: 문헌 레시피 Standardized 입력 변형 1회(§2 규칙, LIT 18.66 붕괴로 발동)  ②TWg·HYBg 시드 200/300(fold 1) + NIRP/DLCN 전이
#  ③TWg fold 2~5(시드 100) → agg --cv TWg_L (문헌 GT 5-fold 표)  ④C3 승자 시드는 C3 판정 후 별도 체인(50)
# 로그 /tmp/chain49.log, 마커 /tmp/chain49_{lits,seeds,cv}.done, /tmp/chain49.done
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_CACHE_U8=1 RPPG_PEARSON_SKIP_RE='^p[0-9]+_v[0-9]' RPPG_HR_LABEL_FILE=1
Y=$PFA/yaml_VIPL; LISTS=$PFA/cache/DataFileLists; GATE=${GATE:-12000}   # 10-17: 체인 52 와 병행(이전 21000 은 단독 전제)
HYB_ENV="RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_DN_ONLY=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"
train_cell() { # train_cell NAME FOLD SEED CSV EPOCHS HYB(0/1) [vipl_yaml extra args]
  local M=$1 K=$2 SEED=$3 CSV=$4 EP=$5 HYB=$6; shift 6
  [ -f $PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch$((EP-1)).pth ] && { echo "skip $M (done)"; return; }
  python scripts/vipl_yaml.py train $M $K $CSV $EP "$@" > /dev/null
  echo "$(date '+%m-%d %H:%M') TRAIN $M start (seed $SEED)"
  if [ $HYB = 1 ]; then ( export $HYB_ENV; RPPG_SEED=$SEED run_gated $GATE $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
  else RPPG_SEED=$SEED run_gated $GATE $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN; fi
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch$((EP-1)).pth" -delete 2>/dev/null
  local HIT=$(grep -ao "HRFILE sidecar hits=[0-9]*" /tmp/ft_$M.log | tail -1 | grep -oE "[0-9]+$")
  echo "  $M sidecar hits=${HIT:-none}"
  [ "${HIT:-0}" -gt 0 ] || { echo "$(date '+%m-%d %H:%M') ${M}_HRFILE_FAIL -> abort"; touch /tmp/chain49.fail; exit 1; }
  python scripts/agg_vipl.py $M 2>&1 | head -3
}
nirp_eval() { # nirp_eval NAME HYB(0/1) [EPOCH=9]
  local M=$1 HYB=$2 EP=${3:-9}
  gen_yaml $M ALLF_TW cotrain_allf_tw x > /dev/null
  [ "$EP" != "9" ] && sed -i "s#_physformer_Epoch9.pth#_physformer_Epoch$EP.pth#" $PFA/yaml_$M/ev_${M}_*.yaml
  for ax in NIRP_STILL NIRP_SMALL NIRP_LARGE DLCN; do
    [ -d $PFA/runs/ev_${M}_$ax ] && { echo "skip ev $M $ax"; continue; }
    if [ $HYB = 1 ]; then ( export RPPG_NOISE_HEAD=1; run_gated 6000 $PFA/yaml_$M/ev_${M}_$ax.yaml /tmp/ev_${M}_$ax.log ${M}_$ax )
    else run_gated 6000 $PFA/yaml_$M/ev_${M}_$ax.yaml /tmp/ev_${M}_$ax.log ${M}_$ax; fi
  done
  python scripts/agg_model2.py $M 2>&1 | grep -E "NIRP|DLCN" | head -6; python scripts/motion_split_cmp.py $M 2>&1 | grep NIRP
}
{
echo "$(date '+%m-%d %H:%M') CHAIN49 waiting for chain46/46b/47 markers"
while [ ! -f /tmp/chain46.done ] || [ ! -f /tmp/chain46b.done ] || [ ! -f /tmp/chain47.done ]; do sleep 600; done
echo "$(date '+%m-%d %H:%M') CHAIN49 START"
cd $ROOT
# ① LITS (문헌 레시피 + Standardized 입력): VIPL 단독, scratch 25 ep, 파형 손실 포함(a=0.1)
if [ ! -f /tmp/chain49_lits.done ]; then
  ( unset RPPG_PEARSON_SKIP_RE; export PF_LOSS_A=0.1; train_cell VF1_LITS_L 1 100 $LISTS/vipl_train1.csv 25 0 --lr 1e-4 --scratch --std )
  python scripts/noise_judge.py vipl VF1_LITS_L VF1_TWg_L VF1_LIT_L
  touch /tmp/chain49_lits.done
fi
# ② 시드 200/300: TWg, HYBg (fold 1) + 전이 평가
if [ ! -f /tmp/chain49_seeds.done ]; then
  for S in 200 300; do
    train_cell VF1_TWg_L_s$S 1 $S $LISTS/vipl_train1_cotrain_allf_tw_u8.csv 10 0; nirp_eval VF1_TWg_L_s$S 0
    train_cell VF1_HYBg_L_s$S 1 $S $LISTS/vipl_train1_cotrain_allf_hyb4_u8.csv 10 1; nirp_eval VF1_HYBg_L_s$S 1
  done
  touch /tmp/chain49_seeds.done
fi
# ③ TWg fold 2~5 (시드 100) → 5-fold 문헌 GT 표
if [ ! -f /tmp/chain49_cv.done ]; then
  for k in 2 3 4 5; do
    [ -f $LISTS/vipl_train${k}_cotrain_allf_tw_u8.csv ] || python scripts/vipl_filelist.py train $k cotrain_allf_tw_u8
    train_cell VF${k}_TWg_L $k 100 $LISTS/vipl_train${k}_cotrain_allf_tw_u8.csv 10 0
  done
  python scripts/agg_vipl.py --cv TWg_L TW_L HYB_L 2>&1 | grep -E "^---|all " ; touch /tmp/chain49_cv.done
fi
echo "$(date '+%m-%d %H:%M') CHAIN49 DONE"; touch /tmp/chain49.done
} >> /tmp/chain49.log 2>&1
