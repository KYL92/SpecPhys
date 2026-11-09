#!/bin/bash
# 체인 124 (65 문서, 사전 등록): 학습된 잡음 빼기 + 이중 렌더 픽셀·색 정답(잡음 영상). 시드 2600~3000, 풀 = 기존 366클립 + 새 이중 렌더 48클립(3,953 청크).
#   F  NF_NSP2DR_L : 최종 레시피(새 풀)   LS NF_LSUB_L : + 학습된 잡음 빼기 + 잡음 영상 특징 분리 감독   L0 NF_LSUB0_L : + 학습된 잡음 빼기, 감독 없음
#   R  NF_NSP2_L_s2100~2500 : 현재 최종 모델의 분리 시험 평가만(서술)
# 사용: nohup bash scripts/run_chain124_lsub.sh LANE pro6000 >> /tmp/chain124_LANE.log 2>&1 &
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
LANE=${1:?LANE}; Q=${2:?QUEUE}; exec 9>/tmp/chain124_$LANE.lock; flock -n 9 || { echo "chain124 $LANE already running (lock) -> exit"; exit 0; }
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; TB=$ROOT/repos/rPPG-Toolbox; TPLN=CoTAM_HYB4_L
CSV=$LISTS/cotrain_allf_hyb4_dr_u8_pool.csv; NCH=3953; SEPCSV=$LISTS/sepeval_u8.csv; MMPD_HO=$LISTS/mmpd_zs_7_33.csv; NFYAML=$PFA/yaml_NF_HYB_L/finetune_NF_HYB_L.yaml
BASE_ENV="RPPG_NF=1 RPPG_SKINW=0.2 NF_SS_TRACE=1 NF_SS_REF=1 NF_ARCH=nsp NF_NSP_LAM=0.01 HYB_NOISE_GT_DIR=$PFA/noise_gt_dr"; AXES="PURE NIRP_STILL NIRP_SMALL NIRP_LARGE DLCN MCD_A MCD_B RTNI RTNC REALTEST"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) [$LANE] $1 -> abort"; touch /tmp/chain124_$LANE.fail; exit 1; }
prune() { find $PFA/runs/finetune_$1/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; }
ckpt() { ls $PFA/runs/finetune_$1/PreTrainedModels/*_Epoch9.pth 2>/dev/null | head -1; }
has_out() { ls $1/*/saved_test_outputs/*.pickle > /dev/null 2>&1; }
grep -q "patch 56" $TB/neural_methods/trainer/FactorizePhysTrainer.py && grep -q "class LearnedNoiseSub" $TB/neural_methods/model/FactorizePhys/FactorizePhysNF.py || fail "PATCH56_MISSING"
[ -f $CSV ] && [ -f $SEPCSV ] && [ -d $PFA/noise_gt_dr ] && [ -f $NFYAML ] || fail "ASSETS_MISSING"
nf_train() { local M=$1 SEED=$2 ARM=$3 ck y; ck=$(ckpt $M); [ -n "$ck" ] && { echo "skip train $M"; return; }
  python scripts/vipl_yaml.py train $M 1 $CSV 10 --test_csv $MMPD_HO --template $NFYAML --lr 1e-4 > /dev/null; y=$PFA/yaml_VIPL/finetune_$M.yaml
  grep -q "NAME: FactorizePhys" $y && grep -q "FILE_LIST_PATH: $CSV" $y && grep -q "FSAM_Res.pth" $y || fail "${M}_YAML"
  echo "$(ts) TRAIN $M start (arm $ARM: $NF_ENV; seed $SEED, lr 1e-4, in-run test MMPD 7-33)"
  ( export $NF_ENV RPPG_CACHE_U8=1 NF_W_DBL=0.5 RPPG_TAUG=1 RPPG_TAUG_P=0.25 RPPG_TAUG_HR_SPLIT=300; RPPG_SEED=$SEED run_gated $GATE $y /tmp/ft_$M.log ${M}_TRAIN )
  ck=$(ckpt $M); [ -n "$ck" ] || fail "${M}_TRAIN_FAIL"; prune $M
  grep -aq "^NF stats epoch 0" /tmp/ft_$M.log || fail "${M}_NF_INACTIVE"; grep -a "\[nf\] active" /tmp/ft_$M.log | grep -q "NF_W_DBL.: 0.5" || fail "${M}_DBL_INACTIVE"
  grep -aq "\[skinbg\] active" /tmp/ft_$M.log || fail "${M}_SKINBG_INACTIVE"; grep -a "NF-REF stats epoch 9" /tmp/ft_$M.log | grep -q "bg=[1-9]" || fail "${M}_NFREF_INACTIVE"
  grep -a "TAUG stats epoch 9" /tmp/ft_$M.log | grep -q "x2" || fail "${M}_TAUG_INACTIVE"; grep -aq "\[nf-arch\] arch=nsp" /tmp/ft_$M.log || fail "${M}_ARCH_INACTIVE"
  grep -a "NF stats epoch 9" /tmp/ft_$M.log | grep -q "ngt_hit=[1-9]" || fail "${M}_NOISE_LABEL_INACTIVE"; grep -a "NF stats epoch 9" /tmp/ft_$M.log | grep -q "gate=[0-9]*/$NCH " || fail "${M}_CHUNK_COUNT"
  if [ $ARM != F ]; then grep -aq "\[lsub\] module active" /tmp/ft_$M.log && grep -aq "^LSUB stats epoch 9" /tmp/ft_$M.log || fail "${M}_LSUB_INACTIVE"; fi
  if [ $ARM = LS ]; then grep -a "^LSUB stats epoch 9" /tmp/ft_$M.log | grep -q "sup=1 w=8.0 .* hit=[1-9]" || fail "${M}_LSUB_SUP_INACTIVE"; fi
  if [ $ARM = L0 ]; then grep -a "^LSUB stats epoch 9" /tmp/ft_$M.log | grep -q "sup=0 " || fail "${M}_LSUB_SUP_LEAK"; fi
  if [ $ARM = F ]; then grep -aq "\[lsub\]" /tmp/ft_$M.log && fail "${M}_LSUB_LEAK"; fi
  grep -aE "NF stats epoch 9|LSUB stats epoch (0|9)" /tmp/ft_$M.log | cut -c1-170; cp /tmp/ft_$M.log $RES/chain124_ft_$M.log
  python scripts/agg_mmpd_cv2.py $M > /tmp/agg_${M}_MMPD.log 2>&1 || fail "${M}_AGG_MMPD"; echo "$(ts) $M MMPD: $(sed -n 2p /tmp/agg_${M}_MMPD.log | cut -c1-120)"; }
ev_check() { [ "$ARM" = F ] || [ "$ARM" = R ] || grep -aq "\[lsub\] module active" $1 || fail "$2_EVAL_WITHOUT_LSUB"; }
nf_eval_vipl() { local M=$1 ck=$(ckpt $1); [ -f $RES/agg_${M}_VIPL.json ] && return
  python scripts/vipl_yaml.py ev $M "$ck" --template $PFA/yaml_NF_HYB_L/ev_NF_HYB_L_DLCN.yaml > /dev/null; grep -q "MODEL_PATH: $ck" $PFA/yaml_VIPL/ev_${M}_VIPL.yaml || fail "${M}_VIPL_YAML"
  echo "$(ts) EVAL $M VIPL start"; ( export $NF_ENV RPPG_CACHE_U8=1; run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL )
  has_out $PFA/runs/ev_${M}_VIPL || fail "${M}_VIPL_FAIL"; ev_check /tmp/ev_${M}_VIPL.log ${M}_VIPL
  python scripts/agg_vipl.py $M > /tmp/agg_${M}_VIPL.log 2>&1 || fail "${M}_AGG_VIPL"; echo "$(ts) $M VIPL: $(grep -E 'all \[문헌' /tmp/agg_${M}_VIPL.log | head -1 | cut -c1-120)"; }
nf_eval_sep() { local M=$1 ck=$(ckpt $1) y; has_out $PFA/runs/ev_${M}_SEPEVAL && [ -f $RES/sepx_$M.json ] && return
  python scripts/vipl_yaml.py ev ${M}_SEPTMP "$ck" --template $PFA/yaml_NF_HYB_L/ev_NF_HYB_L_DLCN.yaml > /dev/null; y=$PFA/yaml_VIPL/ev_${M}_SEPEVAL.yaml
  sed -e "s#$LISTS/vipl_all.csv#$SEPCSV#" -e "s#ev_${M}_SEPTMP_VIPL#ev_${M}_SEPEVAL#g" $PFA/yaml_VIPL/ev_${M}_SEPTMP_VIPL.yaml > $y; rm -f $PFA/yaml_VIPL/ev_${M}_SEPTMP_VIPL.yaml
  grep -q "FILE_LIST_PATH: $SEPCSV" $y && grep -q "MODEL_PATH: $ck" $y && grep -q "runs/ev_${M}_SEPEVAL" $y || fail "${M}_SEP_YAML"
  ( export $NF_ENV RPPG_CACHE_U8=1; run_gated 6000 $y /tmp/ev_${M}_SEPEVAL.log ${M}_SEPEVAL )
  has_out $PFA/runs/ev_${M}_SEPEVAL || fail "${M}_SEP_FAIL"; ev_check /tmp/ev_${M}_SEPEVAL.log ${M}_SEPEVAL
  python scripts/sepx_metrics.py $M > /dev/null 2>&1 && [ -f $RES/sepx_$M.json ] || fail "${M}_SEPX_METRICS"; echo "$(ts) $M SEPEVAL done"; }
nf_eval_axes() { local M=$1 A EXP pre
  for A in $AXES; do has_out $PFA/runs/ev_${M}_$A && continue
    EXP=$(grep -m1 "EXP_DATA_NAME" $PFA/yaml_$TPLN/ev_${TPLN}_$A.yaml 2>/dev/null | awk '{print $2}'); [ -n "$EXP" ] || continue
    pre=1; [ -d $PFA/cache/fp_$EXP ] && [ -f $LISTS/fp_${EXP}_0.0_1.0.csv ] && pre=0
    python scripts/fp_yaml.py ev $M $TPLN $A --preprocess $pre > /dev/null || { echo "$(ts) skip $A (yaml)"; continue; }
    ( export $NF_ENV; run_gated 6000 $PFA/yaml_$M/ev_${M}_$A.yaml /tmp/ev_${M}_$A.log ${M}_$A ); has_out $PFA/runs/ev_${M}_$A || echo "$(ts) WARN ${M}_${A}: no output"
    [ -f /tmp/ev_${M}_$A.log ] && ev_check /tmp/ev_${M}_$A.log ${M}_$A; done
  python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1 || fail "${M}_AGG"; cp /tmp/agg_$M.log $RES/chain124_agg_$M.log; echo "$(ts) AGG $M: $(grep -E 'RTNI|NIRP_LARGE' /tmp/agg_$M.log | head -2 | tr '\n' ' ' | cut -c1-160)"; }
[ $Q = pro6000 ] || fail "unknown queue $Q"
JOBS=""; for S in 2600 2700 2800 2900 3000; do for ARM in LS F L0; do JOBS="$JOBS $ARM:$S"; done; done; for S in 2100 2200 2300 2400 2500; do JOBS="$JOBS R:$S"; done
echo "$(ts) [$LANE] start queue=$Q jobs=$(echo $JOBS | wc -w)"
for J in $JOBS; do ARM=${J%%:*}; S=${J#*:}
  case $ARM in F) M=NF_NSP2DR_L_s$S; NF_ENV="$BASE_ENV";; LS) M=NF_LSUB_L_s$S; NF_ENV="$BASE_ENV NF_LSUB=1 NF_LSUB_SUP=1 NF_LSUB_W=8";; L0) M=NF_LSUB0_L_s$S; NF_ENV="$BASE_ENV NF_LSUB=1";; R) M=NF_NSP2_L_s$S; NF_ENV="RPPG_NF=1 RPPG_SKINW=0.2 NF_SS_TRACE=1 NF_SS_REF=1 NF_ARCH=nsp NF_NSP_LAM=0.01";; esac
  [ -f /tmp/chain124_job_$M.done ] && continue
  exec 8>/tmp/chain124_job_$M.lock; flock -n 8 || { exec 8>&-; continue; }
  [ -f /tmp/chain124_job_$M.done ] && { exec 8>&-; continue; }
  if [ $ARM = R ]; then [ -n "$(ckpt $M)" ] || fail "${M}_REF_CKPT_MISSING"; nf_eval_sep $M
  else nf_train $M $S $ARM; nf_eval_sep $M; nf_eval_vipl $M; nf_eval_axes $M; fi
  touch /tmp/chain124_job_$M.done; echo "$(ts) [$LANE] $M DONE"; exec 8>&-
done
touch /tmp/chain124_$LANE.done; echo "$(ts) [$LANE] LANE_DONE"
