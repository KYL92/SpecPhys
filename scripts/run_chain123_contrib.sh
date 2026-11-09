#!/bin/bash
# 체인 123 (64 문서, 사전 등록): 최종 모델에서 잡음 라벨만 뺀 NF_NSP2NL_L, 풀에서 합성 클립을 뺀 NF_NSP2NS_L, 시드 2100~2500. 기준은 같은 시드 NF_NSP2_L. 사용: nohup bash scripts/run_chain123_contrib.sh LANE pro6000 >> /tmp/chain123_LANE.log 2>&1 &
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
LANE=${1:?LANE}; Q=${2:?QUEUE}; exec 9>/tmp/chain123_$LANE.lock; flock -n 9 || { echo "chain123 $LANE already running (lock) -> exit"; exit 0; }
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; TB=$ROOT/repos/rPPG-Toolbox; TPLN=CoTAM_HYB4_L
CSV0=$LISTS/cotrain_allf_hyb4_u8_pool.csv; CSVNS=$LISTS/cotrain_allf_hyb4_nosyn_u8_pool.csv; MMPD_HO=$LISTS/mmpd_zs_7_33.csv; NFYAML=$PFA/yaml_NF_HYB_L/finetune_NF_HYB_L.yaml
ARCH_ENV="RPPG_NF=1 RPPG_SKINW=0.2 NF_SS_TRACE=1 NF_SS_REF=1 NF_ARCH=nsp NF_NSP_LAM=0.01"; AXES="PURE NIRP_STILL NIRP_SMALL NIRP_LARGE DLCN MCD_A MCD_B RTNI RTNC REALTEST"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) [$LANE] $1 -> abort"; touch /tmp/chain123_$LANE.fail; exit 1; }
prune() { find $PFA/runs/finetune_$1/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; }
ckpt() { echo "$PFA/runs/finetune_$1/PreTrainedModels/SYNTH_${1}_factorizephys_Epoch9.pth"; }
has_out() { ls $1/*/saved_test_outputs/*.pickle > /dev/null 2>&1; }
grep -q "patch 55" $TB/neural_methods/trainer/FactorizePhysTrainer.py && grep -q "def snmf_gate" $TB/neural_methods/model/FactorizePhys/FactorizePhysNF.py || fail "PATCH55_MISSING"; [ -f $NFYAML ] && [ -d $PFA/noise_gt_hyb2 ] || fail "NF_ASSETS_MISSING"
nf_train() { local M=$1 SEED=$2 ARM=$3 ck y CSV NCH; ck=$(ckpt $M); if [ $ARM = NS ]; then CSV=$CSVNS; NCH=3125; else CSV=$CSV0; NCH=3665; fi; [ -f $CSV ] || fail "${M}_CSV_MISSING"
  [ -f "$ck" ] && { echo "skip train $M"; return; }
  python scripts/vipl_yaml.py train $M 1 $CSV 10 --test_csv $MMPD_HO --template $NFYAML --lr 1e-4 > /dev/null; y=$PFA/yaml_VIPL/finetune_$M.yaml
  grep -q "NAME: FactorizePhys" $y && grep -q "FILE_LIST_PATH: $CSV" $y && grep -q "FSAM_Res.pth" $y || fail "${M}_YAML"
  echo "$(ts) TRAIN $M start (arm $ARM: $NF_ENV; pool $(basename $CSV); seed $SEED, lr 1e-4, in-run test MMPD 7-33)"
  ( unset HYB_NOISE_GT_DIR; export $NF_ENV RPPG_CACHE_U8=1 NF_W_DBL=0.5 RPPG_TAUG=1 RPPG_TAUG_P=0.25 RPPG_TAUG_HR_SPLIT=300; RPPG_SEED=$SEED run_gated $GATE $y /tmp/ft_$M.log ${M}_TRAIN )
  [ -f "$ck" ] || fail "${M}_TRAIN_FAIL"; prune $M
  grep -aq "^NF stats epoch 0" /tmp/ft_$M.log || fail "${M}_NF_INACTIVE"; grep -a "\[nf\] active" /tmp/ft_$M.log | grep -q "NF_W_DBL.: 0.5" || fail "${M}_DBL_INACTIVE"
  grep -aq "\[skinbg\] active" /tmp/ft_$M.log || fail "${M}_SKINBG_INACTIVE"; grep -aq "\[nf-ref\] active" /tmp/ft_$M.log && grep -a "NF-REF stats epoch 9" /tmp/ft_$M.log | grep -q "bg=[1-9]" || fail "${M}_NFREF_INACTIVE"
  grep -a "TAUG stats epoch 9" /tmp/ft_$M.log | grep -q "x2" || fail "${M}_TAUG_INACTIVE"
  grep -aq "\[nf-arch\] arch=nsp" /tmp/ft_$M.log || fail "${M}_ARCH_INACTIVE"
  grep -a "NF stats epoch 9" /tmp/ft_$M.log | grep -q "ngt_hit=0 " || fail "${M}_NOISE_LABEL_STILL_USED"; grep -a "NF stats epoch 9" /tmp/ft_$M.log | grep -q "gate=[0-9]*/$NCH " || fail "${M}_CHUNK_COUNT"
  grep -a "NF stats epoch 9" /tmp/ft_$M.log | cut -c1-160; cp /tmp/ft_$M.log $RES/chain123_ft_$M.log
  python scripts/agg_mmpd_cv2.py $M > /tmp/agg_${M}_MMPD.log 2>&1 || fail "${M}_AGG_MMPD"; echo "$(ts) $M MMPD: $(sed -n 2p /tmp/agg_${M}_MMPD.log | cut -c1-120)"; }
nf_eval_vipl() { local M=$1 ck=$(ckpt $1); [ -f $RES/agg_${M}_VIPL.json ] && return
  python scripts/vipl_yaml.py ev $M "$ck" --template $PFA/yaml_NF_HYB_L/ev_NF_HYB_L_DLCN.yaml > /dev/null; grep -q "MODEL_PATH: $ck" $PFA/yaml_VIPL/ev_${M}_VIPL.yaml || fail "${M}_VIPL_YAML"
  echo "$(ts) EVAL $M VIPL start"; ( export $NF_ENV RPPG_CACHE_U8=1; run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL )
  has_out $PFA/runs/ev_${M}_VIPL || fail "${M}_VIPL_FAIL"
  python scripts/agg_vipl.py $M > /tmp/agg_${M}_VIPL.log 2>&1 || fail "${M}_AGG_VIPL"; echo "$(ts) $M VIPL: $(grep -E 'all \[문헌' /tmp/agg_${M}_VIPL.log | head -1 | cut -c1-120)"; }
nf_eval_axes() { local M=$1 A EXP pre
  for A in $AXES; do has_out $PFA/runs/ev_${M}_$A && continue
    EXP=$(grep -m1 "EXP_DATA_NAME" $PFA/yaml_$TPLN/ev_${TPLN}_$A.yaml 2>/dev/null | awk '{print $2}'); [ -n "$EXP" ] || continue
    pre=1; [ -d $PFA/cache/fp_$EXP ] && [ -f $LISTS/fp_${EXP}_0.0_1.0.csv ] && pre=0
    python scripts/fp_yaml.py ev $M $TPLN $A --preprocess $pre > /dev/null || { echo "$(ts) skip $A (yaml)"; continue; }
    ( export $NF_ENV; run_gated 6000 $PFA/yaml_$M/ev_${M}_$A.yaml /tmp/ev_${M}_$A.log ${M}_$A ); has_out $PFA/runs/ev_${M}_$A || echo "$(ts) WARN ${M}_${A}: no output"; done
  python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1 || fail "${M}_AGG"; cp /tmp/agg_$M.log $RES/chain123_agg_$M.log; echo "$(ts) AGG $M: $(grep -E 'RTNI|NIRP_LARGE' /tmp/agg_$M.log | head -2 | tr '\n' ' ' | cut -c1-160)"; }
[ $Q = pro6000 ] || fail "unknown queue $Q"
SEEDS="2100 2200 2300 2400 2500"; JOBS=""; for S in $SEEDS; do for ARM in NL NS; do JOBS="$JOBS $ARM:$S"; done; done
echo "$(ts) [$LANE] start queue=$Q jobs=$(echo $JOBS | wc -w)"
for J in $JOBS; do ARM=${J%%:*}; S=${J#*:}; M=NF_NSP2${ARM}_L_s$S
  [ -f /tmp/chain123_job_$M.done ] && continue
  exec 8>/tmp/chain123_job_$M.lock; flock -n 8 || { exec 8>&-; continue; }
  [ -f /tmp/chain123_job_$M.done ] && { exec 8>&-; continue; }
  if [ $ARM = NL ]; then NF_ENV="$ARCH_ENV"; else NF_ENV="$ARCH_ENV HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"; fi
  nf_train $M $S $ARM; nf_eval_vipl $M; nf_eval_axes $M
  touch /tmp/chain123_job_$M.done; echo "$(ts) [$LANE] $M DONE"; exec 8>&-
done
touch /tmp/chain123_$LANE.done; echo "$(ts) [$LANE] LANE_DONE"
