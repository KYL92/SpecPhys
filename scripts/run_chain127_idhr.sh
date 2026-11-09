#!/bin/bash
# 체인 127 (68 문서, 사전 등록): 정체성–심박 분리 강화·대조. K 결합 대조 NF_NSP2K_L, R 클립 안 심박 변화 NF_NSP2R_L, W 같은 장면 짝 NF_NSP2W_L, 시드 2100~2500.
#   대조: 깨끗한 마스크 NF_NSP2C_L(체인 126)·합성 없음 NF_NSP2NS_L(체인 123), 같은 시드. 평가: VIPL·실제 축·어려운/쉬운 심박 추종 통제 시험.
# 사용: nohup bash scripts/run_chain127_idhr.sh LANE pro6000 >> /tmp/chain127_LANE.log 2>&1 &
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
LANE=${1:?LANE}; Q=${2:?QUEUE}; exec 9>/tmp/chain127_$LANE.lock; flock -n 9 || { echo "chain127 $LANE already running (lock) -> exit"; exit 0; }
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; TB=$ROOT/repos/rPPG-Toolbox; TPLN=CoTAM_HYB4_L
NCH=3665; HSL=$LISTS/hrsweeplo_u8.csv; HSH=$LISTS/hrsweep_u8.csv; MMPD_HO=$LISTS/mmpd_zs_7_33.csv; NFYAML=$PFA/yaml_NF_HYB_L/finetune_NF_HYB_L.yaml
B0="RPPG_NF=1 RPPG_SKINW=0.2 NF_SS_TRACE=1 NF_SS_REF=1 NF_ARCH=nsp NF_NSP_LAM=0.01"; AXES="PURE NIRP_STILL NIRP_SMALL NIRP_LARGE DLCN MCD_A MCD_B RTNI RTNC REALTEST"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) [$LANE] $1 -> abort"; touch /tmp/chain127_$LANE.fail; exit 1; }
prune() { find $PFA/runs/finetune_$1/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; }
ckpt() { ls $PFA/runs/finetune_$1/PreTrainedModels/*_Epoch9.pth 2>/dev/null | head -1; }
has_out() { ls $1/*/saved_test_outputs/*.pickle > /dev/null 2>&1; }
for a in r w k; do [ -f $LISTS/cotrain_allf_hyb4${a}_u8_pool.csv ] && [ -d $PFA/noise_gt_hyb2$a ] || fail "ASSETS_MISSING_$a"; done; [ -f $HSL ] && [ -f $HSH ] && [ -f $NFYAML ] || fail "ASSETS_MISSING"
nf_train() { local M=$1 SEED=$2 ck y; ck=$(ckpt $M); [ -n "$ck" ] && { echo "skip train $M"; return; }
  python scripts/vipl_yaml.py train $M 1 $CSV 10 --test_csv $MMPD_HO --template $NFYAML --lr 1e-4 > /dev/null; y=$PFA/yaml_VIPL/finetune_$M.yaml
  grep -q "NAME: FactorizePhys" $y && grep -q "FILE_LIST_PATH: $CSV" $y && grep -q "FSAM_Res.pth" $y || fail "${M}_YAML"
  echo "$(ts) TRAIN $M start ($NF_ENV; seed $SEED, lr 1e-4, in-run test MMPD 7-33)"
  ( export $NF_ENV RPPG_CACHE_U8=1 NF_W_DBL=0.5 RPPG_TAUG=1 RPPG_TAUG_P=0.25 RPPG_TAUG_HR_SPLIT=300; RPPG_SEED=$SEED run_gated $GATE $y /tmp/ft_$M.log ${M}_TRAIN )
  ck=$(ckpt $M); [ -n "$ck" ] || fail "${M}_TRAIN_FAIL"; prune $M
  grep -aq "^NF stats epoch 0" /tmp/ft_$M.log || fail "${M}_NF_INACTIVE"; grep -a "\[nf\] active" /tmp/ft_$M.log | grep -q "NF_W_DBL.: 0.5" || fail "${M}_DBL_INACTIVE"
  grep -aq "\[skinbg\] active" /tmp/ft_$M.log || fail "${M}_SKINBG_INACTIVE"; grep -a "NF-REF stats epoch 9" /tmp/ft_$M.log | grep -q "bg=[1-9]" || fail "${M}_NFREF_INACTIVE"
  grep -a "TAUG stats epoch 9" /tmp/ft_$M.log | grep -q "x2" || fail "${M}_TAUG_INACTIVE"; grep -aq "\[nf-arch\] arch=nsp" /tmp/ft_$M.log || fail "${M}_ARCH_INACTIVE"
  grep -a "NF stats epoch 9" /tmp/ft_$M.log | grep -q "ngt_hit=[1-9]" || fail "${M}_NOISE_LABEL_INACTIVE"; grep -a "NF stats epoch 9" /tmp/ft_$M.log | grep -q "gate=[0-9]*/$NCH " || fail "${M}_CHUNK_COUNT"
  grep -aq "\[lsub\]\|\[bgref\]" /tmp/ft_$M.log && fail "${M}_MODULE_LEAK"
  grep -a "NF stats epoch 9" /tmp/ft_$M.log | cut -c1-170; cp /tmp/ft_$M.log $RES/chain127_ft_$M.log
  python scripts/agg_mmpd_cv2.py $M > /tmp/agg_${M}_MMPD.log 2>&1 || fail "${M}_AGG_MMPD"; echo "$(ts) $M MMPD: $(sed -n 2p /tmp/agg_${M}_MMPD.log | cut -c1-120)"; }
nf_eval_vipl() { local M=$1 ck=$(ckpt $1); [ -f $RES/agg_${M}_VIPL.json ] && return
  python scripts/vipl_yaml.py ev $M "$ck" --template $PFA/yaml_NF_HYB_L/ev_NF_HYB_L_DLCN.yaml > /dev/null; grep -q "MODEL_PATH: $ck" $PFA/yaml_VIPL/ev_${M}_VIPL.yaml || fail "${M}_VIPL_YAML"
  echo "$(ts) EVAL $M VIPL start"; ( export $NF_ENV RPPG_CACHE_U8=1; run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL )
  has_out $PFA/runs/ev_${M}_VIPL || fail "${M}_VIPL_FAIL"
  python scripts/agg_vipl.py $M > /tmp/agg_${M}_VIPL.log 2>&1 || fail "${M}_AGG_VIPL"; echo "$(ts) $M VIPL: $(grep -E 'all \[문헌' /tmp/agg_${M}_VIPL.log | head -1 | cut -c1-120)"; }
nf_eval_set() { local M=$1 TAG=$2 CS=$3 ck=$(ckpt $1) y; has_out $PFA/runs/ev_${M}_$TAG && return
  python scripts/vipl_yaml.py ev ${M}_${TAG}TMP "$ck" --template $PFA/yaml_NF_HYB_L/ev_NF_HYB_L_DLCN.yaml > /dev/null; y=$PFA/yaml_VIPL/ev_${M}_$TAG.yaml
  sed -e "s#$LISTS/vipl_all.csv#$CS#" -e "s#ev_${M}_${TAG}TMP_VIPL#ev_${M}_$TAG#g" $PFA/yaml_VIPL/ev_${M}_${TAG}TMP_VIPL.yaml > $y; rm -f $PFA/yaml_VIPL/ev_${M}_${TAG}TMP_VIPL.yaml
  grep -q "FILE_LIST_PATH: $CS" $y && grep -q "MODEL_PATH: $ck" $y || fail "${M}_${TAG}_YAML"
  ( export $NF_ENV RPPG_CACHE_U8=1; run_gated 6000 $y /tmp/ev_${M}_$TAG.log ${M}_$TAG ); has_out $PFA/runs/ev_${M}_$TAG || fail "${M}_${TAG}_FAIL"; echo "$(ts) $M $TAG done"; }
nf_eval_axes() { local M=$1 A EXP pre
  for A in $AXES; do has_out $PFA/runs/ev_${M}_$A && continue
    EXP=$(grep -m1 "EXP_DATA_NAME" $PFA/yaml_$TPLN/ev_${TPLN}_$A.yaml 2>/dev/null | awk '{print $2}'); [ -n "$EXP" ] || continue
    pre=1; [ -d $PFA/cache/fp_$EXP ] && [ -f $LISTS/fp_${EXP}_0.0_1.0.csv ] && pre=0
    python scripts/fp_yaml.py ev $M $TPLN $A --preprocess $pre > /dev/null || { echo "$(ts) skip $A (yaml)"; continue; }
    ( export $NF_ENV; run_gated 6000 $PFA/yaml_$M/ev_${M}_$A.yaml /tmp/ev_${M}_$A.log ${M}_$A ); has_out $PFA/runs/ev_${M}_$A || echo "$(ts) WARN ${M}_${A}: no output"; done
  python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1 || fail "${M}_AGG"; cp /tmp/agg_$M.log $RES/chain127_agg_$M.log; echo "$(ts) AGG $M: $(grep -E 'RTNI|NIRP_LARGE' /tmp/agg_$M.log | head -2 | tr '\n' ' ' | cut -c1-160)"; }
[ $Q = pro6000 ] || fail "unknown queue $Q"
JOBS=""; for S in 2100 2200 2300 2400 2500; do for A in K R W; do JOBS="$JOBS $A:$S"; done; done
echo "$(ts) [$LANE] start queue=$Q jobs=$(echo $JOBS | wc -w)"
for J in $JOBS; do ARM=${J%%:*}; S=${J#*:}; a=$(echo $ARM | tr A-Z a-z); CSV=$LISTS/cotrain_allf_hyb4${a}_u8_pool.csv; NF_ENV="$B0 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2$a"; M=NF_NSP2${ARM}_L_s$S
  [ -f /tmp/chain127_job_$M.done ] && continue
  exec 8>/tmp/chain127_job_$M.lock; flock -n 8 || { exec 8>&-; continue; }
  [ -f /tmp/chain127_job_$M.done ] && { exec 8>&-; continue; }
  nf_train $M $S; nf_eval_set $M HRSWEEPLO $HSL; nf_eval_set $M HRSWEEP $HSH; nf_eval_vipl $M; nf_eval_axes $M
  ( HRSWEEP_SET=LO python scripts/hrsweep_metrics.py $M > /dev/null 2>&1; python scripts/hrsweep_metrics.py $M > /dev/null 2>&1 ); [ -f $RES/hrsweeplo_$M.json ] && [ -f $RES/hrsweep_$M.json ] || fail "${M}_HRSWEEP_METRICS"
  touch /tmp/chain127_job_$M.done; echo "$(ts) [$LANE] $M DONE"; exec 8>&-
done
touch /tmp/chain127_$LANE.done; echo "$(ts) [$LANE] LANE_DONE"
