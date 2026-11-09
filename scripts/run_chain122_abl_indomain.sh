#!/bin/bash
# 체인 122 (62 문서 §14, 사전 등록): ① 2×2 아블레이션 NF_NSP2O_L(원본 어텐션 + 투영) 시드 2100~2500 ② 최종 설정의 MMPD in-domain MF{k}_SKN_L_s{600,700,800}(NF_NSP2_L_s{S} 에서 2단계 적응, 체인 115 와 같은 방법). 사용: nohup bash scripts/run_chain122_abl_indomain.sh LANE QUEUE(alpha|pro6000) >> /tmp/chain122_LANE.log 2>&1 &
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
LANE=${1:?LANE}; Q=${2:?QUEUE}; exec 9>/tmp/chain122_$LANE.lock; flock -n 9 || { echo "chain122 $LANE already running (lock) -> exit"; exit 0; }
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; TB=$ROOT/repos/rPPG-Toolbox; TPLN=CoTAM_HYB4_L
CSV=$LISTS/cotrain_allf_hyb4_u8_pool.csv; MMPD_HO=$LISTS/mmpd_zs_7_33.csv; NFYAML=$PFA/yaml_NF_HYB_L/finetune_NF_HYB_L.yaml
BASE_ENV="RPPG_NF=1 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2 RPPG_SKINW=0.2 NF_SS_TRACE=1 NF_SS_REF=1"; AXES="PURE NIRP_STILL NIRP_SMALL NIRP_LARGE DLCN MCD_A MCD_B RTNI RTNC REALTEST"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) [$LANE] $1 -> abort"; touch /tmp/chain122_$LANE.fail; exit 1; }
prune() { find $PFA/runs/finetune_$1/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; }
ckpt() { echo "$PFA/runs/finetune_$1/PreTrainedModels/SYNTH_${1}_factorizephys_Epoch9.pth"; }
has_out() { ls $1/*/saved_test_outputs/*.pickle > /dev/null 2>&1; }
arm_env() { case $1 in NSP2O) echo "NF_ORIGLAYOUT=1 NF_ARCH=nsp NF_NSP_LAM=0.01";; *) fail "unknown arm $1";; esac; }
grep -q "patch 55" $TB/neural_methods/trainer/FactorizePhysTrainer.py && grep -q "def snmf_gate" $TB/neural_methods/model/FactorizePhys/FactorizePhysNF.py || fail "PATCH55_MISSING"; [ -f $NFYAML ] && [ -d $PFA/noise_gt_hyb2 ] || fail "NF_ASSETS_MISSING"
nf_train() { local M=$1 SEED=$2 ARM=$3 ck y; ck=$(ckpt $M)
  [ -f "$ck" ] && { echo "skip train $M"; return; }
  python scripts/vipl_yaml.py train $M 1 $CSV 10 --test_csv $MMPD_HO --template $NFYAML --lr 1e-4 > /dev/null; y=$PFA/yaml_VIPL/finetune_$M.yaml
  grep -q "NAME: FactorizePhys" $y && grep -q "FILE_LIST_PATH: $CSV" $y && grep -q "FSAM_Res.pth" $y || fail "${M}_YAML"
  echo "$(ts) TRAIN $M start (arm $ARM: $NF_ENV; recipe SpecPhys-L; seed $SEED, lr 1e-4, in-run test MMPD 7-33)"
  ( export $NF_ENV RPPG_CACHE_U8=1 NF_W_DBL=0.5 RPPG_TAUG=1 RPPG_TAUG_P=0.25 RPPG_TAUG_HR_SPLIT=300; [ $ARM = X15 ] && export RPPG_TAUG_X15_RE='^subject7[0-9]{5}$' RPPG_TAUG_X15_P=0.25; RPPG_SEED=$SEED run_gated $GATE $y /tmp/ft_$M.log ${M}_TRAIN )
  [ -f "$ck" ] || fail "${M}_TRAIN_FAIL"; prune $M
  grep -aq "^NF stats epoch 0" /tmp/ft_$M.log || fail "${M}_NF_INACTIVE"; grep -a "\[nf\] active" /tmp/ft_$M.log | grep -q "NF_W_DBL.: 0.5" || fail "${M}_DBL_INACTIVE"
  grep -aq "\[skinbg\] active" /tmp/ft_$M.log || fail "${M}_SKINBG_INACTIVE"; grep -aq "\[nf-ref\] active" /tmp/ft_$M.log && grep -a "NF-REF stats epoch 9" /tmp/ft_$M.log | grep -q "bg=[1-9]" || fail "${M}_NFREF_INACTIVE"
  grep -a "TAUG stats epoch 9" /tmp/ft_$M.log | grep -q "x2" || fail "${M}_TAUG_INACTIVE"
    case $ARM in
        *) grep -aq "\[nf-arch\] arch=" /tmp/ft_$M.log || fail "${M}_ARCH_INACTIVE"; grep -a "\[nf-arch\]" /tmp/ft_$M.log | head -1;;
  esac
  case $ARM in NSG|NSG0|SGF|SNMF) grep -aq "NSG stats epoch 9" /tmp/ft_$M.log || fail "${M}_NSGSTATS_MISSING"; grep -a "NSG stats epoch \(0\|9\)" /tmp/ft_$M.log;; esac
  [ $ARM = NSG ] && { grep -a "NSG stats epoch 9" /tmp/ft_$M.log | grep -q "w=0.5" && grep -a "NSG stats epoch 0" /tmp/ft_$M.log | grep -q "ngt=[1-9]" || fail "${M}_NSG_SUP_INACTIVE"; }
  grep -a "NF stats epoch 9" /tmp/ft_$M.log | cut -c1-160; cp /tmp/ft_$M.log $RES/chain122_ft_$M.log
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
  python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1 || fail "${M}_AGG"; cp /tmp/agg_$M.log $RES/chain122_agg_$M.log; echo "$(ts) AGG $M: $(grep -E 'RTNI|NIRP_LARGE' /tmp/agg_$M.log | head -2 | tr '\n' ' ' | cut -c1-160)"; }
fold_train() { local M=$1 K=$2 RCK=$3 SEED=$4 ck y; ck=$(ckpt $M)
  [ -f "$ck" ] && { echo "$(ts) skip train $M"; return 0; }
  [ -f "$RCK" ] || fail "${M}_RESUME_MISSING $RCK"
  python scripts/vipl_yaml.py train $M $K $LISTS/mmpd_train$K.csv 10 --test_csv $LISTS/mmpd_test$K.csv --template $NFYAML --resume "$RCK" --lr 5e-5 > /dev/null; y=$PFA/yaml_VIPL/finetune_$M.yaml
  grep -q "RESUME: $RCK" $y && grep -q "NAME: FactorizePhys" $y && grep -q "FILE_LIST_PATH: $LISTS/mmpd_train$K.csv" $y || fail "${M}_YAML"
  echo "$(ts) TRAIN $M start (MMPD fold $K in-domain, 2-stage from $(basename $RCK), seed $SEED, lr 5e-5, projection on)"
  ( export $BASE_ENV NF_ARCH=nsp NF_NSP_LAM=0.01 NF_W_DBL=0.5 RPPG_TAUG=1 RPPG_TAUG_P=0.25 RPPG_TAUG_HR_SPLIT=300 NF_HR_TB=1 RPPG_CACHE_U8=1; RPPG_SEED=$SEED run_gated $GATE $y /tmp/ft_$M.log ${M}_TRAIN )
  [ -f "$ck" ] || fail "${M}_TRAIN_FAIL"; prune $M
  grep -aq "^NF stats epoch 0" /tmp/ft_$M.log || fail "${M}_NF_INACTIVE"; grep -a "\[nf\] active" /tmp/ft_$M.log | grep -q "hr_tb=1" || fail "${M}_HRTB_INACTIVE"; grep -aq "\[nf-arch\] arch=nsp" /tmp/ft_$M.log || fail "${M}_ARCH_INACTIVE"
  grep -aq "\[skinbg\] active" /tmp/ft_$M.log || fail "${M}_SKINBG_INACTIVE"; grep -a "NF-REF stats epoch 9" /tmp/ft_$M.log | grep -q "bg=[1-9]" || fail "${M}_NFREF_INACTIVE"; grep -a "TAUG stats epoch 9" /tmp/ft_$M.log | grep -q "x2" || fail "${M}_TAUG_INACTIVE"
  cp /tmp/ft_$M.log $RES/chain122_ft_$M.log
  python scripts/agg_mmpd_cv2.py $M > /tmp/agg_${M}_MMPD.log 2>&1 || fail "${M}_AGG"; echo "$(ts) $M MMPD(fold $K): $(sed -n 2p /tmp/agg_${M}_MMPD.log | cut -c1-110)"; }
case $Q in
  alpha) JOBS="NSP2O:2100 NSP2O:2200 NSP2O:2300 NSP2O:2400 NSP2O:2500";;
  pro6000) JOBS=""; for S in 600 700 800; do for K in 1 2 3 4 5; do JOBS="$JOBS SKN:$K:$S"; done; done;;
  *) fail "unknown queue $Q";;
esac
echo "$(ts) [$LANE] start queue=$Q jobs=$(echo $JOBS | wc -w)"
for J in $JOBS; do A=${J%%:*}; R=${J#*:}
  if [ $A = SKN ]; then K=${R%%:*}; S=${R#*:}; M=MF${K}_SKN_L_s$S; else S=$R; M=NF_${A}_L_s$S; fi
  [ -f /tmp/chain122_job_$M.done ] && continue
  exec 8>/tmp/chain122_job_$M.lock; flock -n 8 || { exec 8>&-; continue; }
  [ -f /tmp/chain122_job_$M.done ] && { exec 8>&-; continue; }
  if [ $A = SKN ]; then fold_train $M $K "$(ckpt NF_NSP2_L_s$S)" $S; else ARM=$A; NF_ENV="$BASE_ENV $(arm_env $ARM)"; nf_train $M $S $ARM; nf_eval_vipl $M; nf_eval_axes $M; fi
  touch /tmp/chain122_job_$M.done; echo "$(ts) [$LANE] $M DONE"; exec 8>&-
done
touch /tmp/chain122_$LANE.done; echo "$(ts) [$LANE] LANE_DONE"
