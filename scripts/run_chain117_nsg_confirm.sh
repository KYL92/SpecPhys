#!/bin/bash
# 체인 117 (62 문서 §5 2단계, 사전 등록): 선별 팔과 기준 레시피 NF_SKREF_REP_L 을 새 시드 1100~1500 으로 같은 시드 학습. 사용: ARMS="NSG" nohup bash scripts/run_chain117_nsg_confirm.sh LANE QUEUE(alpha|pro6000) >> /tmp/chain117_LANE.log 2>&1 &
#  ORIGA 는 '원본 어텐션 대비' 서술용으로 함께 돈다.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
LANE=${1:?LANE}; Q=${2:?QUEUE}; exec 9>/tmp/chain117_$LANE.lock; flock -n 9 || { echo "chain117 $LANE already running (lock) -> exit"; exit 0; }
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; TB=$ROOT/repos/rPPG-Toolbox; TPLN=CoTAM_HYB4_L
CSV=$LISTS/cotrain_allf_hyb4_u8_pool.csv; MMPD_HO=$LISTS/mmpd_zs_7_33.csv; NFYAML=$PFA/yaml_NF_HYB_L/finetune_NF_HYB_L.yaml
BASE_ENV="RPPG_NF=1 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2 RPPG_SKINW=0.2 NF_SS_TRACE=1 NF_SS_REF=1"; AXES="PURE NIRP_STILL NIRP_SMALL NIRP_LARGE DLCN MCD_A MCD_B RTNI RTNC REALTEST"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) [$LANE] $1 -> abort"; touch /tmp/chain117_$LANE.fail; exit 1; }
prune() { find $PFA/runs/finetune_$1/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; }
ckpt() { echo "$PFA/runs/finetune_$1/PreTrainedModels/SYNTH_${1}_factorizephys_Epoch9.pth"; }
has_out() { ls $1/*/saved_test_outputs/*.pickle > /dev/null 2>&1; }
arm_env() { case $1 in BASE) echo "";; NSG) echo "NF_ARCH=nsg NF_NSG_W=0.5";; NSG0) echo "NF_ARCH=nsg NF_NSG_W=0";; SGF) echo "NF_ARCH=nsg NF_NSG_REFS=0 NF_NSG_W=0";; NSP) echo "NF_ARCH=nsp";; SNMF) echo "NF_ARCH=snmf";; ORIGA) echo "NF_ORIGLAYOUT=1";; *) fail "unknown arm $1";; esac; }
grep -q "patch 55" $TB/neural_methods/trainer/FactorizePhysTrainer.py && grep -q "def snmf_gate" $TB/neural_methods/model/FactorizePhys/FactorizePhysNF.py || fail "PATCH55_MISSING"; [ -f $NFYAML ] && [ -d $PFA/noise_gt_hyb2 ] || fail "NF_ASSETS_MISSING"
nf_train() { local M=$1 SEED=$2 ARM=$3 ck y; ck=$(ckpt $M)
  [ -f "$ck" ] && { echo "skip train $M"; return; }
  python scripts/vipl_yaml.py train $M 1 $CSV 10 --test_csv $MMPD_HO --template $NFYAML --lr 1e-4 > /dev/null; y=$PFA/yaml_VIPL/finetune_$M.yaml
  grep -q "NAME: FactorizePhys" $y && grep -q "FILE_LIST_PATH: $CSV" $y && grep -q "FSAM_Res.pth" $y || fail "${M}_YAML"
  echo "$(ts) TRAIN $M start (arm $ARM: $NF_ENV; recipe SpecPhys-L; seed $SEED, lr 1e-4, in-run test MMPD 7-33)"
  ( export $NF_ENV RPPG_CACHE_U8=1 NF_W_DBL=0.5 RPPG_TAUG=1 RPPG_TAUG_P=0.25 RPPG_TAUG_HR_SPLIT=300; RPPG_SEED=$SEED run_gated $GATE $y /tmp/ft_$M.log ${M}_TRAIN )
  [ -f "$ck" ] || fail "${M}_TRAIN_FAIL"; prune $M
  grep -aq "^NF stats epoch 0" /tmp/ft_$M.log || fail "${M}_NF_INACTIVE"; grep -a "\[nf\] active" /tmp/ft_$M.log | grep -q "NF_W_DBL.: 0.5" || fail "${M}_DBL_INACTIVE"
  grep -aq "\[skinbg\] active" /tmp/ft_$M.log || fail "${M}_SKINBG_INACTIVE"; grep -aq "\[nf-ref\] active" /tmp/ft_$M.log && grep -a "NF-REF stats epoch 9" /tmp/ft_$M.log | grep -q "bg=[1-9]" || fail "${M}_NFREF_INACTIVE"
  grep -a "TAUG stats epoch 9" /tmp/ft_$M.log | grep -q "x2" || fail "${M}_TAUG_INACTIVE"
  case $ARM in
    ORIGA|BASE) grep -aq "nf-arch" /tmp/ft_$M.log && fail "${M}_ARCH_UNEXPECTED";;
    *) grep -aq "\[nf-arch\] arch=" /tmp/ft_$M.log || fail "${M}_ARCH_INACTIVE"; grep -a "\[nf-arch\]" /tmp/ft_$M.log | head -1;;
  esac
  case $ARM in NSG|NSG0|SGF|SNMF) grep -aq "NSG stats epoch 9" /tmp/ft_$M.log || fail "${M}_NSGSTATS_MISSING"; grep -a "NSG stats epoch \(0\|9\)" /tmp/ft_$M.log;; esac
  [ $ARM = NSG ] && { grep -a "NSG stats epoch 9" /tmp/ft_$M.log | grep -q "w=0.5" && grep -a "NSG stats epoch 0" /tmp/ft_$M.log | grep -q "ngt=[1-9]" || fail "${M}_NSG_SUP_INACTIVE"; }
  grep -a "NF stats epoch 9" /tmp/ft_$M.log | cut -c1-160; cp /tmp/ft_$M.log $RES/chain117_ft_$M.log
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
  python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1 || fail "${M}_AGG"; cp /tmp/agg_$M.log $RES/chain117_agg_$M.log; echo "$(ts) AGG $M: $(grep -E 'RTNI|NIRP_LARGE' /tmp/agg_$M.log | head -2 | tr '\n' ' ' | cut -c1-160)"; }
SEEDS="1100 1200 1300 1400 1500"; ALL=""; for S in $SEEDS; do for ARM in BASE ${ARMS:?ARMS} ORIGA; do ALL="$ALL $ARM:$S"; done; done
case $Q in
  alpha) JOBS=$(echo $ALL | tr ' ' '\n' | awk 'NR%6==1' | tr '\n' ' ');;
  pro6000) JOBS=$(echo $ALL | tr ' ' '\n' | awk 'NR%6!=1' | tr '\n' ' ');;
  *) fail "unknown queue $Q";;
esac
echo "$(ts) [$LANE] start queue=$Q jobs=$(echo $JOBS | wc -w)"
for J in $JOBS; do ARM=${J%%:*}; S=${J#*:}; M=NF_${ARM}_L_s$S; [ $ARM = BASE ] && M=NF_SKREF_REP_L_s$S
  [ -f /tmp/chain117_job_$M.done ] && continue
  exec 8>/tmp/chain117_job_$M.lock; flock -n 8 || { exec 8>&-; continue; }
  [ -f /tmp/chain117_job_$M.done ] && { exec 8>&-; continue; }
  NF_ENV="$BASE_ENV $(arm_env $ARM)"
  nf_train $M $S $ARM; nf_eval_vipl $M; nf_eval_axes $M
  touch /tmp/chain117_job_$M.done; echo "$(ts) [$LANE] $M DONE"; exec 8>&-
done
touch /tmp/chain117_$LANE.done; echo "$(ts) [$LANE] LANE_DONE"
