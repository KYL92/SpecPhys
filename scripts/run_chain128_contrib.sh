#!/bin/bash
# 체인 128 (69 문서, 사전 등록): 기여 근거.
#   (A) 통제 심박 시험(HRSWEEP40·HRSWEEP40LO) 평가만: 공개 FactorizePhys·RhythmFormer·PhysMamba, 이전 PhysFormer + 코퍼스(시드 5), SpecPhys 최종 시드 1600~2000(2100~2500 은 체인 127 에서 평가됨).
#   (D) 실제 학습 영상이 적을 때 합성 클립의 몫: NF_NSP2Q_L(실제 사람 1/4 + 깨끗한 합성 90클립), NF_NSP2QN_L(실제 사람 1/4 만), NF_NSP2SO_L(깨끗한 합성 90클립만), 시드 2100~2500.
# 사용: nohup bash scripts/run_chain128_contrib.sh LANE pro6000 >> /tmp/chain128_LANE.log 2>&1 &
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
LANE=${1:?LANE}; Q=${2:?QUEUE}; exec 9>/tmp/chain128_$LANE.lock; flock -n 9 || { echo "chain128 $LANE already running (lock) -> exit"; exit 0; }
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; TPLN=CoTAM_HYB4_L
HSL=$LISTS/hrsweeplo_u8.csv; HSH=$LISTS/hrsweep_u8.csv; MMPD_HO=$LISTS/mmpd_zs_7_33.csv; NFYAML=$PFA/yaml_NF_HYB_L/finetune_NF_HYB_L.yaml
B0="RPPG_NF=1 RPPG_SKINW=0.2 NF_SS_TRACE=1 NF_SS_REF=1 NF_ARCH=nsp NF_NSP_LAM=0.01"; AXES="PURE NIRP_STILL NIRP_SMALL NIRP_LARGE DLCN MCD_A MCD_B RTNI RTNC REALTEST"
NF_ENV="$B0 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2c"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) [$LANE] $1 -> abort"; touch /tmp/chain128_$LANE.fail; exit 1; }
prune() { find $PFA/runs/finetune_$1/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; }
ckpt() { ls $PFA/runs/finetune_$1/PreTrainedModels/*_Epoch9.pth 2>/dev/null | head -1; }
has_out() { ls $1/*/saved_test_outputs/*.pickle > /dev/null 2>&1; }
for a in q qn so; do [ -f $LISTS/cotrain_scarce_${a}_u8_pool.csv ] || fail "ASSETS_MISSING_$a"; done
[ -f $HSL ] && [ -f $HSH ] && [ -f $NFYAML ] && [ -d $PFA/noise_gt_hyb2c ] || fail "ASSETS_MISSING"
ls $PFA/cache/hrsweep_u8/*_u8stats.npy > /dev/null 2>&1 && ls $PFA/cache/hrsweeplo_u8/*_u8stats.npy > /dev/null 2>&1 || fail "U8STATS_MISSING"
sweep_metrics() { local M=$1
  ( HRSWEEP_SET=LO python scripts/hrsweep_metrics.py $M; python scripts/hrsweep_metrics.py $M; HRSWEEP_SET=LO HRSWEEP_BP=1 python scripts/hrsweep_metrics.py $M; HRSWEEP_BP=1 python scripts/hrsweep_metrics.py $M ) > /dev/null 2>&1
  [ -f $RES/hrsweeplo_$M.json ] && [ -f $RES/hrsweep_$M.json ] && [ -f $RES/hrsweeplo_${M}_bp.json ] && [ -f $RES/hrsweep_${M}_bp.json ] || fail "${M}_HRSWEEP_METRICS"; }
# (A) 기존 방법: VIPL 평가 yaml 을 복제해 시험 목록과 출력 경로만 바꾼다
ev_base() { local NM=$1 TAG=$2 CS=$3 SRC y; SRC=$PFA/yaml_VIPL/ev_${NM}_VIPL.yaml; has_out $PFA/runs/ev_${NM}_$TAG && return; [ -f $SRC ] || fail "${NM}_SRC_YAML"
  y=$PFA/yaml_VIPL/ev_${NM}_$TAG.yaml; sed -e "s#$LISTS/vipl_all.csv#$CS#" -e "s#runs/ev_${NM}_VIPL#runs/ev_${NM}_$TAG#g" $SRC > $y
  grep -q "FILE_LIST_PATH: $CS" $y && grep -q "runs/ev_${NM}_$TAG" $y && ! grep -q "runs/ev_${NM}_VIPL" $y || fail "${NM}_${TAG}_YAML"
  echo "$(ts) EVAL $NM $TAG start ($(grep -m1 'NAME:' $y | tr -s ' '))"
  ( export RPPG_CACHE_U8=1; run_gated 6000 $y /tmp/ev_${NM}_$TAG.log ${NM}_$TAG ); has_out $PFA/runs/ev_${NM}_$TAG || fail "${NM}_${TAG}_FAIL"; echo "$(ts) $NM $TAG done"; }
nf_eval_set() { local M=$1 TAG=$2 CS=$3 ck=$(ckpt $1) y; [ -n "$ck" ] || fail "${M}_NO_CKPT"; has_out $PFA/runs/ev_${M}_$TAG && return
  python scripts/vipl_yaml.py ev ${M}_${TAG}TMP "$ck" --template $PFA/yaml_NF_HYB_L/ev_NF_HYB_L_DLCN.yaml > /dev/null; y=$PFA/yaml_VIPL/ev_${M}_$TAG.yaml
  sed -e "s#$LISTS/vipl_all.csv#$CS#" -e "s#ev_${M}_${TAG}TMP_VIPL#ev_${M}_$TAG#g" $PFA/yaml_VIPL/ev_${M}_${TAG}TMP_VIPL.yaml > $y; rm -f $PFA/yaml_VIPL/ev_${M}_${TAG}TMP_VIPL.yaml
  grep -q "FILE_LIST_PATH: $CS" $y && grep -q "MODEL_PATH: $ck" $y || fail "${M}_${TAG}_YAML"
  ( export $NF_ENV RPPG_CACHE_U8=1; run_gated 6000 $y /tmp/ev_${M}_$TAG.log ${M}_$TAG ); has_out $PFA/runs/ev_${M}_$TAG || fail "${M}_${TAG}_FAIL"; echo "$(ts) $M $TAG done"; }
# (D) 학습
nf_train() { local M=$1 SEED=$2 ARM=$3 ck y CSV NCH; ck=$(ckpt $M); [ -n "$ck" ] && { echo "skip train $M"; return; }
  case $ARM in Q) NCH=1500;; QN) NCH=960;; SO) NCH=540;; *) fail "ARM_$ARM";; esac; CSV=$LISTS/cotrain_scarce_$(echo $ARM | tr A-Z a-z)_u8_pool.csv
  [ "$(($(wc -l < $CSV) - 1))" = "$NCH" ] || fail "${M}_CSV_COUNT"
  python scripts/vipl_yaml.py train $M 1 $CSV 10 --test_csv $MMPD_HO --template $NFYAML --lr 1e-4 > /dev/null; y=$PFA/yaml_VIPL/finetune_$M.yaml
  grep -q "NAME: FactorizePhys" $y && grep -q "FILE_LIST_PATH: $CSV" $y && grep -q "FSAM_Res.pth" $y || fail "${M}_YAML"
  echo "$(ts) TRAIN $M start (arm $ARM, $NCH chunks; seed $SEED, lr 1e-4, in-run test MMPD 7-33)"
  ( export $NF_ENV RPPG_CACHE_U8=1 NF_W_DBL=0.5 RPPG_TAUG=1 RPPG_TAUG_P=0.25 RPPG_TAUG_HR_SPLIT=300; RPPG_SEED=$SEED run_gated $GATE $y /tmp/ft_$M.log ${M}_TRAIN )
  ck=$(ckpt $M); [ -n "$ck" ] || fail "${M}_TRAIN_FAIL"; prune $M
  grep -aq "^NF stats epoch 0" /tmp/ft_$M.log || fail "${M}_NF_INACTIVE"; grep -a "\[nf\] active" /tmp/ft_$M.log | grep -q "NF_W_DBL.: 0.5" || fail "${M}_DBL_INACTIVE"
  grep -aq "\[skinbg\] active" /tmp/ft_$M.log || fail "${M}_SKINBG_INACTIVE"; grep -a "NF-REF stats epoch 9" /tmp/ft_$M.log | grep -q "bg=[1-9]" || fail "${M}_NFREF_INACTIVE"
  grep -a "TAUG stats epoch 9" /tmp/ft_$M.log | grep -q "x2" || fail "${M}_TAUG_INACTIVE"; grep -aq "\[nf-arch\] arch=nsp" /tmp/ft_$M.log || fail "${M}_ARCH_INACTIVE"
  if [ $ARM = QN ]; then grep -a "NF stats epoch 9" /tmp/ft_$M.log | grep -q "ngt_hit=0 " || fail "${M}_NOISE_LABEL_UNEXPECTED"
  else grep -a "NF stats epoch 9" /tmp/ft_$M.log | grep -q "ngt_hit=[1-9]" || fail "${M}_NOISE_LABEL_INACTIVE"; fi
  grep -a "NF stats epoch 9" /tmp/ft_$M.log | grep -q "gate=[0-9]*/$NCH " || fail "${M}_CHUNK_COUNT"
  grep -aq "\[lsub\]\|\[bgref\]" /tmp/ft_$M.log && fail "${M}_MODULE_LEAK"
  grep -a "NF stats epoch 9" /tmp/ft_$M.log | cut -c1-170; cp /tmp/ft_$M.log $RES/chain128_ft_$M.log
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
  python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1 || fail "${M}_AGG"; cp /tmp/agg_$M.log $RES/chain128_agg_$M.log; echo "$(ts) AGG $M: $(grep -E 'RTNI|NIRP_LARGE' /tmp/agg_$M.log | head -2 | tr '\n' ' ' | cut -c1-160)"; }
[ $Q = pro6000 ] || fail "unknown queue $Q"
JOBS="E:FPbase E:RFbase E:PMbase E:CoTAF_TW_L E:CoTAF_TW_L_s200 E:CoTAF_TW_L_s300 E:CoTAF_TW_L_s400 E:CoTAF_TW_L_s500"
for S in 1600 1700 1800 1900 2000; do JOBS="$JOBS F:$S"; done
for S in 2100 2200 2300 2400 2500; do for A in Q QN SO; do JOBS="$JOBS $A:$S"; done; done
echo "$(ts) [$LANE] start queue=$Q jobs=$(echo $JOBS | wc -w)"
for J in $JOBS; do K=${J%%:*}; V=${J#*:}
  case $K in E) M=$V;; F) M=NF_NSP2_L_s$V;; *) M=NF_NSP2${K}_L_s$V;; esac
  [ -f /tmp/chain128_job_$M.done ] && continue
  exec 8>/tmp/chain128_job_$M.lock; flock -n 8 || { exec 8>&-; continue; }
  [ -f /tmp/chain128_job_$M.done ] && { exec 8>&-; continue; }
  case $K in
    E) ev_base $M HRSWEEPLO $HSL; ev_base $M HRSWEEP $HSH;;
    F) nf_eval_set $M HRSWEEPLO $HSL; nf_eval_set $M HRSWEEP $HSH;;
    *) nf_train $M $V $K; nf_eval_set $M HRSWEEPLO $HSL; nf_eval_set $M HRSWEEP $HSH; nf_eval_vipl $M; nf_eval_axes $M;;
  esac
  sweep_metrics $M; touch /tmp/chain128_job_$M.done; echo "$(ts) [$LANE] $M DONE"; exec 8>&-
done
touch /tmp/chain128_$LANE.done; echo "$(ts) [$LANE] LANE_DONE"
