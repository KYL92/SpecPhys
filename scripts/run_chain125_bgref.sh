#!/bin/bash
# 체인 125 (66 문서, 사전 등록): 입력 단계 배경 기준 보정 — 물리 규칙 게이트(패치 57b, 학습 없는 게이트). 시드 2600~3000, 풀 = 체인 124 와 같음(3,953 청크).
#   BGD NF_BGREF_L     : 게이트 보정을 학습과 평가에 모두 적용(새로 학습)
#   BGT NF_NSP2DR_BGT_L : 체인 124 기준 체크포인트(NF_NSP2DR_L)에 평가 때만 적용(학습 없음)
#   기준은 체인 124 의 NF_NSP2DR_L(같은 풀·같은 시드).
# 사용: nohup bash scripts/run_chain125_bgref.sh LANE pro6000 >> /tmp/chain125_LANE.log 2>&1 &
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
LANE=${1:?LANE}; Q=${2:?QUEUE}; exec 9>/tmp/chain125_$LANE.lock; flock -n 9 || { echo "chain125 $LANE already running (lock) -> exit"; exit 0; }
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; TB=$ROOT/repos/rPPG-Toolbox; TPLN=CoTAM_HYB4_L
CSV=$LISTS/cotrain_allf_hyb4_dr_u8_pool.csv; NCH=3953; SEPCSV=$LISTS/sepeval_u8.csv; MMPD_HO=$LISTS/mmpd_zs_7_33.csv; NFYAML=$PFA/yaml_NF_HYB_L/finetune_NF_HYB_L.yaml
NF_ENV="RPPG_NF=1 RPPG_SKINW=0.2 NF_SS_TRACE=1 NF_SS_REF=1 NF_ARCH=nsp NF_NSP_LAM=0.01 HYB_NOISE_GT_DIR=$PFA/noise_gt_dr NF_BGREF=1 NF_BGREF_MODE=gate RPPG_PARSE_CH=1"
AXES="PURE NIRP_STILL NIRP_SMALL NIRP_LARGE DLCN MCD_A MCD_B RTNI RTNC REALTEST"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) [$LANE] $1 -> abort"; touch /tmp/chain125_$LANE.fail; exit 1; }
prune() { find $PFA/runs/finetune_$1/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; }
ckpt() { ls $PFA/runs/finetune_$1/PreTrainedModels/*_Epoch9.pth 2>/dev/null | head -1; }
has_out() { ls $1/*/saved_test_outputs/*.pickle > /dev/null 2>&1; }
grep -q "patch 57" $TB/neural_methods/trainer/FactorizePhysTrainer.py && grep -q 'NF_BGREF_MODE", "learn") == "gate"' $TB/neural_methods/model/FactorizePhys/FactorizePhysNF.py && grep -q "RPPG_PARSE_CH" $TB/dataset/data_loader/BaseLoader.py || fail "PATCH57B_MISSING"
[ -f $CSV ] && [ -f $SEPCSV ] && [ -d $PFA/noise_gt_dr ] && [ -f $NFYAML ] || fail "ASSETS_MISSING"
ev_check() { grep -aq "\[bgref\] module active: mode=gate" $1 && grep -aq "\[parse-ch\] active" $1 || fail "$2_EVAL_WITHOUT_BGREF"; }
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
  ev_check /tmp/ft_$M.log ${M}_TRAIN; grep -a "^BGREF stats epoch 9" /tmp/ft_$M.log | grep -q "ok=[1-9]" || fail "${M}_BGREF_NO_BACKGROUND"
  grep -a "^BGREF stats epoch 9" /tmp/ft_$M.log | grep -q "on(a_skin>0.3)=[1-9]" || fail "${M}_GATE_NEVER_OPEN"
  grep -aE "NF stats epoch 9|BGREF stats epoch (0|9)" /tmp/ft_$M.log | cut -c1-200; cp /tmp/ft_$M.log $RES/chain125_ft_$M.log
  python scripts/agg_mmpd_cv2.py $M > /tmp/agg_${M}_MMPD.log 2>&1 || fail "${M}_AGG_MMPD"; echo "$(ts) $M MMPD: $(sed -n 2p /tmp/agg_${M}_MMPD.log | cut -c1-120)"; }
bgt_setup() { local M=$1 B=$2 y; mkdir -p $PFA/runs/finetune_$M/PreTrainedModels; ln -sfn "$(ckpt $B)" $PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_factorizephys_Epoch9.pth; [ -n "$(ckpt $M)" ] || fail "${M}_CKPT_LINK"
  has_out $PFA/runs/finetune_$M && [ -f $RES/agg_${M}_MMPD.json ] && return
  python scripts/vipl_yaml.py ev ${M}_MMTMP "$(ckpt $M)" --template $PFA/yaml_NF_HYB_L/ev_NF_HYB_L_DLCN.yaml > /dev/null; y=$PFA/yaml_VIPL/ev_${M}_MMPD.yaml
  sed -e "s#$LISTS/vipl_all.csv#$MMPD_HO#" -e "s#runs/ev_${M}_MMTMP_VIPL/outputs#runs/finetune_${M}/outputs#" -e "s#runs/ev_${M}_MMTMP_VIPL#runs/finetune_${M}#" $PFA/yaml_VIPL/ev_${M}_MMTMP_VIPL.yaml > $y; rm -f $PFA/yaml_VIPL/ev_${M}_MMTMP_VIPL.yaml
  grep -q "FILE_LIST_PATH: $MMPD_HO" $y && grep -q "runs/finetune_${M}" $y || fail "${M}_MMPD_YAML"
  echo "$(ts) EVAL $M MMPD (test-time gate on $B)"; ( export $NF_ENV RPPG_CACHE_U8=1; run_gated 6000 $y /tmp/ev_${M}_MMPD.log ${M}_MMPD )
  has_out $PFA/runs/finetune_$M || fail "${M}_MMPD_FAIL"; ev_check /tmp/ev_${M}_MMPD.log ${M}_MMPD
  python scripts/agg_mmpd_cv2.py $M > /tmp/agg_${M}_MMPD.log 2>&1 || fail "${M}_AGG_MMPD"; echo "$(ts) $M MMPD: $(sed -n 2p /tmp/agg_${M}_MMPD.log | cut -c1-120)"; }
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
  python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1 || fail "${M}_AGG"; cp /tmp/agg_$M.log $RES/chain125_agg_$M.log; echo "$(ts) AGG $M: $(grep -E 'RTNI|NIRP_LARGE' /tmp/agg_$M.log | head -2 | tr '\n' ' ' | cut -c1-160)"; }
[ $Q = pro6000 ] || fail "unknown queue $Q"
JOBS=""; for S in 2600 2700 2800 2900 3000; do JOBS="$JOBS BGD:$S BGT:$S"; done
echo "$(ts) [$LANE] start queue=$Q jobs=$(echo $JOBS | wc -w)"
for J in $JOBS; do ARM=${J%%:*}; S=${J#*:}
  if [ $ARM = BGD ]; then M=NF_BGREF_L_s$S; else M=NF_NSP2DR_BGT_L_s$S; fi
  [ -f /tmp/chain125_job_$M.done ] && continue
  exec 8>/tmp/chain125_job_$M.lock; flock -n 8 || { exec 8>&-; continue; }
  [ -f /tmp/chain125_job_$M.done ] && { exec 8>&-; continue; }
  if [ $ARM = BGD ]; then nf_train $M $S; else [ -n "$(ckpt NF_NSP2DR_L_s$S)" ] || fail "BASE_CKPT_MISSING_$S"; bgt_setup $M NF_NSP2DR_L_s$S; fi
  nf_eval_sep $M; nf_eval_vipl $M; nf_eval_axes $M
  touch /tmp/chain125_job_$M.done; echo "$(ts) [$LANE] $M DONE"; exec 8>&-
done
touch /tmp/chain125_$LANE.done; echo "$(ts) [$LANE] LANE_DONE"
