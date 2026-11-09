#!/bin/bash
# 체인 84 (60 문서 §6 갈래 B, 2026-11-02; **Alpha**): SpecFP on 풀 + 저·고심박 합성 40 클립(HRX40 → COTRAIN_HRX40 subject251~290, u8 캐시 hrx40_u8, 합산 목록 cotrain_hrx_u8_pool.csv) NF_HRX_L{,_s200,_s300} — 기준 NF_HYB_L 과 같은 레시피(NF_SUP=1, 증강 없음)
#  → 학습 내 테스트 MMPD held-out(u8) → VIPL(u8) → 실 축(fp_real_* float 캐시, Alpha 보존) → agg. 레인 A = 시드 100·300, B = 시드 200. 사용: nohup bash scripts/run_chain84_nftaug.sh A >> /tmp/chain84_A.log 2>&1 &
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
LANE=${1:?LANE}; exec 9>/tmp/chain84_$LANE.lock; flock -n 9 || { echo "chain84 $LANE already running (lock) -> exit"; exit 0; }
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; TB=$ROOT/repos/rPPG-Toolbox; TPLN=CoTAM_HYB4_L
POOL_CSV=$LISTS/cotrain_allf_hyb4_u8_pool.csv; MMPD_HO=$LISTS/mmpd_zs_7_33.csv; NFYAML=$PFA/yaml_NF_HYB_L/finetune_NF_HYB_L.yaml
NF_ENV="RPPG_NF=1 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"; HRX_CSV=$LISTS/cotrain_hrx_u8_pool.csv; AXES="PURE NIRP_STILL NIRP_SMALL NIRP_LARGE DLCN MCD_A MCD_B RTNI RTNC REALTEST"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) [$LANE] $1 -> abort"; touch /tmp/chain84_$LANE.fail; exit 1; }
prune() { find $PFA/runs/finetune_$1/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; }
ckpt() { echo "$PFA/runs/finetune_$1/PreTrainedModels/SYNTH_${1}_factorizephys_Epoch9.pth"; }
has_out() { ls $1/*/saved_test_outputs/*.pickle > /dev/null 2>&1; }
grep -q "patch 44" $TB/dataset/data_loader/BaseLoader.py || fail "PATCH44_MISSING"; [ -f $NFYAML ] && [ -d $PFA/noise_gt_hyb2 ] || fail "NF_ASSETS_MISSING"
nf_train() { local M=$1 SEED=$2 ck y; ck=$(ckpt $M)
  [ -f "$ck" ] && { echo "skip train $M"; return; }
  python scripts/vipl_yaml.py train $M 1 $HRX_CSV 10 --test_csv $MMPD_HO --template $NFYAML --lr 1e-4 > /dev/null; y=$PFA/yaml_VIPL/finetune_$M.yaml
  grep -q "NAME: FactorizePhys" $y && grep -q "FILE_LIST_PATH: $HRX_CSV" $y && grep -q "FSAM_Res.pth" $y || fail "${M}_YAML"
  echo "$(ts) TRAIN $M start (SpecFP, pool+HRX40 u8 csv $(wc -l < $HRX_CSV) rows, seed $SEED, lr 1e-4, in-run test MMPD 7-33)"
  ( export $NF_ENV RPPG_CACHE_U8=1; RPPG_SEED=$SEED run_gated $GATE $y /tmp/ft_$M.log ${M}_TRAIN )
  [ -f "$ck" ] || fail "${M}_TRAIN_FAIL"; prune $M
  grep -aq "^NF stats epoch 0" /tmp/ft_$M.log || fail "${M}_NF_INACTIVE"; grep -a "NF stats epoch 9" /tmp/ft_$M.log | grep -q "gate=[1-9]" || fail "${M}_NF_GATE0"
  grep -a "NF stats epoch 9" /tmp/ft_$M.log | cut -c1-160; cp /tmp/ft_$M.log $RES/chain84_ft_$M.log
  python scripts/agg_mmpd_cv2.py $M > /tmp/agg_${M}_MMPD.log 2>&1 || fail "${M}_AGG_MMPD"; echo "$(ts) $M MMPD: $(sed -n 2p /tmp/agg_${M}_MMPD.log | cut -c1-120)"; }
nf_eval_vipl() { local M=$1 ck=$(ckpt $1); [ -f $RES/agg_${M}_VIPL.json ] && return
  python scripts/vipl_yaml.py ev $M "$ck" --template $PFA/yaml_NF_HYB_L/ev_NF_HYB_L_DLCN.yaml > /dev/null; grep -q "MODEL_PATH: $ck" $PFA/yaml_VIPL/ev_${M}_VIPL.yaml || fail "${M}_VIPL_YAML"
  echo "$(ts) EVAL $M VIPL start"; ( export $NF_ENV RPPG_CACHE_U8=1; run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL )
  has_out $PFA/runs/ev_${M}_VIPL || fail "${M}_VIPL_FAIL"
  python scripts/agg_vipl.py $M > /tmp/agg_${M}_VIPL.log 2>&1 || fail "${M}_AGG_VIPL"; echo "$(ts) $M VIPL: $(grep -E 'all \[문헌' /tmp/agg_${M}_VIPL.log | head -1 | cut -c1-120)"; }
nf_eval_axes() { local M=$1 A f EXP pre
  for A in $AXES; do has_out $PFA/runs/ev_${M}_$A && continue
    EXP=$(grep -m1 "EXP_DATA_NAME" $PFA/yaml_$TPLN/ev_${TPLN}_$A.yaml 2>/dev/null | awk '{print $2}'); [ -n "$EXP" ] || continue
    pre=1; [ -d $PFA/cache/fp_$EXP ] && [ -f $LISTS/fp_${EXP}_0.0_1.0.csv ] && pre=0
    python scripts/fp_yaml.py ev $M $TPLN $A --preprocess $pre > /dev/null || { echo "$(ts) skip $A (yaml)"; continue; }
    ( export $NF_ENV; run_gated 6000 $PFA/yaml_$M/ev_${M}_$A.yaml /tmp/ev_${M}_$A.log ${M}_$A ); has_out $PFA/runs/ev_${M}_$A || echo "$(ts) WARN ${M}_${A}: no output"; done
  python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1 || fail "${M}_AGG"; cp /tmp/agg_$M.log $RES/chain84_agg_$M.log; echo "$(ts) AGG $M: $(grep -E 'RTNI|NIRP_LARGE' /tmp/agg_$M.log | head -2 | tr '\n' ' ' | cut -c1-160)"; }

prep_hrx() { local k n
  echo "$(ts) [$LANE] waiting for HRX40 build (4 shards)"; while [ "$(grep -l '\[build\] done' /tmp/hrx40_build_[0-3].log 2>/dev/null | wc -l)" -lt 4 ]; do sleep 120; done
  n=$(ls $PFA/HRX40/subject*/ground_truth.txt 2>/dev/null | wc -l); [ "$n" = 40 ] || fail "HRX40_INCOMPLETE n=$n"
  if [ ! -f $LISTS/hrx40_u8_0.0_1.0.csv ]; then
    rm -rf $PFA/COTRAIN_HRX40; mkdir -p $PFA/COTRAIN_HRX40; for k in $(seq 1 40); do ln -sfn $PFA/HRX40/subject$k $PFA/COTRAIN_HRX40/subject$((250 + k)); done
    echo "$(ts) [$LANE] u8 preprocess HRX40 (40 clips)"; ( export RPPG_MP_QUOTA=6; python scripts/pool_preprocess_u8.py COTRAIN_HRX40 hrx40_u8 > /tmp/hrx40_u8.log 2>&1 ) || fail "HRX40_U8_FAIL"; grep -q POOL_PREPROCESS_DONE /tmp/hrx40_u8.log || fail "HRX40_U8_INCOMPLETE"; fi
  python - <<'PY' || fail "HRX_CSV"
import pandas as pd
L = "/SPECPHYS_ROOT/_workspace/data/physformer/cache/DataFileLists"
a = pd.read_csv(f"{L}/cotrain_allf_hyb4_u8_pool.csv")[["input_files"]]; b = pd.read_csv(f"{L}/hrx40_u8_0.0_1.0.csv")[["input_files"]]
c = pd.concat([a, b], ignore_index=True); c.to_csv(f"{L}/cotrain_hrx_u8_pool.csv"); print("combined", len(a), "+", len(b), "=", len(c))
PY
  echo "$(ts) [$LANE] HRX csv ready: $(wc -l < $HRX_CSV) rows"; }
lane() { local S M sfx; for S in "$@"; do sfx=""; [ $S != 100 ] && sfx=_s$S; M=NF_HRX_L$sfx; nf_train $M $S; nf_eval_vipl $M; nf_eval_axes $M; echo "$(ts) [$LANE] $M DONE"; done; }
case $LANE in A) prep_hrx; lane 100 300;; B) while [ ! -f $LISTS/cotrain_hrx_u8_pool.csv ]; do sleep 120; done; sleep 300; lane 200;; *) fail "unknown lane $LANE";; esac
touch /tmp/chain84_$LANE.done; echo "$(ts) [$LANE] LANE_DONE"
