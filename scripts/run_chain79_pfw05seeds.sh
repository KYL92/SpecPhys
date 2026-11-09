#!/bin/bash
# 체인 79 (59 문서 §6.5, 2026-11-02; **Alpha**): PhysFormer + 스펙트럼 가중치 0.5 시드 확장 PF_SPEC_W05_L_s200·_s300 (체인 78 의 시드 1 결과 8.02/9.76 을 3시드로)
#  레시피 = yaml24(PhysFormer 공식: 공개 UBFC 가중치, LR 5e-5·10 epoch·배치 4, 차분 정규화 128×128 청크 160, RPPG_HR_LABEL=cumsum) — 고유 손실 풀 셀 CoTAM_L 과 동일, 데이터만 u8 풀 캐시(수치 동일)
#  → 학습 내 테스트 = MMPD held-out 7~33(u8) → VIPL(u8) → 실 축(float real_* 캐시, 있는 것만) → agg_model2 / agg_mmpd_cv2 / agg_vipl
#  레인 A = 시드 100·300, B = 시드 200 (GPU 32 GB 중 타 사용자 4 GB; PhysFormer ≈9 GB). 캐시(풀 u8·MMPD u8)가 준비될 때까지 대기.
#  사용: nohup bash scripts/run_chain79_pfspec.sh A >> /tmp/chain79_A.log 2>&1 &   (마커 /tmp/chain79_$LANE.done|.fail)
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
LANE=${1:?LANE}; exec 9>/tmp/chain79_$LANE.lock; flock -n 9 || { echo "chain79 $LANE already running (lock) -> exit"; exit 0; }
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; TB=$ROOT/repos/rPPG-Toolbox
POOL_CSV=$LISTS/cotrain_allf_hyb4_u8_pool.csv; MMPD_HO=$LISTS/mmpd_zs_7_33.csv; TPL24=$PFA/yaml24/finetune_CoTAF_DM_L.yaml
AXES="PURE NIRP_STILL NIRP_SMALL NIRP_LARGE DLCN MCD_A MCD_B RTNI RTNC REALTEST C3 C6 E3"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) [$LANE] $1 -> abort"; touch /tmp/chain79_$LANE.fail; exit 1; }
prune() { find $PFA/runs/finetune_$1/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; }
ckpt() { echo "$PFA/runs/finetune_$1/PreTrainedModels/SYNTH_${1}_physformer_Epoch9.pth"; }
has_out() { ls $1/*/saved_test_outputs/*.pickle > /dev/null 2>&1; }
csv_ok() { python - "$1" <<'PY'
import csv, os, sys
rows = list(csv.DictReader(open(sys.argv[1]))); sys.exit(0 if rows and all(os.path.exists(r["input_files"]) for r in rows) else 1)
PY
}
grep -q "RPPG_SPEC" $TB/neural_methods/trainer/PhysFormerTrainer.py || fail "PATCH43_MISSING"
echo "$(ts) [$LANE] waiting for u8 caches (pool csv + MMPD held-out csv paths)"
while ! { [ -f $LISTS/cotrain_allf_hyb4_u8_0.0_1.0.csv ] && csv_ok $POOL_CSV && grep -q "DONE" /tmp/mmpd_u8_alpha.log 2>/dev/null && csv_ok $MMPD_HO; }; do sleep 120; done
echo "$(ts) [$LANE] caches ready"
pf_train() { local M=$1 SEED=$2 ck y; ck=$(ckpt $M)
  [ -f "$ck" ] && { echo "skip train $M"; return; }
  python scripts/vipl_yaml.py train $M 1 $POOL_CSV 10 --test_csv $MMPD_HO --template $TPL24 > /dev/null; y=$PFA/yaml_VIPL/finetune_$M.yaml
  grep -q "NAME: PhysFormer" $y && grep -q "FILE_LIST_PATH: $POOL_CSV" $y && grep -q "UBFC-rPPG_PhysFormer_DiffNormalized.pth" $y && grep -q "LR: 5e-5\|LR: 5.0e-05" $y && grep -q "DATA_FORMAT: NCDHW" $y || fail "${M}_YAML"
  echo "$(ts) TRAIN $M start (PhysFormer, seed $SEED, lr 5e-5, RPPG_SPEC=1 $XENV, pool u8 chunk 160, in-run test MMPD 7-33)"
  ( export RPPG_CACHE_U8=1 RPPG_SPEC=1 $XENV; RPPG_SEED=$SEED run_gated $GATE $y /tmp/ft_$M.log ${M}_TRAIN )
  [ -f "$ck" ] || fail "${M}_TRAIN_FAIL"; prune $M
  grep -aq "\[finetune\] loaded pretrained weights\|loaded pretrained\|RESUME" /tmp/ft_$M.log || echo "$(ts) WARN ${M}: resume line not found"
  grep -aq "\[spec\] active" /tmp/ft_$M.log && grep -aq "SPEC stats epoch 9" /tmp/ft_$M.log || fail "${M}_SPEC_INACTIVE"; grep -a "SPEC stats epoch 9" /tmp/ft_$M.log
  cp /tmp/ft_$M.log $RES/chain79_ft_$M.log
  python scripts/agg_mmpd_cv2.py $M > /tmp/agg_${M}_MMPD.log 2>&1 || fail "${M}_AGG_MMPD"; echo "$(ts) $M MMPD: $(sed -n 2p /tmp/agg_${M}_MMPD.log | cut -c1-120)"; }
pf_eval_vipl() { local M=$1 ck=$(ckpt $1); [ -f $RES/agg_${M}_VIPL.json ] && return
  python scripts/vipl_yaml.py ev $M "$ck" > /dev/null; grep -q "MODEL_PATH: $ck" $PFA/yaml_VIPL/ev_${M}_VIPL.yaml || fail "${M}_VIPL_YAML"
  echo "$(ts) EVAL $M VIPL start"; ( export RPPG_CACHE_U8=1; run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL )
  has_out $PFA/runs/ev_${M}_VIPL || fail "${M}_VIPL_FAIL"
  python scripts/agg_vipl.py $M > /tmp/agg_${M}_VIPL.log 2>&1 || fail "${M}_AGG_VIPL"; echo "$(ts) $M VIPL: $(grep -E 'all \[문헌' /tmp/agg_${M}_VIPL.log | head -1 | cut -c1-120)"; }
pf_eval_axes() { local M=$1 A f EXP
  gen_yaml $M ALLF_HYB4 allf_hyb4 pfspec > /dev/null   # yaml24 복제: ev yaml 13축 + MMPD_B(미사용), MODEL_PATH = 이 셀의 Epoch9
  for A in $AXES; do f=$PFA/yaml_$M/ev_${M}_$A.yaml; [ -f $f ] || continue; has_out $PFA/runs/ev_${M}_$A && continue
    EXP=$(grep -m1 "EXP_DATA_NAME" $f | awk '{print $2}'); [ -f $LISTS/${EXP}_0.0_1.0.csv ] && [ -d $PFA/cache/$EXP ] || { echo "$(ts) skip $A (cache $EXP missing)"; continue; }
    sed -i "s/DO_PREPROCESS: true/DO_PREPROCESS: false/" $f; run_gated 6000 $f /tmp/ev_${M}_$A.log ${M}_$A
    has_out $PFA/runs/ev_${M}_$A || echo "$(ts) WARN ${M}_${A}: no output"; done
  python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1 || fail "${M}_AGG"; cp /tmp/agg_$M.log $RES/chain79_agg_$M.log
  echo "$(ts) AGG $M: $(grep -E 'RTNI|NIRP_LARGE' /tmp/agg_$M.log | head -2 | tr '\n' ' ' | cut -c1-160)"; }
case $LANE in
  A) XENV="SPEC_W_SPEC=0.5"; M=PF_SPEC_W05_L_s200; pf_train $M 200; pf_eval_vipl $M; pf_eval_axes $M; echo "$(ts) [$LANE] $M DONE";;
  B) sleep 240; XENV="SPEC_W_SPEC=0.5"; M=PF_SPEC_W05_L_s300; pf_train $M 300; pf_eval_vipl $M; pf_eval_axes $M; echo "$(ts) [$LANE] $M DONE";;
  *) fail "unknown lane $LANE";;
esac
touch /tmp/chain79_$LANE.done; echo "$(ts) [$LANE] LANE_DONE"
