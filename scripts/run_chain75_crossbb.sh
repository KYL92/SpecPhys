#!/bin/bash
# 체인 75 (59 문서, 2026-11-02; pro6000 컨테이너 안 레인 병렬; 사용자 지시 '다음 후보 진행'):
#  RS1 RS2 RS3  RhythmFormer + SpecFP 출력 스펙트럼 손실(패치 43 RPPG_SPEC=1, HR 대역 NLL 1.0 + 반감 억제 0.5, NF_HR_TB=1) RF_SPEC_L{,_s200,_s300}: 체인 74 RF_POOL_L 과 같은 풀·레시피 → MMPD held-out(학습 내 테스트) → VIPL → 실 축 9개
#  PS1 PS2 PS3  PhysMamba + 스펙트럼 손실 PM_SPEC_L{,_s200,_s300}(차분 라벨 → 누적합 파형 위에서 손실): 같은 평가
#  PD           PhysMamba VIPL 악화 진단: PM_POOL_EP_L(시드 100, 전 epoch 보존) → epoch 0/2/4/6/9 체크포인트 VIPL 평가(agg_PM_POOL_EP_L_E{k}_VIPL) ; PM_POOL_LR5_L(LR 1e-5) → MMPD·VIPL
#  Z            레인 마커 대기 → crossbb_judge.py → float 캐시 삭제
#  사용: bash scripts/run_chain75_crossbb.sh LANE ; 로그 /tmp/chain75_$LANE.log, 마커 /tmp/chain75_$LANE.done | .fail
#    docker exec -d -u user vface_train bash -lc 'cd /SPECPHYS_ROOT && RPPG_MP_QUOTA=4 RPPG_NUM_WORKERS=4 bash scripts/run_chain75_crossbb.sh RS1 >> /tmp/chain75_RS1.log 2>&1'
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
LANE=${1:?LANE}; exec 9>/tmp/chain75_$LANE.lock; flock -n 9 || { echo "chain75 $LANE already running (lock) -> exit"; exit 0; }
cd $ROOT; GATE=${GATE:-24000}; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; TB=$ROOT/repos/rPPG-Toolbox
POOL_CSV=$LISTS/cotrain_allf_hyb4_u8_pool.csv; MMPD_HO=$LISTS/mmpd_zs_7_33.csv; TPLN=CoTAM_HYB4_L
SPEC_ENV="RPPG_SPEC=1 NF_HR_TB=1"; AXES="PURE NIRP_STILL NIRP_SMALL NIRP_LARGE DLCN MCD_A MCD_B RTNI RTNC"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) [$LANE] $1 -> abort"; touch /tmp/chain75_$LANE.fail; exit 1; }
wait_ram() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$1" ]; do echo "$(ts) RAM_WAIT avail<$1G"; sleep 120; done; }
wait_load() { while [ "$(cut -d. -f1 /proc/loadavg)" -gt "$1" ]; do echo "$(ts) LOAD_WAIT 1min>$1"; sleep 300; done; }
prune() { find $PFA/runs/finetune_$1/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; }
lower() { echo "$1" | tr 'A-Z' 'a-z'; }
bb_ckpt() { echo "$PFA/runs/finetune_$1/PreTrainedModels/SYNTH_${1}_$(lower $2)_Epoch${3:-9}.pth"; }
has_out() { ls $1/*/saved_test_outputs/*.pickle > /dev/null 2>&1; }
for f in $POOL_CSV $MMPD_HO; do [ -f $f ] || fail "FILE_MISSING $f"; done
[ "$(ls $PFA/cache/cotrain_allf_hyb4_u8/*_u8stats.npy 2>/dev/null | wc -l)" -ge 366 ] || fail "POOL_U8STATS_MISSING"
python scripts/bb_yaml.py tpl RhythmFormer > /dev/null && python scripts/bb_yaml.py tpl PhysMamba > /dev/null || fail "TPL_GEN"
grep -q "RPPG_SPEC" $TB/neural_methods/trainer/RhythmFormerTrainer.py && grep -q "RPPG_SPEC" $TB/neural_methods/trainer/PhysMambaTrainer.py || fail "PATCH43_MISSING"

# bb_train NAME BACKBONE SEED XENV LR KEEP_EPOCHS(0/1)
bb_train() { local M=$1 BB=$2 SEED=$3 XENV=$4 LR=${5:-1e-4} KEEP=${6:-0} ck y; ck=$(bb_ckpt $M $BB)
  [ -f "$ck" ] && { echo "skip train $M"; return; }
  python scripts/vipl_yaml.py train $M 1 $POOL_CSV 10 --test_csv $MMPD_HO --template $PFA/yaml_BB/tpl_$BB.yaml --lr $LR > /dev/null; y=$PFA/yaml_VIPL/finetune_$M.yaml
  grep -q "NAME: $BB" $y && grep -q "FILE_LIST_PATH: $POOL_CSV" $y && grep -q "RESUME: $TB/final_model_release" $y || fail "${M}_YAML"
  python -c "import yaml, sys; c = yaml.safe_load(open('$y')); sys.exit(0 if abs(float(c['TRAIN']['LR']) - float('$LR')) < 1e-12 else 1)" || fail "${M}_LR"   # [07:05 수정] yaml 은 1e-4 를 0.0001 로 쓰므로 문자열 비교 불가
  [ $BB = RhythmFormer ] && { grep -q "DATA_FORMAT: NDCHW" $y || fail "${M}_FORMAT"; }
  echo "$(ts) TRAIN $M start ($BB, seed $SEED, lr $LR, env '$XENV', keep_epochs $KEEP)"
  wait_ram 14; ( export RPPG_CACHE_U8=1 $XENV; RPPG_SEED=$SEED run_gated $GATE $y /tmp/ft_$M.log ${M}_TRAIN )
  [ -f "$ck" ] || fail "${M}_TRAIN_FAIL"; [ $KEEP = 1 ] || prune $M
  grep -aq "\[finetune\] loaded pretrained weights" /tmp/ft_$M.log || fail "${M}_NO_RESUME"
  case "$XENV" in *RPPG_SPEC=1*) grep -aq "\[spec\] active" /tmp/ft_$M.log && grep -aq "SPEC stats epoch 9" /tmp/ft_$M.log || fail "${M}_SPEC_INACTIVE"; grep -a "SPEC stats epoch 9" /tmp/ft_$M.log;; esac
  grep -a -i -c "nan" /tmp/ft_$M.log | grep -q "^0$" || echo "$(ts) WARN ${M}: 'nan' appears in log"
  cp /tmp/ft_$M.log $RES/chain75_ft_$M.log
  python scripts/agg_mmpd_cv2.py $M > /tmp/agg_${M}_MMPD.log 2>&1 || fail "${M}_AGG_MMPD"; echo "$(ts) $M MMPD: $(sed -n 2p /tmp/agg_${M}_MMPD.log | cut -c1-120)"; }
# bb_eval_vipl NAME BACKBONE CKPT [RUNTAG]  (RUNTAG: 다른 이름으로 집계 — epoch 곡선)
bb_eval_vipl() { local M=$1 BB=$2 ck=$3 TAG=${4:-$1}; [ -f $RES/agg_${TAG}_VIPL.json ] && return
  python scripts/bb_yaml.py ev_axis $TAG $BB "$ck" DLCN --preprocess 0 > /dev/null
  python scripts/vipl_yaml.py ev $TAG "$ck" --template $PFA/yaml_$TAG/ev_${TAG}_DLCN.yaml > /dev/null
  grep -q "NAME: $BB" $PFA/yaml_VIPL/ev_${TAG}_VIPL.yaml && grep -q "MODEL_PATH: $ck" $PFA/yaml_VIPL/ev_${TAG}_VIPL.yaml || fail "${TAG}_VIPL_YAML"
  echo "$(ts) EVAL $TAG VIPL start"; ( export RPPG_CACHE_U8=1; run_gated 6000 $PFA/yaml_VIPL/ev_${TAG}_VIPL.yaml /tmp/ev_${TAG}_VIPL.log ${TAG}_VIPL )
  has_out $PFA/runs/ev_${TAG}_VIPL || fail "${TAG}_VIPL_FAIL"
  python scripts/agg_vipl.py $TAG > /tmp/agg_${TAG}_VIPL.log 2>&1 || fail "${TAG}_AGG_VIPL"; echo "$(ts) $TAG VIPL: $(grep -E 'all \[문헌' /tmp/agg_${TAG}_VIPL.log | head -1 | cut -c1-120)"; }
bb_eval_axes() { local M=$1 BB=$2 ck=$3 PFX A EXP CACHE have y
  PFX=$(python -c "import sys; sys.path.insert(0, 'scripts'); import bb_yaml; print(bb_yaml.BB['$BB']['prefix'])")
  for A in $AXES; do has_out $PFA/runs/ev_${M}_$A && continue
    EXP=$(grep -m1 "EXP_DATA_NAME" $PFA/yaml_$TPLN/ev_${TPLN}_$A.yaml | awk '{print $2}'); CACHE=${PFX}_$EXP
    ( flock 8; have=0; [ -f $LISTS/${CACHE}_0.0_1.0.csv ] && [ -d $PFA/cache/$CACHE ] && have=1
      [ $have = 0 ] && { wait_ram 16; wait_load 48; echo "$(ts) PREPROCESS $CACHE by $M"; }
      python scripts/bb_yaml.py ev_axis $M $BB "$ck" $A --preprocess $((1 - have)) > /dev/null; y=$PFA/yaml_$M/ev_${M}_$A.yaml
      [ $have = 1 ] && sed -i "s/DO_PREPROCESS: true/DO_PREPROCESS: false/" $y
      run_gated 6000 $y /tmp/ev_${M}_$A.log ${M}_$A ) 8>/tmp/chain75_cache_$CACHE.lock
    has_out $PFA/runs/ev_${M}_$A || fail "${M}_${A}_FAIL"; done
  python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1 || fail "${M}_AGG"; cp /tmp/agg_$M.log $RES/chain75_agg_$M.log
  echo "$(ts) AGG $M: $(grep -E 'RTNI|NIRP_LARGE' /tmp/agg_$M.log | head -2 | tr '\n' ' ' | cut -c1-160)"; }
lane_spec() { local BASE=$1 BB=$2 S M sfx ck; [ $BB = PhysMamba ] && { python -W ignore -c "import mamba_ssm, causal_conv1d" 2>/dev/null || fail "MAMBA_IMPORT"; }
  for S in "$@"; do case $S in 100|200|300) ;; *) continue;; esac; sfx=""; [ $S != 100 ] && sfx=_s$S; M=${BASE}$sfx
    bb_train $M $BB $S "$SPEC_ENV" 1e-4 0; ck=$(bb_ckpt $M $BB); bb_eval_vipl $M $BB "$ck"; bb_eval_axes $M $BB "$ck"; echo "$(ts) [$LANE] $M DONE"; done; }
lane_PD() { local k ck M=PM_POOL_EP_L
  python -W ignore -c "import mamba_ssm, causal_conv1d" 2>/dev/null || fail "MAMBA_IMPORT"
  bb_train $M PhysMamba 100 "" 1e-4 1
  for k in 0 2 4 6 9; do ck=$(bb_ckpt $M PhysMamba $k); [ -f "$ck" ] || fail "${M}_E${k}_CKPT_MISSING"; bb_eval_vipl $M PhysMamba "$ck" ${M}_E$k; done
  M=PM_POOL_LR5_L; bb_train $M PhysMamba 100 "" 1e-5 0; ck=$(bb_ckpt $M PhysMamba); bb_eval_vipl $M PhysMamba "$ck"; echo "$(ts) [$LANE] PD DONE"; }
lane_Z() { local L c; for L in RS1 RS2 RS3 PS1 PS2 PS3 PD; do while [ ! -f /tmp/chain75_$L.done ] && [ ! -f /tmp/chain75_$L.fail ]; do sleep 300; done; done
  echo "$(ts) all lanes finished; fails: $(ls /tmp/chain75_*.fail 2>/dev/null | tr '\n' ' ')"
  if [ -f scripts/crossbb_judge.py ]; then python scripts/crossbb_judge.py > $RES/crossbb_judge.txt 2>&1 && echo "$(ts) JUDGE_OK" || echo "$(ts) JUDGE_FAIL"; else echo "$(ts) JUDGE_SCRIPT_MISSING"; fi
  for c in $PFA/cache/rf_real_* $PFA/cache/pm_real_*; do [ -d "$c" ] && drop_cache $(basename $c); done; echo "$(ts) float caches dropped"; }

case $LANE in
  RS1) lane_spec RF_SPEC_L RhythmFormer 100;; RS2) sleep 600; lane_spec RF_SPEC_L RhythmFormer 200;; RS3) sleep 1200; lane_spec RF_SPEC_L RhythmFormer 300;;
  PS1) lane_spec PM_SPEC_L PhysMamba 100;; PS2) sleep 300; lane_spec PM_SPEC_L PhysMamba 200;; PS3) sleep 900; lane_spec PM_SPEC_L PhysMamba 300;;
  PD) sleep 120; lane_PD;; Z) lane_Z;;
  *) fail "unknown lane $LANE";;
esac
touch /tmp/chain75_$LANE.done; echo "$(ts) [$LANE] LANE_DONE"
