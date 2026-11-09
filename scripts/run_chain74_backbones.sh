#!/bin/bash
# 체인 74 (57 문서, 2026-11-02; pro6000 컨테이너 안 레인 병렬; 사용자 지시 '② RhythmFormer·PhysMamba 동일 풀 학습 + MMPD 7:1:2 분할 실험'):
#  R1 R2 R3  RhythmFormer 풀 셀 RF_POOL_L{,_s200,_s300}: 366 클립 하이브리드 풀(u8 캐시, 표준화 입력·라벨, 청크 160) 공개 UBFC 가중치에서 10 epoch·LR 1e-4·배치 4
#            → 학습 내 테스트 = MMPD held-out 7~33(u8) → VIPL(u8) → 실 축 9개(float 표준화 캐시 rf_real_*, 레인 공유 flock) → agg_model2
#  P1 P2 P3  PhysMamba 풀 셀 PM_POOL_L{,_s200,_s300}: 같은 풀(u8, 차분 정규화, 청크 160 — 패치 42 frames=청크 길이) 공개 가중치에서 같은 레시피·같은 평가; P1 끝에 공개 PhysMamba(PMbase) VIPL·MMPD held-out
#  M         SpecFP MMPD 문헌 분할 2단계 ML_NF_L{,_s200,_s300}: NF_HYB_L 시드 체크포인트 출발, 학습 = MMPD 1~23 + 풀(u8), 테스트 27~33, LR 5e-5(체인 73 MMPD fold 레시피) → agg_mmpd_cv2 --lit
#  MO        SpecFP MMPD 단독 ML_NFO_L{,_s200,_s300}: 공개 FactorizePhys 가중치 출발, 학습 = MMPD 1~23 만, LR 1e-4 → agg_mmpd_cv2 --lit
#  Z         레인 마커 대기 → backbone_judge.py → float 캐시 삭제
#  사용: bash scripts/run_chain74_backbones.sh LANE ; 로그 /tmp/chain74_$LANE.log, 마커 /tmp/chain74_$LANE.done | .fail. 기동(컨테이너, 레인마다):
#    docker exec -d -u user vface_train bash -lc 'cd /SPECPHYS_ROOT && RPPG_MP_QUOTA=4 RPPG_NUM_WORKERS=4 bash scripts/run_chain74_backbones.sh R1 >> /tmp/chain74_R1.log 2>&1'
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
LANE=${1:?LANE}; exec 9>/tmp/chain74_$LANE.lock; flock -n 9 || { echo "chain74 $LANE already running (lock) -> exit"; exit 0; }
cd $ROOT; GATE=${GATE:-24000}; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; TB=$ROOT/repos/rPPG-Toolbox
POOL_CSV=$LISTS/cotrain_allf_hyb4_u8_pool.csv; MMPD_HO=$LISTS/mmpd_zs_7_33.csv; LIT_TR=$LISTS/mmpd_lit_train.csv; LIT_TRJ=$LISTS/mmpd_lit_train_cotrain_allf_hyb4_u8.csv; LIT_TE=$LISTS/mmpd_lit_test.csv
NF_ENV="RPPG_NF=1 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2 NF_HR_TB=1"; NFYAML=$PFA/yaml_NF_HYB_L/finetune_NF_HYB_L.yaml; TPLN=CoTAM_HYB4_L
AXES="PURE NIRP_STILL NIRP_SMALL NIRP_LARGE DLCN MCD_A MCD_B RTNI RTNC"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) [$LANE] $1 -> abort"; touch /tmp/chain74_$LANE.fail; exit 1; }
wait_ram() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$1" ]; do echo "$(ts) RAM_WAIT avail<$1G"; sleep 120; done; }
wait_load() { while [ "$(cut -d. -f1 /proc/loadavg)" -gt "$1" ]; do echo "$(ts) LOAD_WAIT 1min>$1"; sleep 300; done; }
prune() { find $PFA/runs/finetune_$1/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; }
lower() { echo "$1" | tr 'A-Z' 'a-z'; }
bb_ckpt() { echo "$PFA/runs/finetune_$1/PreTrainedModels/SYNTH_${1}_$(lower $2)_Epoch9.pth"; }
nf_ckpt() { echo "$PFA/runs/finetune_$1/PreTrainedModels/SYNTH_${1}_factorizephys_Epoch9.pth"; }
has_out() { ls $1/*/saved_test_outputs/*.pickle > /dev/null 2>&1; }
for f in $POOL_CSV $MMPD_HO $LIT_TR $LIT_TRJ $LIT_TE $NFYAML; do [ -f $f ] || fail "FILE_MISSING $f"; done
[ "$(ls $PFA/cache/cotrain_allf_hyb4_u8/*_u8stats.npy 2>/dev/null | wc -l)" -ge 366 ] || fail "POOL_U8STATS_MISSING"
[ "$(ls $PFA/cache/mmpd_all_u8/*_u8stats.npy 2>/dev/null | wc -l)" -ge 660 ] || fail "MMPD_U8STATS_MISSING"
python scripts/bb_yaml.py tpl RhythmFormer > /dev/null && python scripts/bb_yaml.py tpl PhysMamba > /dev/null || fail "TPL_GEN"

# ---------- 백본 풀 셀(레인 R*·P*) ----------
bb_train() { local M=$1 BB=$2 SEED=$3 ck y; ck=$(bb_ckpt $M $BB)
  [ -f "$ck" ] && { echo "skip train $M"; return; }
  python scripts/vipl_yaml.py train $M 1 $POOL_CSV 10 --test_csv $MMPD_HO --template $PFA/yaml_BB/tpl_$BB.yaml --lr 1e-4 > /dev/null; y=$PFA/yaml_VIPL/finetune_$M.yaml
  grep -q "NAME: $BB" $y && grep -q "FILE_LIST_PATH: $POOL_CSV" $y && grep -q "RESUME: $TB/final_model_release" $y || fail "${M}_YAML"
  [ $BB = RhythmFormer ] && { grep -q "DATA_FORMAT: NDCHW" $y || fail "${M}_FORMAT"; }
  echo "$(ts) TRAIN $M start ($BB, seed $SEED, lr 1e-4, pool u8 chunk 160, in-run test MMPD 7-33)"
  wait_ram 14; ( export RPPG_CACHE_U8=1; RPPG_SEED=$SEED run_gated $GATE $y /tmp/ft_$M.log ${M}_TRAIN )
  [ -f "$ck" ] || fail "${M}_TRAIN_FAIL"; prune $M
  grep -aq "\[finetune\] loaded pretrained weights" /tmp/ft_$M.log || fail "${M}_NO_RESUME"
  cp /tmp/ft_$M.log $RES/chain74_ft_$M.log
  python scripts/agg_mmpd_cv2.py $M > /tmp/agg_${M}_MMPD.log 2>&1 || fail "${M}_AGG_MMPD"; echo "$(ts) $M MMPD: $(sed -n 2p /tmp/agg_${M}_MMPD.log | cut -c1-120)"; }
bb_eval_vipl() { local M=$1 BB=$2 ck=$3; [ -f $RES/agg_${M}_VIPL.json ] && return
  python scripts/bb_yaml.py ev_axis $M $BB "$ck" DLCN --preprocess 0 > /dev/null
  python scripts/vipl_yaml.py ev $M "$ck" --template $PFA/yaml_$M/ev_${M}_DLCN.yaml > /dev/null
  grep -q "NAME: $BB" $PFA/yaml_VIPL/ev_${M}_VIPL.yaml || fail "${M}_VIPL_YAML"
  echo "$(ts) EVAL $M VIPL start"; ( export RPPG_CACHE_U8=1; run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL )
  has_out $PFA/runs/ev_${M}_VIPL || fail "${M}_VIPL_FAIL"
  python scripts/agg_vipl.py $M > /tmp/agg_${M}_VIPL.log 2>&1 || fail "${M}_AGG_VIPL"; echo "$(ts) $M VIPL: $(grep -E 'all \[문헌' /tmp/agg_${M}_VIPL.log | head -1 | cut -c1-120)"; }
bb_eval_axes() { local M=$1 BB=$2 ck=$3 PFX A EXP CACHE have y
  PFX=$(python -c "import sys; sys.path.insert(0, 'scripts'); import bb_yaml; print(bb_yaml.BB['$BB']['prefix'])")
  for A in $AXES; do has_out $PFA/runs/ev_${M}_$A && continue
    EXP=$(grep -m1 "EXP_DATA_NAME" $PFA/yaml_$TPLN/ev_${TPLN}_$A.yaml | awk '{print $2}'); CACHE=${PFX}_$EXP
    ( flock 8; have=0; [ -f $LISTS/${CACHE}_0.0_1.0.csv ] && [ -d $PFA/cache/$CACHE ] && have=1
      [ $have = 0 ] && { wait_ram 16; wait_load 48; echo "$(ts) PREPROCESS $CACHE by $M"; }
      python scripts/bb_yaml.py ev_axis $M $BB "$ck" $A --preprocess $((1 - have)) > /dev/null; y=$PFA/yaml_$M/ev_${M}_$A.yaml
      [ $have = 1 ] && sed -i "s/DO_PREPROCESS: true/DO_PREPROCESS: false/" $y
      run_gated 6000 $y /tmp/ev_${M}_$A.log ${M}_$A ) 8>/tmp/chain74_cache_$CACHE.lock
    has_out $PFA/runs/ev_${M}_$A || fail "${M}_${A}_FAIL"; done
  python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1 || fail "${M}_AGG"; cp /tmp/agg_$M.log $RES/chain74_agg_$M.log
  echo "$(ts) AGG $M: $(grep -E 'RTNI|NIRP_LARGE' /tmp/agg_$M.log | head -2 | tr '\n' ' ' | cut -c1-160)"; }
lane_R() { local S M sfx ck; for S in "$@"; do sfx=""; [ $S != 100 ] && sfx=_s$S; M=RF_POOL_L$sfx
  bb_train $M RhythmFormer $S; ck=$(bb_ckpt $M RhythmFormer); bb_eval_vipl $M RhythmFormer "$ck"; bb_eval_axes $M RhythmFormer "$ck"; echo "$(ts) [$LANE] $M DONE"; done; }
lane_P() { local S M sfx ck; python -W ignore -c "import mamba_ssm, causal_conv1d" 2>/dev/null || fail "MAMBA_IMPORT"
  for S in "$@"; do sfx=""; [ $S != 100 ] && sfx=_s$S; M=PM_POOL_L$sfx
  bb_train $M PhysMamba $S; ck=$(bb_ckpt $M PhysMamba); bb_eval_vipl $M PhysMamba "$ck"; bb_eval_axes $M PhysMamba "$ck"; echo "$(ts) [$LANE] $M DONE"; done; }
pm_base() { local M=PMbase ck=$TB/final_model_release/UBFC-rPPG_PhysMamba_DiffNormalized.pth   # 공개 PhysMamba 가중치: VIPL + MMPD held-out(u8) — H-B2 의 PhysMamba 기준선
  bb_eval_vipl $M PhysMamba $ck
  if [ ! -f $RES/agg_${M}_MMPD.json ]; then
    sed "s#$LISTS/vipl_all.csv#$MMPD_HO#; s#runs/ev_${M}_VIPL#runs/ev_${M}_MMPDZS#g" $PFA/yaml_VIPL/ev_${M}_VIPL.yaml > $PFA/yaml_VIPL/ev_${M}_MMPDZS.yaml
    grep -q "$MMPD_HO" $PFA/yaml_VIPL/ev_${M}_MMPDZS.yaml || fail "${M}_MMPDZS_YAML"
    ( export RPPG_CACHE_U8=1; run_gated 6000 $PFA/yaml_VIPL/ev_${M}_MMPDZS.yaml /tmp/ev_${M}_MMPDZS.log ${M}_MMPDZS )
    python scripts/agg_mmpd_cv2.py --outdir $PFA/runs/ev_${M}_MMPDZS $M > /tmp/agg_${M}_MMPD.log 2>&1 || fail "${M}_AGG_MMPD"; echo "$(ts) $M MMPD: $(sed -n 2p /tmp/agg_${M}_MMPD.log | cut -c1-120)"; fi; }
# ---------- SpecFP MMPD 문헌 분할(레인 M·MO) ----------
lit_train() { local M=$1 TR=$2 RCK=$3 SEED=$4 LR=$5 ck y; ck=$(nf_ckpt $M)   # lit_train NAME TRAIN_CSV RESUME_CKPT(""=템플릿의 공개 가중치) SEED LR
  [ -f "$ck" ] && { echo "skip train $M"; return; }
  if [ -n "$RCK" ]; then [ -f "$RCK" ] || fail "${M}_RESUME_MISSING $RCK"; python scripts/vipl_yaml.py train $M 1 $TR 10 --test_csv $LIT_TE --template $NFYAML --resume "$RCK" --lr $LR > /dev/null
  else python scripts/vipl_yaml.py train $M 1 $TR 10 --test_csv $LIT_TE --template $NFYAML --lr $LR > /dev/null; fi
  y=$PFA/yaml_VIPL/finetune_$M.yaml; grep -q "NAME: FactorizePhys" $y && grep -q "FILE_LIST_PATH: $TR" $y && grep -q "FILE_LIST_PATH: $LIT_TE" $y || fail "${M}_YAML"
  echo "$(ts) TRAIN $M start (SpecFP, train $(basename $TR), resume $(basename "${RCK:-public_FactorizePhys}"), seed $SEED, lr $LR, test MMPD 27-33)"
  wait_ram 14; ( export $NF_ENV RPPG_CACHE_U8=1; RPPG_SEED=$SEED run_gated $GATE $y /tmp/ft_$M.log ${M}_TRAIN )
  [ -f "$ck" ] || fail "${M}_TRAIN_FAIL"; prune $M
  grep -aq "^NF stats epoch 0" /tmp/ft_$M.log || fail "${M}_NF_INACTIVE"; grep -a "\[nf\] active" /tmp/ft_$M.log | grep -q "hr_tb=1" || fail "${M}_HRTB_INACTIVE"
  grep -a "NAN_SKIPPED" /tmp/ft_$M.log | tail -n 1; grep -a "NF stats epoch 9" /tmp/ft_$M.log | cut -c1-160; cp /tmp/ft_$M.log $RES/chain74_ft_$M.log
  python scripts/agg_mmpd_cv2.py --lit $M > /tmp/agg_${M}_MMPD.log 2>&1 || fail "${M}_AGG_LIT"; echo "$(ts) $M MMPD lit: $(sed -n 2p /tmp/agg_${M}_MMPD.log | cut -c1-120)"; }
lane_M() { local S sfx; for S in 100 200 300; do sfx=""; [ $S != 100 ] && sfx=_s$S; lit_train ML_NF_L$sfx $LIT_TRJ "$(nf_ckpt NF_HYB_L$sfx)" $S 5e-5; done; }
lane_MO() { local S sfx; for S in 100 200 300; do sfx=""; [ $S != 100 ] && sfx=_s$S; lit_train ML_NFO_L$sfx $LIT_TR "" $S 1e-4; done; }
# ---------- 판정(레인 Z) ----------
lane_Z() { local L c; for L in R1 R2 R3 P1 P2 P3 M MO; do while [ ! -f /tmp/chain74_$L.done ] && [ ! -f /tmp/chain74_$L.fail ]; do sleep 300; done; done
  echo "$(ts) all lanes finished; fails: $(ls /tmp/chain74_*.fail 2>/dev/null | tr '\n' ' ')"
  if [ -f scripts/backbone_judge.py ]; then python scripts/backbone_judge.py > $RES/backbone_judge.txt 2>&1 && echo "$(ts) JUDGE_OK" || echo "$(ts) JUDGE_FAIL"; else echo "$(ts) JUDGE_SCRIPT_MISSING (run manually)"; fi
  for c in $PFA/cache/rf_real_* $PFA/cache/pm_real_*; do [ -d "$c" ] && drop_cache $(basename $c); done; echo "$(ts) float caches dropped"; }

case $LANE in
  R1) lane_R 100;; R2) sleep 600; lane_R 200;; R3) sleep 1200; lane_R 300;;
  P1) lane_P 100; pm_base;; P2) sleep 300; lane_P 200;; P3) sleep 900; lane_P 300;;
  M) lane_M;; MO) sleep 120; lane_MO;; Z) lane_Z;;
  *) fail "unknown lane $LANE";;
esac
touch /tmp/chain74_$LANE.done; echo "$(ts) [$LANE] LANE_DONE"
