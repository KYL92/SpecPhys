#!/bin/bash
# 체인 73 (55 문서, 2026-11-01; pro6000 컨테이너 안에서 레인별 병렬 실행, 사용자 지시 '남은 후보 다 실험 + VRAM 최대 활용'):
#  남은 후보 3종 — ① 국소 교란 프록시(패치 39 NF_SS_LOCAL) ② in-domain 2단계 적응(VIPL·MMPD·MR-NIRP, 라벨 사용) ③ 증강 일관성(패치 39 NF_SS_AUG).
#  사용: bash scripts/run_chain73_candidates.sh LANE     LANE ∈ A1 A2 A3 B C D Z   (각 레인은 독립 flock·로그·마커; 동시에 기동)
#   A1 NF_SS_LOC_L ×3시드 | A2 NF_SS_AUG_L ×3 | A3 NF_SS_AUGTR_L ×3 — zero-shot: 학습 → 13축(기존 fp 캐시) → VIPL(u8) → MMPD 6묶음(레인 공유 float 캐시, 묶음별 flock) → 집계
#   B  VIPL 5-fold 2단계: VF{k}_NFs2_L(NF_HYB_L_s200 출발, 시드 200)·VF{k}_NFs3_L(_s300, 300)·VF{k}_NFSS_L(NF_SS_TR_L 출발 + NF_SS_TRACE=1, 100) → agg_vipl --cv
#      [11-01 18:10 패치 40] in-domain MMPD·MR-NIRP 첫 실행은 u8 차분 라벨 투입 사고로 무효 → 로더 복원 + NF_HR_TB=1 로 C·C2·C3·D·D2·D3 재실행; 사후 ④ 레인 T1 = NF_HYB_TB_L ×3(기준 레시피 + NF_HR_TB=1)
#      [11-01 13:50 분할] C·D 도 시드 200/300 을 C2·C3·D2·D3 레인으로 분리(시드 100 fold 는 C·D 가 완료); Z 는 A1 A2 A3 B2 B3 B4 C2 C3 D2 D3 대기
#      [11-01 11:25 분할] B 는 VIPL fold 학습이 1 run ≈60 분(동시 부하)이라 15 run 직렬이 병목 → B2(NFs2 ×5)·B3(NFs3 ×5)·B4(NFSS ×5) 세 레인으로 나눠 동시 실행(B 는 호환용으로 남김)
#   C  MMPD 5-fold 2단계: mmpd_all_u8 캐시(스트리밍, 660 클립) → MF{k}_NF_L(시드 100, NF_HYB_L 출발) → MF{k}_NFs2_L·MF{k}_NFs3_L → agg_mmpd_cv2 --cv
#   D  MR-NIRP 5-fold 2단계: nirp_train_u8 캐시(스트리밍, 184 클립) → nirp_fold_lists → NR{k}_NF_L(시드 100) → NIRP 3축 평가 → agg_fold_axes; 시드 200·300
#   Z  레인 마커 전부 대기 → candidates_judge.py → MMPD 공유 float 캐시 삭제
#  로그 /tmp/chain73_$LANE.log, 마커 /tmp/chain73_$LANE.done, 실패 /tmp/chain73_$LANE.fail. 기동(컨테이너, 레인마다):
#    docker exec -d -u user vface_train bash -lc 'cd /SPECPHYS_ROOT && RPPG_MP_QUOTA=4 RPPG_NUM_WORKERS=4 bash scripts/run_chain73_candidates.sh A1 >> /tmp/chain73_A1.log 2>&1'
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}   # [11-01 17:55] numpy/BLAS 스레드 과점유 방지(레인 7개 동시: load 171, 15 s/it 사고)
LANE=${1:?LANE}; exec 9>/tmp/chain73_$LANE.lock; flock -n 9 || { echo "chain73 $LANE already running (lock) -> exit"; exit 0; }
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; FPLR=1e-4; TPLN=CoTAM_HYB4_L; NFYAML=$PFA/yaml_NF_HYB_L/finetune_NF_HYB_L.yaml
NF_ENV="RPPG_NF=1 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) [$LANE] $1 -> abort"; touch /tmp/chain73_$LANE.fail; exit 1; }
ckpt() { echo "$PFA/runs/finetune_$1/PreTrainedModels/SYNTH_${1}_factorizephys_Epoch9.pth"; }
wait_ram() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$1" ]; do echo "$(ts) RAM_WAIT avail<$1G"; sleep 120; done; }
wait_load() { while [ "$(cut -d. -f1 /proc/loadavg)" -gt "$1" ]; do echo "$(ts) LOAD_WAIT 1min>$1"; sleep 300; done; }
cell_env() { case $1 in *_LOC_*) echo "NF_SS_TRACE=1 NF_SS_LOCAL=1";; *_AUGTR_*) echo "NF_SS_AUG=1 NF_SS_TRACE=1";; *_AUG_*) echo "NF_SS_AUG=1";; *_TB_*) echo "NF_HR_TB=1";; *) echo "";; esac; }   # [11-01 18:10] *_TB_* = 사후 ④ 정본 일치 심박 라벨(패치 40)
prune() { find $PFA/runs/finetune_$1/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; }
# ---------- zero-shot 셀(레인 A*) ----------
train_cell() { local M=$1 SEED=$2
  [ -f "$(ckpt $M)" ] && { echo "skip train $M"; return; }
  python scripts/fp_yaml.py train $M $TPLN $FPLR > /dev/null
  echo "$(ts) TRAIN $M start (seed $SEED, lr $FPLR, pool $TPLN, env: '$(cell_env $M)')"
  wait_ram 14; ( export $NF_ENV $(cell_env $M); RPPG_SEED=$SEED run_gated $GATE $PFA/yaml_$M/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
  [ -f "$(ckpt $M)" ] || fail "${M}_TRAIN_FAIL"; prune $M
  grep -aq "^NF stats epoch 0" /tmp/ft_$M.log || fail "${M}_NF_INACTIVE"
  grep -a "NF stats epoch 9" /tmp/ft_$M.log | grep -q "gate=[1-9]" || fail "${M}_NF_GATE0"; grep -a "NF stats epoch 0" /tmp/ft_$M.log | grep -q "ngt_hit=[1-9]" || fail "${M}_NGT0"
  case $M in *_LOC_*) grep -a "NF-SS stats epoch 9" /tmp/ft_$M.log | grep -q -E "loc=1 mloc=(0\.[0-9]*[1-9]|[1-9])" || fail "${M}_LOC_INACTIVE";; esac
  case $M in *_LOC_*|*_AUGTR_*) grep -a "NF-SS stats epoch 9" /tmp/ft_$M.log | grep -q -E "excl=(0\.[0-9]*[1-9]|[1-9])" || fail "${M}_EXCL0";; esac
  case $M in *_AUG_*|*_AUGTR_*) grep -a "NF-AUG stats epoch 9" /tmp/ft_$M.log | grep -q -E "inv=(0\.[0-9]*[1-9]|[1-9])" || fail "${M}_AUG_INACTIVE";; esac
  case $M in *_TB_*) grep -a "\[nf\] active" /tmp/ft_$M.log | grep -q "hr_tb=1" || fail "${M}_HRTB_INACTIVE";; esac
  grep -a "NAN_SKIPPED" /tmp/ft_$M.log | tail -n 1; grep -a -E "NF stats epoch 9|NF-SS stats epoch 9|NF-AUG stats epoch 9" /tmp/ft_$M.log; cp /tmp/ft_$M.log $RES/chain73_ft_$M.log; }
eval_axes() { local M=$1
  for f in $PFA/yaml_$TPLN/ev_${TPLN}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${TPLN}_//"); case $A in MMPD_B*|VIPL) continue;; esac
    [ -d $PFA/runs/ev_${M}_$A ] && continue
    EXP=$(grep -m1 "EXP_DATA_NAME" "$f" | awk '{print $2}'); pre=1; [ -d $PFA/cache/fp_$EXP ] && pre=0
    python scripts/fp_yaml.py ev $M $TPLN $A --preprocess $pre > /dev/null
    ( export $NF_ENV; run_gated 3000 $PFA/yaml_$M/ev_${M}_$A.yaml /tmp/ev_${M}_$A.log ${M}_$A ); done; }
eval_mmpd_shared() { local M=$1 b CACHE y   # 레인 공유 float 캐시 fp_mmpd_cand_B$b: 묶음별 flock 안에서 (없으면 이 모델이 전처리하며 평가) / (있으면 재사용)
  for b in 1 2 3 4 5 6; do [ -d $PFA/runs/ev_${M}_MMPD_B$b ] && continue; CACHE=fp_mmpd_cand_B$b
    ( flock 8; have=0; [ -f $LISTS/${CACHE}_0.0_1.0.csv ] && [ -d $PFA/cache/$CACHE ] && [ "$(ls $PFA/cache/$CACHE | wc -l)" -gt 100 ] && have=1
      [ $have = 0 ] && { wait_ram 16; wait_load 40; }
      python scripts/fp_yaml.py ev $M $TPLN MMPD_B$b --preprocess 1 > /dev/null; y=$PFA/yaml_$M/ev_${M}_MMPD_B$b.yaml
      sed -i "s#EXP_DATA_NAME: fp_mmpd_[a-z0-9]*_B$b#EXP_DATA_NAME: $CACHE#" $y; [ $have = 1 ] && sed -i "s/DO_PREPROCESS: true/DO_PREPROCESS: false/" $y
      ( export $NF_ENV; run_gated 3000 $y /tmp/ev_${M}_MMPD_B$b.log ${M}_MMPD_B$b ) ) 8>/tmp/fp_mmpd_cand_B$b.lock; done; }
eval_vipl() { local M=$1; [ -f $RES/agg_${M}_VIPL.json ] && return
  python scripts/vipl_yaml.py ev $M "$(ckpt $M)" --template $PFA/yaml_$M/ev_${M}_DLCN.yaml > /dev/null
  ( export $NF_ENV RPPG_CACHE_U8=1; run_gated 3000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL )
  python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; }
agg_cell() { local M=$1; python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; python scripts/agg_mmpd.py $M >> /tmp/agg_$M.log 2>&1; echo "$(ts) AGG_${M}_DONE $(grep -E 'RTNI|NIRP_LARGE' /tmp/agg_$M.log | head -2 | tr '\n' ' ' | cut -c1-120)"; }
zs_lane() { local base=$1 S sfx M
  for S in 100 200 300; do sfx=""; [ $S != 100 ] && sfx=_s$S; M=${base}$sfx
    train_cell $M $S; eval_axes $M; eval_vipl $M; eval_mmpd_shared $M; agg_cell $M; echo "$(ts) [$LANE] $M DONE"; done; }
# ---------- in-domain 2단계 셀(레인 B·C·D) ----------
fold_train() { # fold_train NAME TRAIN_CSV TEST_CSV FOLD RESUME_CKPT SEED EXTRA_ENV ("" 가능)
  local M=$1 TR=$2 TE=$3 K=$4 RCK=$5 SEED=$6 XENV=$7; local ck=$(ckpt $M)
  [ -f "$ck" ] && { echo "skip train $M"; return; }
  [ -f "$RCK" ] || fail "${M}_RESUME_MISSING $RCK"; [ -f "$TR" ] && [ -f "$TE" ] || fail "${M}_LIST_MISSING"
  python scripts/vipl_yaml.py train $M $K "$TR" 10 --test_csv "$TE" --template $NFYAML --resume "$RCK" --lr 5e-5 > /dev/null
  grep -q "RESUME: $RCK" $PFA/yaml_VIPL/finetune_$M.yaml && grep -q "NAME: FactorizePhys" $PFA/yaml_VIPL/finetune_$M.yaml || fail "${M}_YAML"
  echo "$(ts) TRAIN $M start (fold $K, 2-stage from $(basename $RCK), seed $SEED, lr 5e-5, env '$XENV')"
  wait_ram 14; ( export $NF_ENV $XENV RPPG_CACHE_U8=1; RPPG_SEED=$SEED run_gated $GATE $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
  [ -f "$ck" ] || fail "${M}_TRAIN_FAIL"; prune $M
  grep -aq "^NF stats epoch 0" /tmp/ft_$M.log || fail "${M}_NF_INACTIVE"; case "$XENV" in *NF_HR_TB=1*) grep -a "\[nf\] active" /tmp/ft_$M.log | grep -q "hr_tb=1" || fail "${M}_HRTB_INACTIVE";; esac
  grep -a "NAN_SKIPPED" /tmp/ft_$M.log | tail -n 1; cp /tmp/ft_$M.log $RES/chain73_ft_$M.log; }
lane_B() { local K T
  for K in 1 2 3 4 5; do
    fold_train VF${K}_NFs2_L $LISTS/vipl_train$K.csv $LISTS/vipl_test$K.csv $K "$(ckpt NF_HYB_L_s200)" 200 "RPPG_PEARSON_SKIP_RE=^p[0-9]+_v[0-9] RPPG_HR_LABEL_FILE=1"
    grep -a "NF stats epoch 0" /tmp/ft_VF${K}_NFs2_L.log | grep -q "hr_sidecar=[1-9]" || fail "VF${K}_NFs2_L_HRFILE"
    fold_train VF${K}_NFs3_L $LISTS/vipl_train$K.csv $LISTS/vipl_test$K.csv $K "$(ckpt NF_HYB_L_s300)" 300 "RPPG_PEARSON_SKIP_RE=^p[0-9]+_v[0-9] RPPG_HR_LABEL_FILE=1"
    fold_train VF${K}_NFSS_L $LISTS/vipl_train$K.csv $LISTS/vipl_test$K.csv $K "$(ckpt NF_SS_TR_L)" 100 "RPPG_PEARSON_SKIP_RE=^p[0-9]+_v[0-9] RPPG_HR_LABEL_FILE=1 NF_SS_TRACE=1"
    grep -aq "^NF-SS stats epoch 0" /tmp/ft_VF${K}_NFSS_L.log || fail "VF${K}_NFSS_L_NFSS_INACTIVE"
    for T in NFs2_L NFs3_L NFSS_L; do python scripts/agg_vipl.py VF${K}_$T 2>&1 | grep -E "all \[문헌" | sed "s/^/  VF${K}_$T /"; done; done
  for T in NFs2_L NFs3_L NFSS_L; do python scripts/agg_vipl.py --cv $T 2>&1 | grep -E "all \[문헌" | sed "s/^/  CV_$T /"; done; }
lane_B2() { local K; for K in 1 2 3 4 5; do
    fold_train VF${K}_NFs2_L $LISTS/vipl_train$K.csv $LISTS/vipl_test$K.csv $K "$(ckpt NF_HYB_L_s200)" 200 "RPPG_PEARSON_SKIP_RE=^p[0-9]+_v[0-9] RPPG_HR_LABEL_FILE=1"
    grep -a "NF stats epoch 0" /tmp/ft_VF${K}_NFs2_L.log | grep -q "hr_sidecar=[1-9]" || fail "VF${K}_NFs2_L_HRFILE"
    python scripts/agg_vipl.py VF${K}_NFs2_L 2>&1 | grep -E "all \[문헌" | sed "s/^/  VF${K}_NFs2_L /"; done
  python scripts/agg_vipl.py --cv NFs2_L 2>&1 | grep -E "all \[문헌" | sed "s/^/  CV_NFs2_L /"; }
lane_B3() { local K; for K in 1 2 3 4 5; do
    fold_train VF${K}_NFs3_L $LISTS/vipl_train$K.csv $LISTS/vipl_test$K.csv $K "$(ckpt NF_HYB_L_s300)" 300 "RPPG_PEARSON_SKIP_RE=^p[0-9]+_v[0-9] RPPG_HR_LABEL_FILE=1"
    grep -a "NF stats epoch 0" /tmp/ft_VF${K}_NFs3_L.log | grep -q "hr_sidecar=[1-9]" || fail "VF${K}_NFs3_L_HRFILE"
    python scripts/agg_vipl.py VF${K}_NFs3_L 2>&1 | grep -E "all \[문헌" | sed "s/^/  VF${K}_NFs3_L /"; done
  python scripts/agg_vipl.py --cv NFs3_L 2>&1 | grep -E "all \[문헌" | sed "s/^/  CV_NFs3_L /"; }
lane_B4() { local K; [ -f "$(ckpt NF_SS_TR_L)" ] || fail "NF_SS_TR_L_CKPT"; for K in 1 2 3 4 5; do
    fold_train VF${K}_NFSS_L $LISTS/vipl_train$K.csv $LISTS/vipl_test$K.csv $K "$(ckpt NF_SS_TR_L)" 100 "RPPG_PEARSON_SKIP_RE=^p[0-9]+_v[0-9] RPPG_HR_LABEL_FILE=1 NF_SS_TRACE=1"
    grep -aq "^NF-SS stats epoch 0" /tmp/ft_VF${K}_NFSS_L.log || fail "VF${K}_NFSS_L_NFSS_INACTIVE"
    python scripts/agg_vipl.py VF${K}_NFSS_L 2>&1 | grep -E "all \[문헌" | sed "s/^/  VF${K}_NFSS_L /"; done
  python scripts/agg_vipl.py --cv NFSS_L 2>&1 | grep -E "all \[문헌" | sed "s/^/  CV_NFSS_L /"; }
lane_C() { local K S T
  if [ ! -d $PFA/cache/mmpd_all_u8 ] || [ "$(ls $PFA/cache/mmpd_all_u8 | wc -l)" -lt 15000 ]; then
    echo "$(ts) MMPD u8 cache build start (660 clips, 8 workers)"; wait_ram 20; nice -n 10 python scripts/nirp_pool_stream.py $PFA/COTRAIN_MMPD_ALL $PFA/cache/mmpd_all_u8 --workers 8 > /tmp/mmpd_u8_build.log 2>&1
    echo "$(ts) MMPD u8 cache files=$(ls $PFA/cache/mmpd_all_u8 | wc -l) $(grep -c '^ok' /tmp/mmpd_u8_build.log) ok"; fi
  python - <<'EOF' || fail "MMPD_LISTS"
import pandas as pd, os, subprocess
L = "/SPECPHYS_ROOT/_workspace/data/physformer/cache/DataFileLists"; bad = 0
for k in range(1, 6):
    for nm in (f"mmpd_train{k}.csv", f"mmpd_test{k}.csv"):
        df = pd.read_csv(f"{L}/{nm}"); miss = [p for p in df["input_files"] if not os.path.exists(p)]; bad += len(miss)
        print(nm, len(df), "missing", len(miss))
if bad:
    print("regenerating lists from cache"); subprocess.run(["python", "scripts/mmpd_filelist.py", "folds"], check=True)
    for k in range(1, 6): subprocess.run(["python", "scripts/mmpd_filelist.py", "test", str(k)], check=True); subprocess.run(["python", "scripts/mmpd_filelist.py", "train", str(k)], check=True)
EOF
  for S in 100; do case $S in 100) T=NF_L; R0=NF_HYB_L;; 200) T=NFs2_L; R0=NF_HYB_L_s200;; 300) T=NFs3_L; R0=NF_HYB_L_s300;; esac   # [11-01 18:10] 시드 200/300 은 C2/C3; 라벨 = 패치 40(u8 복원 + NF_HR_TB)
    for K in 1 2 3 4 5; do fold_train MF${K}_$T $LISTS/mmpd_train$K.csv $LISTS/mmpd_test$K.csv $K "$(ckpt $R0)" $S "NF_HR_TB=1"; python scripts/agg_mmpd_cv2.py MF${K}_$T 2>&1 | grep -E "^all|Walking" | head -2 | sed "s/^/  MF${K}_$T /"; done
    python scripts/agg_mmpd_cv2.py --cv $T 2>&1 | grep -E "all |Walking|non-stationary" | head -4 | sed "s/^/  MCV5_$T /"; done; }
lane_D() { local K S T M A
  if ! python scripts/nirp_cache_check.py > /tmp/nirp_cache_check.log 2>&1; then
    echo "$(ts) NIRP u8 cache build start (184 clips, 8 workers)"; wait_ram 20; nice -n 10 python scripts/nirp_pool_stream.py $PFA/NIRP_TRAIN $PFA/cache/nirp_train_u8 --workers 8 > /tmp/nirp_u8_build.log 2>&1
    python scripts/nirp_cache_check.py > /tmp/nirp_cache_check.log 2>&1 || { tail -n 3 /tmp/nirp_cache_check.log; fail "NIRP_CACHE_INCOMPLETE"; }; fi
  python scripts/nirp_cache_check.py --csv | tail -n 1; python scripts/nirp_fold_lists.py | tail -n 5
  for S in 100; do case $S in 100) T=NF_L; R0=NF_HYB_L;; 200) T=NFs2_L; R0=NF_HYB_L_s200;; 300) T=NFs3_L; R0=NF_HYB_L_s300;; esac   # [11-01 18:10] 시드 200/300 은 D2/D3; 라벨 = 패치 40
    for K in 1 2 3 4 5; do M=NR${K}_$T; fold_train $M $LISTS/nirp_only_train$K.csv $LISTS/nirp_only_test$K.csv $K "$(ckpt $R0)" $S "NF_HR_TB=1"
      for A in NIRP_STILL NIRP_SMALL NIRP_LARGE; do [ -d $PFA/runs/ev_${M}_$A ] && continue; python scripts/fp_yaml.py ev $M $TPLN $A --preprocess 0 > /dev/null; ( export $NF_ENV; run_gated 3000 $PFA/yaml_$M/ev_${M}_$A.yaml /tmp/ev_${M}_$A.log ${M}_$A ); done
      python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; grep -E "NIRP_LARGE" /tmp/agg_$M.log | head -1 | sed "s/^/  $M /"; done
    python scripts/agg_fold_axes.py --folds nirp_folds.json --pattern "NR{k}_$T" --tag $T --out NRCV | tail -n 4; done
  [ -f $RES/agg_NRCV_HYBN_L.json ] || python scripts/agg_fold_axes.py --folds nirp_folds.json --pattern "NF{k}_HYBN_L" --tag HYBN_L --out NRCV | tail -n 1; }
lane_Cs() { local S=$1 T R0 K; case $S in 200) T=NFs2_L; R0=NF_HYB_L_s200;; 300) T=NFs3_L; R0=NF_HYB_L_s300;; *) fail "SEED";; esac   # [11-01 13:50 분할] C 의 시드 200/300 을 레인으로
  [ -d $PFA/cache/mmpd_all_u8 ] && [ "$(ls $PFA/cache/mmpd_all_u8 | wc -l)" -ge 15000 ] || fail "MMPD_U8_MISSING"
  for K in 1 2 3 4 5; do fold_train MF${K}_$T $LISTS/mmpd_train$K.csv $LISTS/mmpd_test$K.csv $K "$(ckpt $R0)" $S "NF_HR_TB=1"; python scripts/agg_mmpd_cv2.py MF${K}_$T 2>&1 | grep -E "^all|Walking" | head -2 | sed "s/^/  MF${K}_$T /"; done
  python scripts/agg_mmpd_cv2.py --cv NF_L $T 2>&1 | grep -E "^---|all |Walking" | head -8; }
lane_Ds() { local S=$1 T R0 K M A; case $S in 200) T=NFs2_L; R0=NF_HYB_L_s200;; 300) T=NFs3_L; R0=NF_HYB_L_s300;; *) fail "SEED";; esac   # [11-01 13:50 분할] D 의 시드 200/300 을 레인으로
  python scripts/nirp_cache_check.py > /dev/null 2>&1 || fail "NIRP_U8_MISSING"; [ -f $LISTS/nirp_only_train1.csv ] || python scripts/nirp_fold_lists.py > /dev/null
  for K in 1 2 3 4 5; do M=NR${K}_$T; fold_train $M $LISTS/nirp_only_train$K.csv $LISTS/nirp_only_test$K.csv $K "$(ckpt $R0)" $S "NF_HR_TB=1"
    for A in NIRP_STILL NIRP_SMALL NIRP_LARGE; do [ -d $PFA/runs/ev_${M}_$A ] && continue; python scripts/fp_yaml.py ev $M $TPLN $A --preprocess 0 > /dev/null; ( export $NF_ENV; run_gated 3000 $PFA/yaml_$M/ev_${M}_$A.yaml /tmp/ev_${M}_$A.log ${M}_$A ); done
    python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; grep -E "NIRP_LARGE" /tmp/agg_$M.log | head -1 | sed "s/^/  $M /"; done
  python scripts/agg_fold_axes.py --folds nirp_folds.json --pattern "NR{k}_NF_L" --tag NF_L --out NRCV | tail -n 1; python scripts/agg_fold_axes.py --folds nirp_folds.json --pattern "NR{k}_$T" --tag $T --out NRCV | tail -n 4; }
lane_Z() { local L
  for L in A1 A2 A3 B2 B3 B4 C C2 C3 D D2 D3 T1; do until [ -f /tmp/chain73_$L.done ] || [ -f /tmp/chain73_$L.fail ]; do sleep 300; done; [ -f /tmp/chain73_$L.fail ] && echo "$(ts) WARN lane $L failed — 판정은 가능한 결과로"; done
  python scripts/candidates_judge.py | tee /tmp/candidates_judge.txt | grep -E "^===|H-C|가드|lock-in" | head -n 80
  for b in 1 2 3 4 5 6; do drop_cache fp_mmpd_cand_B$b; done; }

echo "$(ts) CHAIN73 $LANE START (quota ${RPPG_MP_QUOTA:-?}, workers ${RPPG_NUM_WORKERS:-?}, host $(hostname))"
[ -d $PFA/COTRAIN_ALLF_HYB4 ] && [ "$(ls $PFA/COTRAIN_ALLF_HYB4 | wc -l)" = 366 ] || fail "POOL_HYB4"
[ "$(ls $PFA/noise_gt_hyb2 | wc -l)" = 48 ] || fail "NOISE_GT"
for S in "" _s200 _s300; do [ -f "$(ckpt NF_HYB_L$S)" ] || fail "BASE_CKPT_MISSING NF_HYB_L$S"; done
case $LANE in
  A1) zs_lane NF_SS_LOC_L;; A2) zs_lane NF_SS_AUG_L;; A3) zs_lane NF_SS_AUGTR_L;;
  B) [ -f "$(ckpt NF_SS_TR_L)" ] || fail "NF_SS_TR_L_CKPT"; lane_B;; B2) lane_B2;; B3) lane_B3;; B4) lane_B4;; C) lane_C;; C2) lane_Cs 200;; C3) lane_Cs 300;; D) lane_D;; D2) lane_Ds 200;; D3) lane_Ds 300;; T1) zs_lane NF_HYB_TB_L;; Z) lane_Z;;
  *) fail "UNKNOWN_LANE";; esac
echo "$(ts) CHAIN73 $LANE DONE"; touch /tmp/chain73_$LANE.done
