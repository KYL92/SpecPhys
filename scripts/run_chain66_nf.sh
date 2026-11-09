#!/bin/bash
# 체인 66 (48 문서, 2026-10-27): NF-Phys(FactorizePhys 트렁크 + rank-2 잡음 인수분해 FSAM, 툴박스 패치 37) Phase 2 zero-shot + Phase 3 VIPL in-domain 2단계.
#  Phase 2: 하이브리드 풀 COTRAIN_ALLF_HYB4(366) + 잡음 정답 noise_gt_hyb2(DN 48) → NF_HYB_L(시드 100 → 200·300),
#           절제 NF_HYB_NOSUP_L(교란 감독·탈상관 없음)·NF_HYB_R1_L(rank-1 + 시간 배치 + 스펙트럼 손실) 시드 100.
#           평가: 13축(float fp_real_* 캐시) + MMPD float 6묶음(54b 방식, 묶음마다 캐시 삭제) + VIPL u8 zero-shot.
#  Phase 3: NF_HYB_L(시드 100) → VIPL fold 별 2단계 적응 VF{k}_NF_L(장치 HR 사이드카 스펙트럼 손실, 파형 손실 제외, lr 5e-5, 10 ep, 마지막 epoch) → agg --cv NF_L.
#  순서: Phase 2 시드 100 3셀(학습 → 평가) → Phase 3 → Phase 2 시드 200·300 → nf_judge. 로그 /tmp/chain66.log, 마커 /tmp/chain66.done.
#  기동: sg video -c "RPPG_MP_QUOTA=3 RPPG_NUM_WORKERS=4 bash scripts/run_chain66_nf.sh" (GPU 장치가 video 그룹 전용, 10-27).
source /SPECPHYS_ROOT/scripts/chain_lib.sh
exec 9>/tmp/chain66.lock; flock -n 9 || { echo "chain66 already running (lock) -> exit"; exit 0; }   # 2중 기동 방지(10-27 사고)
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; LISTS=$PFA/cache/DataFileLists; FPLR=1e-4; TPLN=CoTAM_HYB4_L
NF_ENV="RPPG_NF=1 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) $1 -> abort"; touch /tmp/chain66.fail; exit 1; }
ckpt() { echo "$PFA/runs/finetune_$1/PreTrainedModels/SYNTH_${1}_factorizephys_Epoch9.pth"; }
wait_ram() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$1" ]; do echo "$(ts) RAM_WAIT avail<$1G"; sleep 120; done; }
wait_load() { while [ "$(cut -d. -f1 /proc/loadavg)" -gt "$1" ]; do echo "$(ts) LOAD_WAIT 1min>$1"; sleep 300; done; }
cell_env() { case $1 in *NOSUP*) echo "NF_SUP=0";; *_R1_*) echo "NF_RANK1=1";; *) echo "";; esac; }
train_nf() { local M=$1 SEED=$2
  [ -f "$(ckpt $M)" ] && { echo "skip train $M"; return; }
  python scripts/fp_yaml.py train $M $TPLN $FPLR > /dev/null
  echo "$(ts) TRAIN $M start (seed $SEED, lr $FPLR, pool $TPLN, ablation env: '$(cell_env $M)')"
  wait_ram 14; ( export $NF_ENV $(cell_env $M); RPPG_SEED=$SEED run_gated $GATE $PFA/yaml_$M/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
  [ -f "$(ckpt $M)" ] || fail "${M}_TRAIN_FAIL"
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
  grep -aq "^NF stats epoch 0" /tmp/ft_$M.log || fail "${M}_NF_INACTIVE"
  case $M in *NOSUP*|*_R1_*) ;; *) grep -a "NF stats epoch 9" /tmp/ft_$M.log | grep -q "gate=[1-9]" || fail "${M}_NF_GATE0"; grep -a "NF stats epoch 0" /tmp/ft_$M.log | grep -q "ngt_hit=[1-9]" || fail "${M}_NGT0";; esac
  grep -a "NF stats epoch" /tmp/ft_$M.log | tail -n 1; cp /tmp/ft_$M.log $RES/chain66_ft_$M.log; }
eval_axes() { local M=$1
  for f in $PFA/yaml_$TPLN/ev_${TPLN}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${TPLN}_//"); case $A in MMPD_B*|VIPL) continue;; esac
    [ -d $PFA/runs/ev_${M}_$A ] && continue
    EXP=$(grep -m1 "EXP_DATA_NAME" "$f" | awk '{print $2}'); pre=1; [ -d $PFA/cache/fp_$EXP ] && pre=0
    python scripts/fp_yaml.py ev $M $TPLN $A --preprocess $pre > /dev/null
    ( export $NF_ENV $(cell_env $M); run_gated 3000 $PFA/yaml_$M/ev_${M}_$A.yaml /tmp/ev_${M}_$A.log ${M}_$A ); done; }
eval_mmpd_float() { local b M CACHE fpdone   # 54b 방식: 묶음마다 float 전처리(워커 quota) → 모든 셀 평가 → 캐시 삭제
  for b in 1 2 3 4 5 6; do CACHE=fp_mmpd_nf_B$b; fpdone=0; [ -f $PFA/cache/DataFileLists/${CACHE}_0.0_1.0.csv ] && [ -d $PFA/cache/$CACHE ] && fpdone=1
    for M in "$@"; do [ -d $PFA/runs/ev_${M}_MMPD_B$b ] && continue
      [ $fpdone = 0 ] && { wait_ram 16; wait_load 30; }
      python scripts/fp_yaml.py ev $M $TPLN MMPD_B$b --preprocess $((1-fpdone)) > /dev/null; y=$PFA/yaml_$M/ev_${M}_MMPD_B$b.yaml
      sed -i "s#EXP_DATA_NAME: fp_mmpd_[a-z0-9]*_B$b#EXP_DATA_NAME: $CACHE#" $y; [ $fpdone = 1 ] && sed -i "s/DO_PREPROCESS: true/DO_PREPROCESS: false/" $y
      ( export $NF_ENV $(cell_env $M); run_gated 3000 $y /tmp/ev_${M}_MMPD_B$b.log ${M}_MMPD_B$b ); fpdone=1; done
    drop_cache $CACHE; done; }
eval_vipl() { local M=$1; [ -f $RES/agg_${M}_VIPL.json ] && return
  python scripts/vipl_yaml.py ev $M "$(ckpt $M)" --template $PFA/yaml_$M/ev_${M}_DLCN.yaml > /dev/null
  ( export $NF_ENV $(cell_env $M) RPPG_CACHE_U8=1; run_gated 3000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL )
  python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; }
agg_cell() { local M=$1; python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; python scripts/agg_mmpd.py $M >> /tmp/agg_$M.log 2>&1; echo "$(ts) AGG_${M}_DONE $(grep -E 'RTNI|NIRP_SMALL' /tmp/agg_$M.log | head -2 | tr '\n' ' ' | cut -c1-120)"; }
vcell_nf() { local K=$1; local M=VF${K}_NF_L; local ck=$(ckpt $M)
  if [ ! -f $ck ]; then
    python scripts/vipl_yaml.py train $M $K $LISTS/vipl_train$K.csv 10 --template $PFA/yaml_NF_HYB_L/finetune_NF_HYB_L.yaml --resume "$(ckpt NF_HYB_L)" --lr 5e-5 > /dev/null
    grep -q "RESUME: $(ckpt NF_HYB_L)" $PFA/yaml_VIPL/finetune_$M.yaml && grep -q "NAME: FactorizePhys" $PFA/yaml_VIPL/finetune_$M.yaml || fail "${M}_YAML"
    echo "$(ts) TRAIN $M start (fold $K, 2-stage from NF_HYB_L, lr 5e-5, device-HR spectral loss, waveform loss skipped)"
    ( export $NF_ENV RPPG_CACHE_U8=1 RPPG_PEARSON_SKIP_RE='^p[0-9]+_v[0-9]' RPPG_HR_LABEL_FILE=1; RPPG_SEED=100 run_gated $GATE $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
    [ -f $ck ] || fail "${M}_TRAIN_FAIL"
    grep -a "NF stats epoch 0" /tmp/ft_$M.log | grep -q "hr_sidecar=[1-9]" || fail "${M}_HRFILE_FAIL"
    grep -a "NF stats epoch 0" /tmp/ft_$M.log | grep -q "skip=[1-9]" || fail "${M}_SKIP_FAIL"
    find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; cp /tmp/ft_$M.log $RES/chain66_ft_$M.log; fi
  python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; echo "$(ts) CHAIN66 $M DONE"; }

echo "$(ts) CHAIN66 START (quota ${RPPG_MP_QUOTA:-?}, workers ${RPPG_NUM_WORKERS:-?})"
[ -d $PFA/COTRAIN_ALLF_HYB4 ] && [ "$(ls $PFA/COTRAIN_ALLF_HYB4 | wc -l)" = 366 ] || fail "POOL_HYB4"
[ "$(ls $PFA/noise_gt_hyb2 | wc -l)" = 48 ] || fail "NOISE_GT"
# Phase 2, 시드 100
P2="NF_HYB_L NF_HYB_NOSUP_L NF_HYB_R1_L"
for M in $P2; do train_nf $M 100; done
for M in $P2; do eval_axes $M; done
for M in $P2; do eval_vipl $M; done
eval_mmpd_float $P2
for M in $P2; do agg_cell $M; done
python scripts/nf_judge.py --phase2 2>&1 | grep -E "^===|H-B" | head -n 20
# Phase 3: VIPL 5-fold 2단계
for K in 1 2 3 4 5; do vcell_nf $K; done
python scripts/agg_vipl.py --cv NF_L 2>&1 | grep -E "all \[문헌"
python scripts/nf_judge.py --phase3 2>&1 | grep -E "^===|H-B" | head -n 12
# Phase 2, 시드 200·300 (NF_HYB_L 만)
for S in 200 300; do train_nf NF_HYB_L_s$S $S; done
for S in 200 300; do eval_axes NF_HYB_L_s$S; eval_vipl NF_HYB_L_s$S; done
eval_mmpd_float NF_HYB_L_s200 NF_HYB_L_s300
for S in 200 300; do agg_cell NF_HYB_L_s$S; done
drop_cache fp_cotrain_allf_hyb4
python scripts/nf_judge.py | tee /tmp/nf_judge.txt | grep -E "^===|H-B" | head -n 40
echo "$(ts) CHAIN66 DONE"; touch /tmp/chain66.done
