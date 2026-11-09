#!/bin/bash
# 체인 72 (53 문서, 2026-10-31; RTX PRO 6000 머신에서 실행): 자기 지도 분리 신호 연구 — 교란 정답이 없는 실 클립에서의 rPPG–교란 분리(툴박스 패치 38).
#  기준 NF_HYB_L(스펙트럼 감독 FactorizePhys + 하이브리드 풀 366, 3시드 기존) 레시피에 학습 신호만 추가:
#    NF_SS_TR_L   = 전역 trace 자기 감독(출력 스펙트럼 배제 + 교란 슬롯 스펙트럼 표적, NF_SS_TRACE=1)
#    NF_SS_AT_L   = 어텐션 공간 정칙화(피부 사전 + 교란 영역 억제 + 슬롯 공간 분리, NF_SS_ATT=1)
#    NF_SS_BOTH_L = 둘 다(1차 셀)
#  각 시드 100·200·300. 평가: 13축(float fp_real_* 캐시) + MMPD float 6묶음(묶음마다 캐시 삭제) + VIPL u8 zero-shot. 판정 scripts/selfsup_judge.py(H-L1~L4).
#  순서: 시드 100 3셀(학습 → 13축 → VIPL → MMPD → 집계 → 중간 판정) → 시드 200·300 → 최종 판정. 로그 /tmp/chain72.log, 마커 /tmp/chain72.done. flock.
#  기동(PRO 6000, 46 GB 이상 RAM 가정): RPPG_MP_QUOTA=6 RPPG_NUM_WORKERS=6 nohup bash scripts/run_chain72_selfsup.sh >> /tmp/chain72.log 2>&1 &
#  전제: 54 문서 이관 완료(verify_pro6000.sh 통과), results/selfsup_nuis_peaks.json(Phase 0 피크 캐시) 존재.
#  pro6000 에서는 컨테이너 안에서 기동: docker exec -d vface_train bash -lc 'cd /SPECPHYS_ROOT && RPPG_MP_QUOTA=6 RPPG_NUM_WORKERS=6 nohup bash scripts/run_chain72_selfsup.sh >> /tmp/chain72.log 2>&1 &'  (/tmp = 호스트 /SPECPHYS_TMP/container)
source /SPECPHYS_ROOT/scripts/chain_lib.sh
exec 9>/tmp/chain72.lock; flock -n 9 || { echo "chain72 already running (lock) -> exit"; exit 0; }
cd $ROOT; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results; FPLR=1e-4; TPLN=CoTAM_HYB4_L
NF_ENV="RPPG_NF=1 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) $1 -> abort"; touch /tmp/chain72.fail; exit 1; }
ckpt() { echo "$PFA/runs/finetune_$1/PreTrainedModels/SYNTH_${1}_factorizephys_Epoch9.pth"; }
wait_ram() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$1" ]; do echo "$(ts) RAM_WAIT avail<$1G"; sleep 120; done; }
wait_load() { while [ "$(cut -d. -f1 /proc/loadavg)" -gt "$1" ]; do echo "$(ts) LOAD_WAIT 1min>$1"; sleep 300; done; }
cell_env() { case $1 in *_TR_*) echo "NF_SS_TRACE=1";; *_AT_*) echo "NF_SS_ATT=1";; *_BOTH_*) echo "NF_SS_TRACE=1 NF_SS_ATT=1";; *) echo "";; esac; }
train_cell() { local M=$1 SEED=$2
  [ -f "$(ckpt $M)" ] && { echo "skip train $M"; return; }
  python scripts/fp_yaml.py train $M $TPLN $FPLR > /dev/null
  echo "$(ts) TRAIN $M start (seed $SEED, lr $FPLR, pool $TPLN, signal env: '$(cell_env $M)')"
  wait_ram 14; ( export $NF_ENV $(cell_env $M); RPPG_SEED=$SEED run_gated $GATE $PFA/yaml_$M/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
  [ -f "$(ckpt $M)" ] || fail "${M}_TRAIN_FAIL"
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
  grep -aq "^NF stats epoch 0" /tmp/ft_$M.log || fail "${M}_NF_INACTIVE"
  grep -aq "^NF-SS stats epoch 0" /tmp/ft_$M.log || fail "${M}_NFSS_INACTIVE"
  grep -a "NF stats epoch 9" /tmp/ft_$M.log | grep -q "gate=[1-9]" || fail "${M}_NF_GATE0"; grep -a "NF stats epoch 0" /tmp/ft_$M.log | grep -q "ngt_hit=[1-9]" || fail "${M}_NGT0"
  case $M in *_TR_*|*_BOTH_*) grep -a "NF-SS stats epoch 9" /tmp/ft_$M.log | grep -q -E "excl=(0\.[0-9]*[1-9]|[1-9])" || fail "${M}_EXCL0";; esac
  case $M in *_AT_*|*_BOTH_*) grep -a "NF-SS stats epoch 9" /tmp/ft_$M.log | grep -q -E "valid=[1-9]" || fail "${M}_SKIN_INVALID";; esac
  grep -a "NAN_SKIPPED" /tmp/ft_$M.log | tail -n 1
  grep -a -E "NF stats epoch 9|NF-SS stats epoch 9" /tmp/ft_$M.log; cp /tmp/ft_$M.log $RES/chain72_ft_$M.log; }
eval_axes() { local M=$1
  for f in $PFA/yaml_$TPLN/ev_${TPLN}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${TPLN}_//"); case $A in MMPD_B*|VIPL) continue;; esac
    [ -d $PFA/runs/ev_${M}_$A ] && continue
    EXP=$(grep -m1 "EXP_DATA_NAME" "$f" | awk '{print $2}'); pre=1; [ -d $PFA/cache/fp_$EXP ] && pre=0
    python scripts/fp_yaml.py ev $M $TPLN $A --preprocess $pre > /dev/null
    ( export $NF_ENV; run_gated 3000 $PFA/yaml_$M/ev_${M}_$A.yaml /tmp/ev_${M}_$A.log ${M}_$A ); done; }
eval_mmpd_float() { local b M CACHE fpdone   # 묶음마다 float 전처리(워커 quota) → 모든 셀 평가 → 캐시 삭제
  for b in 1 2 3 4 5 6; do CACHE=fp_mmpd_ss_B$b; fpdone=0; [ -f $PFA/cache/DataFileLists/${CACHE}_0.0_1.0.csv ] && [ -d $PFA/cache/$CACHE ] && fpdone=1
    for M in "$@"; do [ -d $PFA/runs/ev_${M}_MMPD_B$b ] && continue
      [ $fpdone = 0 ] && { wait_ram 16; wait_load 30; }
      python scripts/fp_yaml.py ev $M $TPLN MMPD_B$b --preprocess $((1-fpdone)) > /dev/null; y=$PFA/yaml_$M/ev_${M}_MMPD_B$b.yaml
      sed -i "s#EXP_DATA_NAME: fp_mmpd_[a-z0-9]*_B$b#EXP_DATA_NAME: $CACHE#" $y; [ $fpdone = 1 ] && sed -i "s/DO_PREPROCESS: true/DO_PREPROCESS: false/" $y
      ( export $NF_ENV; run_gated 3000 $y /tmp/ev_${M}_MMPD_B$b.log ${M}_MMPD_B$b ); fpdone=1; done
    drop_cache $CACHE; done; }
eval_vipl() { local M=$1; [ -f $RES/agg_${M}_VIPL.json ] && return
  until [ "$(ls $PFA/cache/vipl_u8 2>/dev/null | wc -l)" -ge 35380 ] && [ ! -d $PFA/cache/.rsync-partial ]; do echo "$(ts) VIPL_CACHE_WAIT $(ls $PFA/cache/vipl_u8 2>/dev/null | wc -l)/35380"; sleep 300; done   # [11-01] pro6000: u8 캐시 pull 완료까지 대기
  python scripts/vipl_yaml.py ev $M "$(ckpt $M)" --template $PFA/yaml_$M/ev_${M}_DLCN.yaml > /dev/null
  ( export $NF_ENV RPPG_CACHE_U8=1; run_gated 3000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL )
  python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; }
agg_cell() { local M=$1; python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; python scripts/agg_mmpd.py $M >> /tmp/agg_$M.log 2>&1; echo "$(ts) AGG_${M}_DONE $(grep -E 'RTNI|NIRP_LARGE' /tmp/agg_$M.log | head -2 | tr '\n' ' ' | cut -c1-120)"; }
run_seed() { local S=$1 sfx=""; [ $S != 100 ] && sfx=_s$S; local CELLS="NF_SS_TR_L$sfx NF_SS_AT_L$sfx NF_SS_BOTH_L$sfx" M
  for M in $CELLS; do train_cell $M $S; done
  for M in $CELLS; do eval_axes $M; done
  for M in $CELLS; do eval_vipl $M; done
  eval_mmpd_float $CELLS
  for M in $CELLS; do agg_cell $M; done; }

echo "$(ts) CHAIN72 START (quota ${RPPG_MP_QUOTA:-?}, workers ${RPPG_NUM_WORKERS:-?}, host $(hostname))"
[ -d $PFA/COTRAIN_ALLF_HYB4 ] && [ "$(ls $PFA/COTRAIN_ALLF_HYB4 | wc -l)" = 366 ] || fail "POOL_HYB4"
[ "$(ls $PFA/noise_gt_hyb2 | wc -l)" = 48 ] || fail "NOISE_GT"
[ -f $RES/selfsup_nuis_peaks.json ] || fail "PHASE0_PEAKS_MISSING"
for S in 100 200 300; do [ -f "$(ckpt NF_HYB_L$([ $S = 100 ] || echo _s$S))" ] || echo "$(ts) WARN baseline ckpt NF_HYB_L seed $S missing (판정은 agg json 만 사용)"; done
run_seed 100
python scripts/selfsup_judge.py 2>&1 | grep -E "^===|H-L|가드" | head -n 30
run_seed 200; run_seed 300
python scripts/selfsup_judge.py | tee /tmp/selfsup_judge.txt | grep -E "^===|H-L|가드" | head -n 60
echo "$(ts) CHAIN72 DONE"; touch /tmp/chain72.done
