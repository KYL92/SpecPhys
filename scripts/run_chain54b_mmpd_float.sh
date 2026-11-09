#!/bin/bash
# 체인 54b (39 문서 §5 10-21): 체인 54 의 MMPD 평가를 float 전처리 경로(정본 agg_mmpd.py, MMPD_B1~6)로 다시 수행.
#  이유: uint8 캐시 경로(mmpd_all_u8 → Raw 72x72)가 FactorizePhys 에서 무효(공개 체크포인트 13.16 vs float 9.79, p=.011).
#  메모리: 전처리 워커 1개(≈4.5 GB), 각 묶음 전처리 전 가용 RAM ≥ 16 GB 대기. 체인 54 완료(/tmp/chain54.done) 후 실행. 로그 /tmp/chain54b.log, 마커 /tmp/chain54b.done.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
cd $ROOT; RES=$ROOT/_workspace/data/results
ckpt() { echo "$PFA/runs/finetune_$1/PreTrainedModels/SYNTH_${1}_factorizephys_Epoch9.pth"; }
is_hyb() { case $1 in *HYB*) return 0;; *) return 1;; esac; }
tpl() { case $1 in *HYB*) echo CoTAM_HYB4_L;; *) echo CoTAF_TW_L;; esac; }
wait_ram() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$1" ]; do echo "$(date +%H:%M) RAM_WAIT avail<$1G"; sleep 120; done; }
wait_load() { while [ "$(cut -d. -f1 /proc/loadavg)" -gt "$1" ]; do echo "$(date +%H:%M) LOAD_WAIT 1min>$1"; sleep 300; done; }   # 10-21: 24코어 박스 load 88 에서 전처리 291 s/편 → 한가할 때만
MODELS="FP_OFF_TW FP_OFF_HYB FP_OFF_TW_s200 FP_OFF_HYB_s200 FP_OFF_TW_s300 FP_OFF_HYB_s300"
until [ -f /tmp/chain54.done ]; do sleep 120; done
# 10-21 11:40 virtual-face-04 요청: 체인 52 Phase 3(시드 200·300) 종료 후에만 실행(학습 속도 보호)
until [ -f /tmp/chain52p3_s200.done ] && [ -f /tmp/chain52p3_s300.done ]; do sleep 600; done
echo "$(date '+%m-%d %H:%M') CHAIN54B START (MMPD float path, quota ${RPPG_MP_QUOTA:-?})"
for b in 1 2 3 4 5 6; do CACHE=fp_mmpd_pftw_B$b; fpdone=0; [ -f $PFA/cache/DataFileLists/${CACHE}_0.0_1.0.csv ] && fpdone=1   # 10-22: 템플릿별 EXP_DATA_NAME(pftw/hyb4)이 달라 HYB 셀이 캐시를 못 찾던 결함 수정 — 한 캐시(pftw) 공유
  for M in $MODELS; do [ -d $PFA/runs/ev_${M}_MMPD_B$b ] && continue; PFN=$(tpl $M)
    [ $fpdone = 0 ] && { wait_ram 16; wait_load 30; }
    python scripts/fp_yaml.py ev $M $PFN MMPD_B$b --preprocess $((1-fpdone)) > /dev/null; y=$PFA/yaml_$M/ev_${M}_MMPD_B$b.yaml
    sed -i "s#EXP_DATA_NAME: fp_mmpd_[a-z0-9]*_B$b#EXP_DATA_NAME: $CACHE#" $y; [ $fpdone = 1 ] && sed -i "s/DO_PREPROCESS: true/DO_PREPROCESS: false/" $y; fpdone=1
    if is_hyb $M; then ( export RPPG_NOISE_HEAD=1; run_gated 3000 $y /tmp/ev_${M}_MMPD_B$b.log ${M}_MMPD_B$b ); else run_gated 3000 $y /tmp/ev_${M}_MMPD_B$b.log ${M}_MMPD_B$b; fi
  done
  drop_cache fp_mmpd_pftw_B$b; drop_cache fp_mmpd_hyb4_B$b; done
for M in $MODELS; do python scripts/agg_mmpd.py $M > /tmp/aggm_$M.log 2>&1; echo "$(date +%H:%M) AGGM_${M}_DONE $(grep -E 'held-out\(7-33\)' /tmp/aggm_$M.log | head -1 | cut -c1-60)"; done
python scripts/fp_official_judge.py | tee /tmp/fp_official_judge.txt | tail -n 8
echo "$(date '+%m-%d %H:%M') CHAIN54B DONE"; touch /tmp/chain54b.done
