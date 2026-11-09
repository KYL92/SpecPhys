#!/bin/bash
# 체인 68 (49 문서, 2026-10-28): 플리커 개정판 코퍼스 COTRAIN_ALLF_TWFR(203; 플리커 49 클립 주파수 0.8–2.5 Hz 분산·진폭 0.02) → 채택 레시피 PhysFormer
#  CoTAF_FR_L 시드 100·200·300: 13축 + MMPD + VIPL + 띠 밖 플리커 축 RTNI17/RTNI24(공식 3시드·플리커 제거 3시드도 평가) → 판정.
#  마커 대기: /tmp/fr_pool.done(코퍼스), /tmp/flicker_axes.done(평가 축). 로그 /tmp/chain68.log, 마커 /tmp/chain68.done. flock.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
exec 9>/tmp/chain68.lock; flock -n 9 || { echo "chain68 already running (lock) -> exit"; exit 0; }
cd $ROOT; RES=$ROOT/_workspace/data/results
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) $1 -> abort"; touch /tmp/chain68.fail; exit 1; }
pool_ok() { [ -d $PFA/COTRAIN_ALLF_TWFR ] && [ "$(ls $PFA/COTRAIN_ALLF_TWFR 2>/dev/null | wc -l)" = 203 ] && [ "$(find $PFA/COTRAIN_ALLF_TWFR -xtype l | wc -l)" = 0 ]; }
axes_ok() { [ "$(ls $PFA/RTNI17 2>/dev/null | wc -l)" = 8 ] && [ "$(ls $PFA/RTNI24 2>/dev/null | wc -l)" = 8 ] && [ -f /tmp/flicker_axes.done ]; }
until { [ -f /tmp/fr_pool.done ] || pool_ok; } && axes_ok; do sleep 60; done
pool_ok || fail "POOL_FR"; axes_ok || fail "AXES"
echo "$(ts) CHAIN68 START pool=$(ls $PFA/COTRAIN_ALLF_TWFR | wc -l) flicker_log=$(ls $PFA/COTRAIN_ALLF_TWFR_flicker_log.json 2>/dev/null)"
for S in 100 200 300; do sfx=""; [ $S != 100 ] && sfx=_s$S; M=CoTAF_FR_L$sfx
  gen_yaml $M ALLF_TWFR allf_twfr fr > /dev/null
  CK=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  if [ -f "$CK" ]; then eval_only $M allf_twfr fr; else echo "$(ts) TRAIN $M start (seed $S)"; train_eval $M allf_twfr fr $S; fi
  [ -f "$CK" ] || fail "${M}_TRAIN"
  [ -f $RES/agg_${M}_VIPL.json ] || { python scripts/vipl_yaml.py ev $M "$CK" > /dev/null
    ( export RPPG_CACHE_U8=1; run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL ); python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; }
done
# 띠 밖 플리커 축: RTNI yaml 복제(DATA_PATH/EXP_DATA_NAME/run 경로 치환), 축당 첫 모델만 전처리
ev_axis() { local M=$1 AX=$2 pre=$3; local src=$PFA/yaml_$M/ev_${M}_RTNI.yaml y=$PFA/yaml_$M/ev_${M}_$AX.yaml
  [ -f $src ] || fail "${M}_RTNI_YAML_MISSING"; [ -d $PFA/runs/ev_${M}_$AX ] && return
  sed -e "s#physformer/RTNI\$#physformer/$AX#" -e "s#EXP_DATA_NAME: real_RTNI\$#EXP_DATA_NAME: real_$AX#" -e "s#ev_${M}_RTNI#ev_${M}_$AX#g" -e "s#DO_PREPROCESS: false#DO_PREPROCESS: $pre#; s#DO_PREPROCESS: true#DO_PREPROCESS: $pre#" $src > $y
  grep -q "physformer/$AX" $y && grep -q "real_$AX" $y && grep -q "ev_${M}_$AX" $y || fail "${M}_${AX}_YAML"
  run_gated 6000 $y /tmp/ev_${M}_$AX.log ${M}_$AX; }
MODELS="CoTAF_FR_L CoTAF_FR_L_s200 CoTAF_FR_L_s300 CoTAF_TW_L CoTAF_TW_L_s200 CoTAF_TW_L_s300 CoTAF_NOFLK_L CoTAF_NOFLK_L_s200 CoTAF_NOFLK_L_s300"
for AX in RTNI17 RTNI24; do pre=true; [ -f $PFA/cache/DataFileLists/real_${AX}_0.0_1.0.csv ] && pre=false
  for M in $MODELS; do ev_axis $M $AX $pre; pre=false; done; done
python scripts/flicker_axes.py $MODELS | tee $RES/flicker_axes.txt
python scripts/realonly_judge.py CoTAF_FR_L CoTAF_TW_L | tee $RES/fr_judge_vs_official.txt | tail -n 20
python scripts/realonly_judge.py CoTAF_FR_L CoTAF_NOFLK_L | tee $RES/fr_judge_vs_noflk.txt | tail -n 20
echo "$(ts) CHAIN68 DONE"; touch /tmp/chain68.done
