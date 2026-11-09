#!/bin/bash
# 체인 71 (51 문서, 2026-10-29, 사용자 지시 '단일 코퍼스로 전 데이터셋 최적' 검정): 복합 변형 2종(COMBO = 광대역 0.02 + 움직임·드리프트 ×0.5, COMBON = 좁은 띠 0.02 + ×0.5)
#  × 시드 100·200·300 → 13축 + MMPD + VIPL + 띠 밖 플리커(RTNI17/24) + (사후) MR-NIRP 주차장·940 nm 7축 → `corpus_trend_judge.py --single`(H-S1).
#  풀 마커 /tmp/pool_<V>.done 대기(`build_variants_1029.sh`). 로그 /tmp/chain71.log, 마커 /tmp/chain71.done. flock.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
exec 9>/tmp/chain71.lock; flock -n 9 || { echo "chain71 already running (lock) -> exit"; exit 0; }
cd $ROOT; RES=$ROOT/_workspace/data/results
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) $1 -> abort"; touch /tmp/chain71.fail; exit 1; }
VARS="COMBO COMBON"; EXTRA="NIRP_GSTILL NIRP_GSMALL NIRP_GLARGE NIRP_STILL940 NIRP_SMALL940 NIRP_GSTILL940 NIRP_GSMALL940"
ev_from() { local M=$1 AX=$2 SRCAX=$3; local src=$PFA/yaml_$M/ev_${M}_$SRCAX.yaml y=$PFA/yaml_$M/ev_${M}_$AX.yaml
  [ -f $src ] || fail "${M}_${SRCAX}_YAML_MISSING"; [ -d $PFA/runs/ev_${M}_$AX ] && return
  local pre=true; [ -f $PFA/cache/DataFileLists/real_${AX}_0.0_1.0.csv ] && [ -d $PFA/cache/real_$AX ] && pre=false
  sed -e "s#physformer/$SRCAX\$#physformer/$AX#" -e "s#EXP_DATA_NAME: real_$SRCAX\$#EXP_DATA_NAME: real_$AX#" -e "s#ev_${M}_$SRCAX#ev_${M}_$AX#g" -e "s#DO_PREPROCESS: false#DO_PREPROCESS: $pre#; s#DO_PREPROCESS: true#DO_PREPROCESS: $pre#" $src > $y
  grep -q "physformer/$AX" $y && grep -q "real_$AX" $y && grep -q "ev_${M}_$AX" $y || fail "${M}_${AX}_YAML"
  run_gated 6000 $y /tmp/ev_${M}_$AX.log ${M}_$AX; }
cell() { local V=$1 S=$2 sfx=""; [ $S != 100 ] && sfx=_s$S; local M=CoTAF_${V}_L$sfx P=ALLF_TW$V; local c=allf_tw$(echo $V | tr 'A-Z' 'a-z') e=$(echo $V | tr 'A-Z' 'a-z')
  until [ -f /tmp/pool_$V.done ] || { [ -d $PFA/COTRAIN_$P ] && [ "$(ls $PFA/COTRAIN_$P 2>/dev/null | wc -l)" = 203 ] && [ -f $PFA/COTRAIN_${P}_log.json ]; }; do sleep 60; done
  [ "$(ls $PFA/COTRAIN_$P | wc -l)" = 203 ] && [ "$(find $PFA/COTRAIN_$P -xtype l | wc -l)" = 0 ] || fail "POOL_$V"
  gen_yaml $M $P $c $e > /dev/null
  local CK=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  if [ -f "$CK" ]; then eval_only $M $c $e; else echo "$(ts) TRAIN $M start (seed $S, pool $P)"; train_eval $M $c $e $S; fi
  [ -f "$CK" ] || fail "${M}_TRAIN"
  [ -f $RES/agg_${M}_VIPL.json ] || { python scripts/vipl_yaml.py ev $M "$CK" > /dev/null
    ( export RPPG_CACHE_U8=1; run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL ); python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; }
  for AX in RTNI17 RTNI24; do ev_from $M $AX RTNI; done
  echo "$(ts) CHAIN71 $M DONE"; }
echo "$(ts) CHAIN71 START (quota ${RPPG_MP_QUOTA:-?}, workers ${RPPG_NUM_WORKERS:-?})"
for V in $VARS; do for S in 100 200 300; do cell $V $S; done; done
ALLM="CoTAF_COMBO_L CoTAF_COMBON_L"
python scripts/flicker_axes.py $ALLM CoTAF_TW_L CoTAF_NOFLK_L CoTAF_FR_L CoTAF_FR03_L CoTAF_MIX_L CoTAF_N02_L CoTAF_FR04_L CoTAF_WIDE_L CoTAF_NARROW_L > /dev/null 2>&1
python scripts/corpus_trend_judge.py --single 2>&1 | tee $RES/corpus_trend_single.txt | grep -E "^===|H-S1|복합" | head -n 24
# 사후 축(주차장·940 nm): 캐시가 없으면 축당 첫 모델이 전처리
for AX in $EXTRA; do for V in $VARS; do for S in 100 200 300; do sfx=""; [ $S != 100 ] && sfx=_s$S; ev_from CoTAF_${V}_L$sfx $AX NIRP_STILL; done; done; done
python scripts/extra_axes.py --axes $EXTRA --models $ALLM | tee -a $RES/extra_axes.txt
python scripts/corpus_trend_judge.py --single 2>&1 | tee $RES/corpus_trend_single_posthoc.txt | grep -E "사후.*복합|H-S1" | head -n 12
for AX in $EXTRA; do drop_cache real_$AX; done
echo "$(ts) CHAIN71 DONE"; touch /tmp/chain71.done
