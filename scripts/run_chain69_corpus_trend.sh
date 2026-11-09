#!/bin/bash
# 체인 69 (50 문서, 2026-10-28, 사용자 /goal): 코퍼스 교란 설정 경향 — 짝 풀 변형 6종(FR03·MIX·N02·FR04·WIDE·NARROW, `build_variants_1028.sh` 가 순차 생성)
#  → 채택 레시피 PhysFormer 셀 CoTAF_<V>_L 시드 100(13축 + MMPD + VIPL + 띠 밖 플리커 1.7/2.4 Hz) → `corpus_trend_judge.py --select`(50 §3 선택 규칙)
#  → 선택 변형(최대 3) 시드 200·300 → 최종 판정. 로그 /tmp/chain69.log, 마커 /tmp/chain69.done. flock.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
exec 9>/tmp/chain69.lock; flock -n 9 || { echo "chain69 already running (lock) -> exit"; exit 0; }
cd $ROOT; RES=$ROOT/_workspace/data/results
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) $1 -> abort"; touch /tmp/chain69.fail; exit 1; }
VARS="FR03 MIX N02 FR04 WIDE NARROW"
ev_axis() { local M=$1 AX=$2; local src=$PFA/yaml_$M/ev_${M}_RTNI.yaml y=$PFA/yaml_$M/ev_${M}_$AX.yaml
  [ -f $src ] || fail "${M}_RTNI_YAML_MISSING"; [ -d $PFA/runs/ev_${M}_$AX ] && return
  [ -f $PFA/cache/DataFileLists/real_${AX}_0.0_1.0.csv ] || fail "${AX}_CACHE_MISSING"
  sed -e "s#physformer/RTNI\$#physformer/$AX#" -e "s#EXP_DATA_NAME: real_RTNI\$#EXP_DATA_NAME: real_$AX#" -e "s#ev_${M}_RTNI#ev_${M}_$AX#g" -e "s#DO_PREPROCESS: true#DO_PREPROCESS: false#" $src > $y
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
  for AX in RTNI17 RTNI24; do ev_axis $M $AX; done
  echo "$(ts) CHAIN69 $M DONE"; }
ALLM() { for V in $VARS; do echo -n "CoTAF_${V}_L "; done; echo -n "CoTAF_TW_L CoTAF_NOFLK_L CoTAF_FR_L"; }
echo "$(ts) CHAIN69 START (quota ${RPPG_MP_QUOTA:-?}, workers ${RPPG_NUM_WORKERS:-?})"
for V in $VARS; do cell $V 100; done
python scripts/flicker_axes.py $(ALLM) > /dev/null 2>&1
python scripts/corpus_trend_judge.py --select 2>&1 | tee $RES/corpus_trend_phase1.txt | tail -n 12
SEL=$(cat $RES/corpus_trend_selected.txt 2>/dev/null)
echo "$(ts) PHASE2 selected: '$SEL'"
for V in $SEL; do for S in 200 300; do cell $V $S; done; done
python scripts/flicker_axes.py $(ALLM) > /dev/null 2>&1
python scripts/corpus_trend_judge.py 2>&1 | tee $RES/corpus_trend_final.txt | tail -n 40
echo "$(ts) CHAIN69 DONE"; touch /tmp/chain69.done
