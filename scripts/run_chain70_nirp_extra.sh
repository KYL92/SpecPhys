#!/bin/bash
# 체인 70 (50 문서 §2b, 2026-10-28 사용자 '옵션 A'): MR-NIRP 추가 RGB 축(사후 평가) — 주차장 975 nm 정지/소동작/대동작 + 940 nm 주행 정지/소동작·주차장 정지/소동작
#  = 7 축 × (공식·플리커 제거·광대역 0.02 3시드 + 체인 69 변형 셀 전부) zero-shot 평가(PhysFormer, 축당 첫 모델만 전처리) → extra_axes.py → corpus_trend_judge.py(사후 표적).
#  체인 69 와 병행: 체크포인트가 있는 셀부터 평가하고 체인 69 종료(/tmp/chain69.done) 후 마지막 일괄 평가. 캐시는 마지막에 삭제. 로그 /tmp/chain70.log, 마커 /tmp/chain70.done. flock.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
exec 9>/tmp/chain70.lock; flock -n 9 || { echo "chain70 already running (lock) -> exit"; exit 0; }
cd $ROOT; RES=$ROOT/_workspace/data/results
ts() { date '+%m-%d %H:%M'; }
fail() { echo "$(ts) $1 -> abort"; touch /tmp/chain70.fail; exit 1; }
AXES_A="NIRP_GSTILL NIRP_GSMALL NIRP_GLARGE"; AXES_B="NIRP_STILL940 NIRP_SMALL940 NIRP_GSTILL940 NIRP_GSMALL940"
until [ -f /tmp/nirp_extra_A.done ] && [ -f /tmp/nirp_extra_B.done ]; do sleep 120; done
for AX in $AXES_A $AXES_B; do n=$(ls $PFA/$AX 2>/dev/null | wc -l); [ "$n" -ge 15 ] || fail "AXIS_${AX}_n=$n"; echo "$(ts) axis $AX n=$n"; done   # 일부 피험자는 세션 zip 이 없어 18명(10-28 확인)
echo "$(ts) CHAIN70 START (7 axes)"
models() { for V in TW NOFLK FR FR03 MIX N02 FR04 WIDE NARROW; do for sfx in "" _s200 _s300; do M=CoTAF_${V}_L$sfx; [ -f $PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth ] && [ -f $PFA/yaml_$M/ev_${M}_NIRP_STILL.yaml ] && echo $M; done; done; }
ev_axis() { local M=$1 AX=$2; local src=$PFA/yaml_$M/ev_${M}_NIRP_STILL.yaml y=$PFA/yaml_$M/ev_${M}_$AX.yaml
  [ -d $PFA/runs/ev_${M}_$AX ] && return; local pre=true; [ -f $PFA/cache/DataFileLists/real_${AX}_0.0_1.0.csv ] && [ -d $PFA/cache/real_$AX ] && pre=false
  sed -e "s#physformer/NIRP_STILL\$#physformer/$AX#" -e "s#EXP_DATA_NAME: real_NIRP_STILL\$#EXP_DATA_NAME: real_$AX#" -e "s#ev_${M}_NIRP_STILL#ev_${M}_$AX#g" -e "s#DO_PREPROCESS: false#DO_PREPROCESS: $pre#; s#DO_PREPROCESS: true#DO_PREPROCESS: $pre#" $src > $y
  grep -q "physformer/$AX" $y && grep -q "real_$AX" $y && grep -q "ev_${M}_$AX" $y || fail "${M}_${AX}_YAML"
  run_gated 6000 $y /tmp/ev_${M}_$AX.log ${M}_$AX; }
pass() { local M AX; for AX in $AXES_A $AXES_B; do for M in $(models); do ev_axis $M $AX; done; done; }
until [ -f /tmp/chain69.done ]; do pass; sleep 600; done
pass
python scripts/extra_axes.py --axes $AXES_A $AXES_B --models $(for V in TW NOFLK FR FR03 MIX N02 FR04 WIDE NARROW; do echo -n "CoTAF_${V}_L "; done) | tee $RES/extra_axes.txt
python scripts/corpus_trend_judge.py 2>&1 | tee $RES/corpus_trend_final_posthoc.txt | grep -E "사후|^===" | head -n 20
for AX in $AXES_A $AXES_B; do drop_cache real_$AX; done
echo "$(ts) CHAIN70 DONE"; touch /tmp/chain70.done
