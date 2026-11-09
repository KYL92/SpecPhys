#!/bin/bash
# 체인 50c (32 문서 §8 10-19, 사전 등록 외 사후 확장): VIPL 단독 + 시간 배속 증강 + 검증 선택 셀 VF{k}_BASEAV_L fold 2~5 (시드 100).
#  근거: Phase 1 fold 1 에서 이 조합이 문헌 GT 5.87 로 공동학습 앵커 TWg 6.09 와 동등(ns) → "공동학습 없이 문헌 레시피 요소 2개로 같은 수준" 주장의 5-fold 확인.
#  체인 50 완료(/tmp/chain50.done, Phase 2b 포함) 대기 후 게이트 12 GB 로 체인 53 과 병행. 로그 /tmp/chain50c.log, 마커 /tmp/chain50c.done. epoch 곡선·전이 평가는 생략(검증 선택 읽기만).
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_CACHE_U8=1 RPPG_PEARSON_SKIP_RE='^p[0-9]+_v[0-9]' RPPG_HR_LABEL_FILE=1 RPPG_VALID_METRIC=mae
Y=$PFA/yaml_VIPL; LISTS=$PFA/cache/DataFileLists; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results
cd $ROOT
fail() { echo "$(date '+%m-%d %H:%M') $1 -> abort"; touch /tmp/chain50c.fail; exit 1; }
cell() { local M=$1 K=$2 SEED=$3 EP=10
  [ -f $RES/agg_${M}_VIPL.json ] && { echo "skip $M (done)"; return; }
  [ -f $LISTS/vipl_valid$K.csv ] || python scripts/vipl_filelist.py valid $K > /dev/null
  [ -f $LISTS/vipl_train${K}_xv.csv ] || python scripts/vipl_filelist.py train $K --xv > /dev/null
  python scripts/vipl_yaml.py train $M $K $LISTS/vipl_train${K}_xv.csv $EP --valid_csv $LISTS/vipl_valid$K.csv > /dev/null
  echo "$(date '+%m-%d %H:%M') TRAIN $M start (seed $SEED, taug=1, valid=1, fold $K)"
  RPPG_TAUG=1 RPPG_SEED=$SEED run_gated $GATE $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN
  [ -f $PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch$((EP-1)).pth ] || fail "${M}_TRAIN_FAIL"
  local HIT=$(grep -ao "HRFILE sidecar hits=[0-9]*" /tmp/ft_$M.log | tail -1 | grep -oE "[0-9]+$"); echo "  $M sidecar hits=${HIT:-none}"
  [ "${HIT:-0}" -gt 0 ] || fail "${M}_HRFILE_FAIL"
  grep -aq "TAUG stats epoch 0.*x2" /tmp/ft_$M.log || fail "${M}_TAUG_FAIL"
  local BEST=$(grep -ao "best trained epoch: [0-9]*" /tmp/ft_$M.log | tail -1 | grep -oE "[0-9]+$"); [ -n "$BEST" ] || fail "${M}_BEST_FAIL"
  echo "  $M best epoch=$BEST | valid MAE per epoch: $(grep -a 'Validation chunk HR MAE' /tmp/ft_$M.log | sed 's/.*MAE=\([0-9.]*\).*/\1/' | tr '\n' ' ')"
  echo $BEST > $PFA/runs/finetune_$M/best_epoch.txt
  python scripts/agg_vipl.py $M 2>&1 | head -3
  echo "$(date '+%m-%d %H:%M') CELL_${M}_DONE"; }
echo "$(date '+%m-%d %H:%M') CHAIN50c waiting for chain50.done"; until [ -f /tmp/chain50.done ]; do sleep 600; done
echo "$(date '+%m-%d %H:%M') CHAIN50c START (BASEAV fold 2~5)"
for k in 2 3 4 5; do cell VF${k}_BASEAV_L $k 100; done
python scripts/agg_vipl.py --cv BASEAV_L 2>&1 | grep -E "^---|all "; python scripts/taug_judge.py phase2 BASEAV_L
echo "$(date '+%m-%d %H:%M') CHAIN50c DONE"; touch /tmp/chain50c.done
