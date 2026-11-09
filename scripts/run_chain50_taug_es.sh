#!/bin/bash
# 체인 50 (32 문서, 2026-10-17): 시간 배속 증강(A, 패치 #30 RPPG_TAUG) + 검증 기반 epoch 선택(V, 패치 #31 + vipl_valid{k}.csv) — fold 1 파일럿 6 셀
#  → Phase 2(자동 조건부: 공동학습 승자 시드 200/300 + fold 2~5 → 5-fold 문헌 표) → Phase 2b(H-T2 지지 시 VIPL 단독+증강 fold 2~5).
# 체인 49(/tmp/chain49.done)·51(/tmp/chain51.done) 완료 대기 후 단독 GPU(게이트 21 GB). 로그 /tmp/chain50.log, 마커 /tmp/chain50_{p1,p2,p2b}.done, /tmp/chain50.done
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_CACHE_U8=1 RPPG_PEARSON_SKIP_RE='^p[0-9]+_v[0-9]' RPPG_HR_LABEL_FILE=1 RPPG_VALID_METRIC=mae
Y=$PFA/yaml_VIPL; LISTS=$PFA/cache/DataFileLists; GATE=${GATE:-12000}; RES=$ROOT/_workspace/data/results   # 10-17 /goal: 체인 52 와 병행(게이트 12 GB)
cd $ROOT
fail() { echo "$(date '+%m-%d %H:%M') $1 -> abort"; touch /tmp/chain50.fail; exit 1; }
nirp_eval() { local M=$1 EP=$2; gen_yaml $M ALLF_TW cotrain_allf_tw x > /dev/null
  [ "$EP" != "9" ] && sed -i "s#_physformer_Epoch9.pth#_physformer_Epoch$EP.pth#" $PFA/yaml_$M/ev_${M}_*.yaml
  for ax in NIRP_STILL NIRP_SMALL NIRP_LARGE DLCN; do [ -d $PFA/runs/ev_${M}_$ax ] && continue; run_gated 6000 $PFA/yaml_$M/ev_${M}_$ax.yaml /tmp/ev_${M}_$ax.log ${M}_$ax; done
  python scripts/agg_model2.py $M 2>&1 | grep -E "NIRP|DLCN" | head -6; }
# cell NAME FOLD SEED TRAIN_CSV TAUG(0/1) VALID(0/1) POOL(0/1)
cell() { local M=$1 K=$2 SEED=$3 CSV=$4 TAUG=$5 V=$6 POOL=$7 EP=10
  [ -f $RES/agg_${M}_VIPL.json ] && { echo "skip $M (done)"; return; }
  local extra=""; [ $V = 1 ] && extra="--valid_csv $LISTS/vipl_valid$K.csv"
  python scripts/vipl_yaml.py train $M $K $CSV $EP $extra > /dev/null
  echo "$(date '+%m-%d %H:%M') TRAIN $M start (seed $SEED, taug=$TAUG, valid=$V, fold $K)"
  if [ $TAUG = 1 ]; then RPPG_TAUG=1 RPPG_SEED=$SEED run_gated $GATE $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN
  else RPPG_SEED=$SEED run_gated $GATE $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN; fi
  [ -f $PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch$((EP-1)).pth ] || fail "${M}_TRAIN_FAIL"
  local HIT=$(grep -ao "HRFILE sidecar hits=[0-9]*" /tmp/ft_$M.log | tail -1 | grep -oE "[0-9]+$"); echo "  $M sidecar hits=${HIT:-none}"
  [ "${HIT:-0}" -gt 0 ] || fail "${M}_HRFILE_FAIL"
  if [ $TAUG = 1 ]; then grep -a "TAUG stats epoch 0" /tmp/ft_$M.log | cut -c1-120; grep -aq "TAUG stats epoch 0.*x2" /tmp/ft_$M.log || fail "${M}_TAUG_FAIL"; fi
  local BEST=$((EP-1))
  if [ $V = 1 ]; then BEST=$(grep -ao "best trained epoch: [0-9]*" /tmp/ft_$M.log | tail -1 | grep -oE "[0-9]+$"); [ -n "$BEST" ] || fail "${M}_BEST_FAIL"
    echo "  $M best epoch=$BEST | valid MAE per epoch: $(grep -a 'Validation chunk HR MAE' /tmp/ft_$M.log | sed 's/.*MAE=\([0-9.]*\).*/\1/' | tr '\n' ' ')"; fi
  echo $BEST > $PFA/runs/finetune_$M/best_epoch.txt
  ln -sfn $PFA/runs/finetune_$M/PreTrainedModels $PFA/runs/finetune_$M/epochs_keep
  python scripts/agg_vipl.py $M 2>&1 | head -3
  bash scripts/chain46_epoch_curve.sh $M EP_$M $K 0 1 2 3 4 5 6 7 8 9 2>&1 | grep -E "all \[문헌|missing" | sed 's/^/    /' | head -12
  [ $POOL = 1 ] && nirp_eval $M $BEST
  echo "$(date '+%m-%d %H:%M') CELL_${M}_DONE"; }
echo "$(date '+%m-%d %H:%M') CHAIN50 waiting for chain49/chain51 markers"
until [ -f /tmp/chain49.done ]; do sleep 600; done   # 체인 51 은 평가만 남아 비대기(10-17)
echo "$(date '+%m-%d %H:%M') CHAIN50 START (Phase 1, fold 1)"
TWPOOL=cotrain_allf_tw_u8
cell VF1_TWgA_L  1 100 $LISTS/vipl_train1_$TWPOOL.csv    1 0 1
cell VF1_TWgV_L  1 100 $LISTS/vipl_train1_xv_$TWPOOL.csv 0 1 1
cell VF1_TWgAV_L 1 100 $LISTS/vipl_train1_xv_$TWPOOL.csv 1 1 1
cell VF1_BASEA_L  1 100 $LISTS/vipl_train1.csv    1 0 0
cell VF1_BASEV_L  1 100 $LISTS/vipl_train1_xv.csv 0 1 0
cell VF1_BASEAV_L 1 100 $LISTS/vipl_train1_xv.csv 1 1 0
python scripts/taug_judge.py phase1 1; touch /tmp/chain50_p1.done
W=$(python -c "import json;print(json.load(open('$RES/taug_phase1.json'))['winner'])"); HT2=$(python -c "import json;print(json.load(open('$RES/taug_phase1.json'))['h_t2'])")
echo "$(date '+%m-%d %H:%M') PHASE1 winner=$W h_t2=$HT2"
cfg() { case $1 in TWgA) echo "plain 1 0";; TWgV) echo "xv 0 1";; TWgAV) echo "xv 1 1";; esac; }
if [ "$W" != "none" ]; then
  read LST TA VV <<< "$(cfg $W)"
  for S in 200 300; do csv=$LISTS/vipl_train1_$TWPOOL.csv; [ $LST = xv ] && csv=$LISTS/vipl_train1_xv_$TWPOOL.csv; cell VF1_${W}_L_s$S 1 $S $csv $TA $VV 1; done
  for k in 2 3 4 5; do
    if [ $LST = xv ]; then [ -f $LISTS/vipl_valid$k.csv ] || python scripts/vipl_filelist.py valid $k > /dev/null; [ -f $LISTS/vipl_train${k}_xv_$TWPOOL.csv ] || python scripts/vipl_filelist.py train $k --xv $TWPOOL > /dev/null; csv=$LISTS/vipl_train${k}_xv_$TWPOOL.csv
    else csv=$LISTS/vipl_train${k}_$TWPOOL.csv; fi
    cell VF${k}_${W}_L $k 100 $csv $TA $VV 0
  done
  python scripts/agg_vipl.py --cv ${W}_L 2>&1 | grep -E "^---|all "; python scripts/taug_judge.py phase2 ${W}_L
fi
touch /tmp/chain50_p2.done
if [ "$HT2" = "True" ]; then
  for k in 2 3 4 5; do cell VF${k}_BASEA_L $k 100 $LISTS/vipl_train$k.csv 1 0 0; done
  python scripts/agg_vipl.py --cv BASEA_L 2>&1 | grep -E "^---|all "; python scripts/taug_judge.py phase2 BASEA_L
fi
touch /tmp/chain50_p2b.done
echo "$(date '+%m-%d %H:%M') CHAIN50 DONE"; touch /tmp/chain50.done
