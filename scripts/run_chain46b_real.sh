#!/bin/bash
# 체인 46b (28단계 보강, 체인 46 CV 완료 후 자동): ①REAL 셀 — VIPL 4-fold + 공식 코퍼스의 실 데이터 161만(가상얼굴 2xx·6xx 제외)
#   → TW 대비 가상얼굴 42클립의 순수 기여 분리(H-C1 해석의 교란 제거: BASE 붕괴는 '정렬 라벨 데이터 부재' 효과일 수 있음).
#   ②fold 1 BASE epoch 곡선 재현(VF1_BASEC_L, 동일 시드, epoch 체크포인트 보존 후 held-out 평가).
# 로그 /tmp/chain46b.log, 마커 /tmp/chain46b.done
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_CACHE_U8=1 RPPG_PEARSON_SKIP_RE='^p[0-9]+_v[0-9]'
Y=$PFA/yaml_VIPL; LISTS=$PFA/cache/DataFileLists; EPOCHS=${VIPL_EPOCHS:-10}
{
echo "$(date '+%m-%d %H:%M') CHAIN46B START (waiting chain46 cv)"
while [ ! -f /tmp/chain46_cv.done ]; do sleep 300; done
cd $ROOT
# 풀: 실 161 (COTRAIN_ALLF_TW 에서 subject 201~299, 601~699 제외)
P=$PFA/COTRAIN_ALLF_REAL; rm -rf $P; mkdir -p $P
for s in $PFA/COTRAIN_ALLF_TW/subject*; do n=${s##*subject}; { [ $n -ge 201 ] && [ $n -le 299 ]; } || { [ $n -ge 601 ] && [ $n -le 699 ]; } || ln -sfn "$(readlink -f $s)" $P/subject$n; done
echo "POOL_ALLF_REAL n=$(ls $P | wc -l) (expect 161)"
[ -f $LISTS/cotrain_allf_real_u8_0.0_1.0.csv ] || { wait_disk; python scripts/pool_preprocess_u8.py COTRAIN_ALLF_REAL cotrain_allf_real_u8 > /tmp/pool_cotrain_allf_real_u8.log 2>&1; tail -1 /tmp/pool_cotrain_allf_real_u8.log; }
for k in 1 2 3 4 5; do
  python scripts/vipl_filelist.py train $k cotrain_allf_real_u8
  M=VF${k}_REAL_L; CSV=$LISTS/vipl_train${k}_cotrain_allf_real_u8.csv
  [ -f $PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch$((EPOCHS-1)).pth ] && { echo "skip $M (done)"; continue; }
  python scripts/vipl_yaml.py train $M $k $CSV $EPOCHS > /dev/null
  echo "$(date '+%m-%d %H:%M') TRAIN $M start"
  RPPG_SEED=100 run_gated 21000 $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch$((EPOCHS-1)).pth" -delete 2>/dev/null
  python scripts/agg_vipl.py $M 2>&1 | head -3
done
python scripts/agg_vipl.py --cv REAL_L > /tmp/agg_vipl_cv_real.log 2>&1; cat /tmp/agg_vipl_cv_real.log
# fold 1 BASE epoch 곡선 (체크포인트 보존 keeper 병행)
M=VF1_BASEC_L; python scripts/vipl_yaml.py train $M 1 $LISTS/vipl_train1.csv $EPOCHS > /dev/null
mkdir -p $PFA/runs/finetune_$M/epochs_keep
( for e in 0 1 2 3 4 5 6 7 8; do f=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch$e.pth; until [ -f "$f" ] || [ -f /tmp/chain46b_basec.done ]; do sleep 60; done; [ -f "$f" ] && { sleep 30; cp "$f" $PFA/runs/finetune_$M/epochs_keep/; }; done ) &
echo "$(date '+%m-%d %H:%M') TRAIN $M start"; RPPG_SEED=100 run_gated 21000 $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN; touch /tmp/chain46b_basec.done
python scripts/agg_vipl.py $M 2>&1 | head -3
bash scripts/chain46_epoch_curve.sh $M EP1BASE 1 0 1 2 4 6 8
python scripts/vipl_judge.py cv > /tmp/chain46_judge_cv.log 2>&1; cat /tmp/chain46_judge_cv.log
echo "$(date '+%m-%d %H:%M') CHAIN46B DONE"; touch /tmp/chain46b.done
} >> /tmp/chain46b.log 2>&1
