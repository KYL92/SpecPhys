#!/bin/bash
# 체인 46 (28단계): VIPL-HR 벤치마크 — ①제로샷 평가(공식/하이브리드 5시드·소프·POS/CHROM) ②피험자 배타 5-fold 교차검증 3셀(BASE/TW/HYB)
# 로그 /tmp/chain46.log, 완료 마커 /tmp/chain46.done (단계 마커 /tmp/chain46_{zs,cv}.done). 설계서 _workspace/30_vipl_benchmark.md
# 전제: scripts/vipl_preprocess.py 가 cache/vipl_u8 을 완성(로그 VIPL_PREPROCESS_DONE). uint8 캐시 모드 필수.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_CACHE_U8=1
# 레시피 수정(10-14 06:50, fold1 BASE 결함 진단 후): VIPL 파형 라벨은 영상과 위상 동기 안 됨(지연 0 상관 ≈0) → VIPL 청크(p{S}_v{V}…)는
# Neg-Pearson 파형 손실 제외, 주파수(HR) 손실만 감독. 풀 청크(subjectN…)는 기존대로. 트레이너 패치 #17.
export RPPG_PEARSON_SKIP_RE='^p[0-9]+_v[0-9]'
S=/tmp/claude-1000/-home-user-Desktop-Virtual-Face/85a0ece1-eaf6-417d-82b4-58ed42595ae4/scratchpad
Y=$PFA/yaml_VIPL; LISTS=$PFA/cache/DataFileLists
EPOCHS=${VIPL_EPOCHS:-10}
HYB_ENV="RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_DN_ONLY=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"
{
echo "$(date '+%m-%d %H:%M') CHAIN46 START epochs=$EPOCHS"
cd $ROOT
# S1: 전처리 완료 대기
[ -f /tmp/chain46_zs.done ] || while ! grep -q VIPL_PREPROCESS_DONE $S/vipl_preprocess.log 2>/dev/null; do sleep 120; done
grep VIPL_PREPROCESS_DONE $S/vipl_preprocess.log; echo "no-face=$(grep -c 'No Face' $S/vipl_preprocess.log) fail=$(grep -c 'FAIL' $S/vipl_preprocess.log)"
echo "cache GB=$(du -sBG $(readlink -f $PFA/cache/vipl_u8) | cut -f1) n_meta=$(ls $PFA/cache/vipl_u8/*_meta.json | wc -l)"
# S2: 파일 목록
python scripts/vipl_filelist.py all
for k in 1 2 3 4 5; do python scripts/vipl_filelist.py test $k; done
if [ ! -f /tmp/chain46_zs.done ]; then  # 제로샷 단계는 1회만(재기동 시 건너뜀)
# S3a: 고전 기준선(CPU, 백그라운드)
( python scripts/eval_classical_vipl.py 4 > /tmp/chain46_classical.log 2>&1 && touch /tmp/chain46_classical.done ) &
# S3b: 제로샷 평가 (기존 가중치, 학습 없음)
for s in "" _s200 _s300 _s400 _s500; do
  M=CoTAF_TW_L$s; python scripts/vipl_yaml.py ev $M $PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth > /dev/null
  run_gated 6000 $Y/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL
done
for s in "" _s200 _s300 _s400 _s500 _SOUP; do
  M=CoTAM_HYB4${s}_L; [ "$s" = "_SOUP" ] && M=CoTAM_HYB4_SOUP_L || M=CoTAM_HYB4_L$s
  python scripts/vipl_yaml.py ev $M $PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth > /dev/null
  ( export RPPG_NOISE_HEAD=1; run_gated 6000 $Y/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL )
done
python scripts/agg_vipl.py CoTAF_TW_L CoTAF_TW_L_s200 CoTAF_TW_L_s300 CoTAF_TW_L_s400 CoTAF_TW_L_s500 CoTAM_HYB4_L CoTAM_HYB4_L_s200 CoTAM_HYB4_L_s300 CoTAM_HYB4_L_s400 CoTAM_HYB4_L_s500 CoTAM_HYB4_SOUP_L > /tmp/agg_vipl_zeroshot.log 2>&1
while [ ! -f /tmp/chain46_classical.done ]; do sleep 60; done; tail -3 /tmp/chain46_classical.log
python scripts/agg_vipl.py POS CHROM >> /tmp/agg_vipl_zeroshot.log 2>&1
python scripts/vipl_judge.py zeroshot > /tmp/chain46_judge_zs.log 2>&1; cat /tmp/chain46_judge_zs.log
echo "$(date '+%m-%d %H:%M') ZEROSHOT DONE"; touch /tmp/chain46_zs.done
fi
# S4: 학습 풀 uint8 캐시(1회, 교차검증 전 기간 보존)
for P in ALLF_TW:cotrain_allf_tw_u8 ALLF_HYB4:cotrain_allf_hyb4_u8; do
  pool=${P%%:*}; cn=${P##*:}
  [ -f $LISTS/${cn}_0.0_1.0.csv ] || { wait_disk; python scripts/pool_preprocess_u8.py COTRAIN_$pool $cn > /tmp/pool_$cn.log 2>&1; tail -1 /tmp/pool_$cn.log; }
done
# S5: 5-fold 교차검증, fold 우선 순서(셀 균형), 시드 100
for k in 1 2 3 4 5; do
  python scripts/vipl_filelist.py train $k
  python scripts/vipl_filelist.py train $k cotrain_allf_tw_u8
  python scripts/vipl_filelist.py train $k cotrain_allf_hyb4_u8
  for CELL in BASE TW HYB; do
    M=VF${k}_${CELL}_L
    case $CELL in BASE) CSV=$LISTS/vipl_train$k.csv;; TW) CSV=$LISTS/vipl_train${k}_cotrain_allf_tw_u8.csv;; HYB) CSV=$LISTS/vipl_train${k}_cotrain_allf_hyb4_u8.csv;; esac
    [ -f $PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch$((EPOCHS-1)).pth ] && { echo "skip $M (done)"; continue; }
    python scripts/vipl_yaml.py train $M $k $CSV $EPOCHS > /dev/null
    echo "$(date '+%m-%d %H:%M') TRAIN $M start"
    if [ $CELL = HYB ]; then ( export $HYB_ENV; RPPG_SEED=100 run_gated 21000 $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
    else RPPG_SEED=100 run_gated 21000 $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN; fi
    find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch$((EPOCHS-1)).pth" -delete 2>/dev/null
    python scripts/agg_vipl.py $M 2>&1 | head -3
  done
done
python scripts/agg_vipl.py --cv BASE_L TW_L HYB_L > /tmp/agg_vipl_cv.log 2>&1; cat /tmp/agg_vipl_cv.log
python scripts/vipl_judge.py cv > /tmp/chain46_judge_cv.log 2>&1; cat /tmp/chain46_judge_cv.log
echo "$(date '+%m-%d %H:%M') CV DONE"; touch /tmp/chain46_cv.done
echo "$(date '+%m-%d %H:%M') CHAIN46 DONE"; touch /tmp/chain46.done
} >> /tmp/chain46.log 2>&1
