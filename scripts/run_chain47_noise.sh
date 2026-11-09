#!/bin/bash
# 체인 47 통합 트랙 LN (29단계, 설계 _workspace/31_noise_benchmark.md v2): 문헌 정합 프로토콜 + 노이즈 환경 집중을 한 트랙으로.
#  공통 프로토콜(모든 셀): VIPL 청크 HR 라벨 = 장치 HR 사이드카(RPPG_HR_LABEL_FILE=1), VIPL 파형 손실 제외, 평가 = VIPL fold 1 held-out(문헌 GT 주·정본 부) + 휴대폰 소스 부분집합 + MR-NIRP driving 3축 + DLCN.
#  체인 46(CV)과 GPU 번갈아 사용: 본 체인 게이트 12 GB(체인 46 학습 중에도 시작), 체인 46 은 21 GB 게이트라 본 셀 종료 후 재개.
#  순서: TWg(앵커) → LIT(문헌 레시피 기준선) → HYBg → 기존 VF1_TW/HYB NIRP 평가 → MOBg·MOBHYBg → MR-NIRP in-domain 5-fold × {TWN, HYBN}
# 로그 /tmp/chain47.log, 마커 /tmp/chain47_{L,N1,N2,N3}.done, /tmp/chain47.done
# v3(10-14 22:10 재기동): ①트레이너 사이드카 경로 결함 수정 후 재시작(결함판 TWg 산출물은 VF1_TWdup_L 로 보존 = TW 결정론 복제)
#   ②스테이지 A 는 누락 피험자만 단일 워커로 복구(3 워커 전처리가 메모리 압박으로 주행 영상 110개 누락) ③모든 학습 셀에 사이드카 적중 검증(0이면 FAIL·중단)
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_CACHE_U8=1 RPPG_PEARSON_SKIP_RE='^p[0-9]+_v[0-9]' RPPG_HR_LABEL_FILE=1
S=/tmp/claude-1000/-home-user-Desktop-Virtual-Face/85a0ece1-eaf6-417d-82b4-58ed42595ae4/scratchpad
Y=$PFA/yaml_VIPL; LISTS=$PFA/cache/DataFileLists; GATE=12000
HYB_ENV="RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_DN_ONLY=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25 HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2"
train_cell() { # train_cell NAME CSV EPOCHS HYB(0/1) [vipl_yaml extra args]
  local M=$1 CSV=$2 EP=$3 HYB=$4; shift 4
  [ -f $PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch$((EP-1)).pth ] && { echo "skip $M (done)"; return; }
  python scripts/vipl_yaml.py train $M 1 $CSV $EP "$@" > /dev/null
  echo "$(date '+%m-%d %H:%M') TRAIN $M start"
  if [ $HYB = 1 ]; then ( export $HYB_ENV; RPPG_SEED=100 run_gated $GATE $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN )
  else RPPG_SEED=100 run_gated $GATE $Y/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN; fi
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch$((EP-1)).pth" -delete 2>/dev/null
  local HIT=$(grep -ao "HRFILE sidecar hits=[0-9]*" /tmp/ft_$M.log | tail -1 | grep -oE "[0-9]+$")
  echo "  $M sidecar hits(epoch0 first 200 batches)=${HIT:-none}"
  [ "${HIT:-0}" -gt 0 ] || { echo "$(date '+%m-%d %H:%M') ${M}_HRFILE_FAIL: sidecar HR label not applied -> abort chain"; touch /tmp/chain47.fail; exit 1; }
  python scripts/agg_vipl.py $M 2>&1 | head -3
}
nirp_eval() { # nirp_eval NAME HYB(0/1) [EPOCH=9]: NIRP 3축 + DLCN 평가(기존 float32 캐시) → agg_model2 + 움직임 분할
  local M=$1 HYB=$2 EP=${3:-9}
  gen_yaml $M ALLF_TW cotrain_allf_tw x > /dev/null
  [ "$EP" != "9" ] && sed -i "s#_physformer_Epoch9.pth#_physformer_Epoch$EP.pth#" $PFA/yaml_$M/ev_${M}_*.yaml
  for ax in NIRP_STILL NIRP_SMALL NIRP_LARGE DLCN; do
    [ -d $PFA/runs/ev_${M}_$ax ] && { echo "skip ev $M $ax"; continue; }
    if [ $HYB = 1 ]; then ( export RPPG_NOISE_HEAD=1; run_gated 6000 $PFA/yaml_$M/ev_${M}_$ax.yaml /tmp/ev_${M}_$ax.log ${M}_$ax )
    else run_gated 6000 $PFA/yaml_$M/ev_${M}_$ax.yaml /tmp/ev_${M}_$ax.log ${M}_$ax; fi
  done
  python scripts/agg_model2.py $M 2>&1 | grep -E "NIRP|DLCN" | head -6; python scripts/motion_split_cmp.py $M 2>&1 | grep NIRP
}
{
echo "$(date '+%m-%d %H:%M') CHAIN47 START (merged track LN, gate ${GATE}MB)"
cd $ROOT
# A. MR-NIRP 학습 풀 uint8 캐시 복구 (CPU, 단일 워커, 병행): 누락 피험자만 NIRP_TRAIN_MISS 로 재처리 → 커버리지 검증 → csv 재작성 → 마커
( if python scripts/nirp_cache_check.py; then echo "NIRP cache already complete"; else
    for try in 1 2; do
      wait_disk; RPPG_MP_QUOTA=1 nice -n 10 python scripts/pool_preprocess_u8.py NIRP_TRAIN_MISS nirp_train_u8 > /tmp/pool_nirp_train_u8_miss$try.log 2>&1
      tail -c 300 /tmp/pool_nirp_train_u8_miss$try.log | tr '\r' '\n' | tail -1
      python scripts/nirp_cache_check.py && break
    done
  fi
  if python scripts/nirp_cache_check.py; then python scripts/nirp_cache_check.py --csv; echo "$(date '+%m-%d %H:%M') NIRP_CACHE_OK"; touch /tmp/chain47_nirpcache.done
  else echo "$(date '+%m-%d %H:%M') NIRP_CACHE_FAIL (still missing after 2 tries) -> C3 blocked"; fi ) &
# B. 정합 앵커 + 문헌 레시피 기준선 + 하이브리드 (fold 1)
if [ ! -f /tmp/chain47_L.done ]; then
  train_cell VF1_TWg_L $LISTS/vipl_train1_cotrain_allf_tw_u8.csv 10 0; nirp_eval VF1_TWg_L 0
  ( unset RPPG_PEARSON_SKIP_RE; export PF_LOSS_A=0.1; train_cell VF1_LIT_L $LISTS/vipl_train1.csv 25 0 --lr 1e-4 --scratch ); nirp_eval VF1_LIT_L 0 24
  train_cell VF1_HYBg_L $LISTS/vipl_train1_cotrain_allf_hyb4_u8.csv 10 1; nirp_eval VF1_HYBg_L 1
  echo "$(date '+%m-%d %H:%M') STAGE B DONE"; touch /tmp/chain47_L.done
fi
# C1. 기존 fold 1 모델(파형 라벨)의 NIRP/DLCN 전이 평가 — 라벨 효과 대조
if [ ! -f /tmp/chain47_N1.done ]; then nirp_eval VF1_TW_L 0; nirp_eval VF1_HYB_L 1; touch /tmp/chain47_N1.done; fi
# C2. 휴대폰 소스 집중 셀
if [ ! -f /tmp/chain47_N2.done ]; then
  python scripts/vipl_filelist.py train 1 --src 2 cotrain_allf_tw_u8; python scripts/vipl_filelist.py train 1 --src 2 cotrain_allf_hyb4_u8
  train_cell VF1_MOBg_L $LISTS/vipl_train1_src2_cotrain_allf_tw_u8.csv 10 0; nirp_eval VF1_MOBg_L 0
  train_cell VF1_MOBHYBg_L $LISTS/vipl_train1_src2_cotrain_allf_hyb4_u8.csv 10 1; nirp_eval VF1_MOBHYBg_L 1
  touch /tmp/chain47_N2.done
fi
# C3. MR-NIRP in-domain 5-fold (VIPL fold 1 held-out 고정)
while [ ! -f /tmp/chain47_nirpcache.done ]; do [ -f /tmp/chain47.fail ] && exit 1; sleep 300; done
for k in 1 2 3 4 5; do
  python scripts/nirp_filelist.py train $k nirp_train_u8 cotrain_allf_tw_u8; python scripts/nirp_filelist.py train $k nirp_train_u8 cotrain_allf_hyb4_u8
  train_cell NF${k}_TWN_L $LISTS/nirp_train${k}_nirp_train_u8_cotrain_allf_tw_u8.csv 10 0; nirp_eval NF${k}_TWN_L 0
  train_cell NF${k}_HYBN_L $LISTS/nirp_train${k}_nirp_train_u8_cotrain_allf_hyb4_u8.csv 10 1; nirp_eval NF${k}_HYBN_L 1
done
touch /tmp/chain47_N3.done
echo "$(date '+%m-%d %H:%M') CHAIN47 DONE"; touch /tmp/chain47.done
} >> /tmp/chain47.log 2>&1
