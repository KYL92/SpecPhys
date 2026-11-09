#!/bin/bash
# 체인 51 v2 (기여 점검 — 데이터 편 대조군, 2026-10-17): 실-only 코퍼스 모델 CoTReal_L(실 161 클립 = 공식 코퍼스에서 가상얼굴 42 제거, 동일 레시피:
#  UBFC 체크포인트 미세조정 10 ep·lr 5e-5·batch 4·cumsum 라벨) 시드 100(기존 체크포인트, 미평가)·200·300 → 13축 + MMPD 6배치(캐시 공유) + VIPL zero-shot
#  → CoTAF_TW_L 3시드와 축별 쌍대(가상얼굴 42 클립의 zero-shot 순수 기여, 33 문서). 체인 49 와 GPU 병행(게이트 12 GB).
# v2: 시드 학습은 float 전처리(워커 6개 × 2~5 GB, 16:05 메모리 압박 사고) 대신 체인 46b 가 만든 uint8 풀 캐시 cotrain_allf_real_u8(수치 동치, 1,854 청크)을
#     VIPL-HR 로더 + csv 목록(realonly_u8.csv)으로 읽어 학습(RPPG_CACHE_U8=1, 전처리 없음). 로그 /tmp/chain51.log, 마커 /tmp/chain51.done
source /SPECPHYS_ROOT/scripts/chain_lib.sh
cd $ROOT
LISTS=$PFA/cache/DataFileLists; MODELS="CoTReal_L CoTReal_L_s200 CoTReal_L_s300"
echo "$(date '+%m-%d %H:%M') CHAIN51 v2 START"
for S in 200 300; do
  M=CoTReal_L_s$S; gen_yaml $M REALONLY realonly rl > /dev/null   # 축 평가 yaml(13축+MMPD, 캐시 mmpd_rl_*)
  ck=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  [ -f $ck ] && { echo "skip train $M"; continue; }
  python scripts/vipl_yaml.py train $M 1 $LISTS/realonly_u8.csv 10 > /dev/null   # 학습 yaml: VIPL-HR 로더 + u8 풀 목록(VIPL 청크 없음), 자체 테스트 = VIPL fold 1(참고)
  echo "$(date '+%m-%d %H:%M') TRAIN $M start (seed $S, u8 pool cache)"
  RPPG_CACHE_U8=1 RPPG_SEED=$S run_gated 12000 $PFA/yaml_VIPL/finetune_$M.yaml /tmp/ft_$M.log ${M}_TRAIN
  [ -f $ck ] || { echo "$(date '+%m-%d %H:%M') ${M}_TRAIN_FAIL -> abort"; touch /tmp/chain51.fail; exit 1; }
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
done
# 13축 (float 캐시, MMPD 제외)
for M in $MODELS; do
  for f in $PFA/yaml_$M/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*|VIPL) continue;; esac
    [ -d $PFA/runs/ev_${M}_$A ] && { echo "skip ev $M $A"; continue; }; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
done
# MMPD 6배치: 캐시(mmpd_rl_B*) 를 3 모델이 공유한 뒤 삭제
for k in 1 2 3 4 5 6; do
  for M in $MODELS; do [ -d $PFA/runs/ev_${M}_MMPD_B$k ] && continue; run_gated 6000 $PFA/yaml_$M/ev_${M}_MMPD_B$k.yaml /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k; done
  drop_cache mmpd_rl_B$k
done
for M in $MODELS; do python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; python scripts/agg_mmpd.py $M >> /tmp/agg_$M.log 2>&1; echo "$(date +%H:%M) AGG_${M}_DONE"; grep -E "^(MCD|PURE|DLCN|NIRP|REALTEST|RTN|C3|C6|E3|MMPD)" /tmp/agg_$M.log | head -16; done
# VIPL zero-shot (uint8 공유 캐시)
export RPPG_CACHE_U8=1
for M in $MODELS; do
  ck=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  python scripts/vipl_yaml.py ev $M $ck > /dev/null
  run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL
  python scripts/agg_vipl.py $M 2>&1 | grep -E "^---|all "
done
unset RPPG_CACHE_U8
python scripts/realonly_judge.py 2>&1 | tail -40
echo "$(date '+%m-%d %H:%M') CHAIN51 DONE"; touch /tmp/chain51.done
