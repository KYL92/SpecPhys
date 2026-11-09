#!/bin/bash
# 체인 45 (27단계): 학습된 라우터(트랙 A, CPU) + 시드 앙상블/가중치 소프(트랙 B, 평가만). 재학습 없음.
# 로그 /tmp/chain45.log, 완료 마커 /tmp/chain45.done. 설계서 _workspace/29_router_soup.md
source /SPECPHYS_ROOT/scripts/chain_lib.sh
M=CoTAM_HYB4_SOUP_L
{
echo "$(date '+%m-%d %H:%M') CHAIN45 START"
cd $ROOT
# B2-1: 가중치 소프(이미 생성됐으면 재사용) + 평가 yaml(HYB4와 동일 구성, 평가 전용)
[ -f $PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth ] || python scripts/make_soup.py CoTAM_HYB4_L $M
gen_yaml $M ALLF_HYB4 allf_hyb4soup h4soup > /dev/null; rm -f $PFA/yaml_$M/finetune_$M.yaml   # 학습 yaml 제거(평가 전용 명시)
echo "$(date '+%m-%d %H:%M') SOUP_READY n_ev_yaml=$(ls $PFA/yaml_$M/ev_*.yaml | wc -l)"
# A-1: 관측 특징 추출(CPU 2워커, 백그라운드) — 완료 마커 /tmp/chain45_feat.done
rm -f /tmp/chain45_feat.done
( python scripts/mmpd_obs_features.py 2 > /tmp/chain45_feat.log 2>&1 && touch /tmp/chain45_feat.done ) &
# B1: 시드 예측 앙상블(CPU)
python scripts/seed_ensemble.py > /tmp/chain45_ens.log 2>&1 && echo "$(date '+%m-%d %H:%M') ENSEMBLE_OK" || echo "ENSEMBLE_FAIL"
grep -E "MMPD|held-out" /tmp/chain45_ens.log
# B2-2: 소프 평가(GPU 게이트, 13축 + MMPD B1~6, 배치마다 drop_cache) — 노이즈 헤드 키 strict 로드용 env
( export RPPG_NOISE_HEAD=1; eval_only $M allf_hyb4soup h4soup )
cat /tmp/agg_$M.log
# A-2: 특징 완료 대기 후 라우터 CV
while [ ! -f /tmp/chain45_feat.done ]; do sleep 60; done
tail -2 /tmp/chain45_feat.log
python scripts/learn_router.py > /tmp/chain45_router.log 2>&1 && echo "$(date '+%m-%d %H:%M') ROUTER_OK" || echo "ROUTER_FAIL"
cat /tmp/chain45_router.log
echo "$(date '+%m-%d %H:%M') CHAIN45 DONE"
touch /tmp/chain45.done
} >> /tmp/chain45.log 2>&1
