#!/bin/bash
# 체인 41d+42 (논문 완성도): ①HYB6(층화 DN + 대역 한정 비율 GT) 3시드 ②시드 5개 확장(TW·HYB4 s400/s500) ③5시드 풀링 재판정
source /SPECPHYS_ROOT/scripts/chain_lib.sh
{
echo "$(date '+%m-%d %H:%M') CHAIN41D START"
# ---------- 41d: HYB6 (환경은 여기서만 하이브리드) ----------
(
export RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_ADAPTIVE=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25 HYB_W_CONF=1.0
export HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb4 RPPG_CONF_DUMP=1
make_pool ALLF_HYB6 $PFA/COTRAIN_OFFICIAL $PFA/TRAIN24DNT $PFA/TRAIN24DNT2 $PFA/MMPD_TRAIN
for SEED in 100 200 300; do
  S=""; [ "$SEED" != "100" ] && S=_s$SEED
  M=CoTAM_HYB6_L$S
  gen_yaml $M ALLF_HYB6 allf_hyb6 hyb6 > /dev/null
  train_eval $M allf_hyb6 hyb6 $SEED
  echo "== cmp $M vs CoTAF_TW_L$S"
  ( cd $ROOT && python scripts/cmp_model.py $M CoTAF_TW_L$S 2>/dev/null | head -1 )
  ( cd $ROOT && python scripts/motion_split_cmp.py $M CoTAF_TW_L$S 2>/dev/null | sed -n '2,4p' )
done
( cd $ROOT && python scripts/conf_routing.py CoTAM_HYB6_L )
)
echo "$(date '+%m-%d %H:%M') CHAIN41D DONE"

# ---------- 42: 시드 확장 (하이브리드 env 없이 — TW는 기본, HYB4는 V4 env) ----------
echo "$(date '+%m-%d %H:%M') CHAIN42 START (시드 5개 확장)"
for SEED in 400 500; do
  M=CoTAF_TW_L_s$SEED
  gen_yaml $M OFFICIAL official tw5s$SEED > /dev/null
  train_eval $M official tw5s$SEED $SEED
done
(
export RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_DN_ONLY=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25
export HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2
for SEED in 400 500; do
  M=CoTAM_HYB4_L_s$SEED
  gen_yaml $M ALLF_HYB4 allf_hyb4 h4s$SEED > /dev/null
  train_eval $M allf_hyb4 h4s$SEED $SEED
done
)
# 5시드 풀링 재판정
( cd $ROOT && python scripts/pool5_judge.py )
echo "$(date '+%m-%d %H:%M') CHAIN42 DONE"
} >> /tmp/chain42.log 2>&1
