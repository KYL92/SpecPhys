#!/bin/bash
# 체인 41 (24단계): 노이즈 세기 감독 + 학습된 신뢰도 + 적응형 SNR 사전 (설계 _workspace/26)
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_ADAPTIVE=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25 HYB_W_CONF=1.0
export HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb3 RPPG_CONF_DUMP=1
{
echo "$(date '+%m-%d %H:%M') CHAIN41 START"
# DN 48 재빌드 (세기 보존 npz 사이드카)
for cfg in train24dn train24dn2; do
  U=$(echo $cfg | tr a-z A-Z | sed 's/TRAIN24DN2/TRAIN24DN2/;s/^TRAIN24DN$/TRAIN24DN/')
  rm -rf $PFA/$U
  ( cd $ROOT && python generator/build_synth_dataset.py --config $PFA/configs/$cfg.json > /tmp/build_${cfg}_v2.log 2>&1 ) \
    && echo BUILD_${U}_OK || echo BUILD_${U}_FAIL
done
rm -rf $PFA/noise_gt_hyb3; mkdir -p $PFA/noise_gt_hyb3
for s in $PFA/TRAIN24DN/subject* $PFA/TRAIN24DN2/subject*; do cp "$s/noise_gt.npz" "$PFA/noise_gt_hyb3/$(basename $s).npz" 2>/dev/null; done
echo "SIDECAR3 n=$(ls $PFA/noise_gt_hyb3 | wc -l)"
make_pool ALLF_HYB5 $PFA/COTRAIN_OFFICIAL $PFA/TRAIN24DN $PFA/TRAIN24DN2 $PFA/MMPD_TRAIN
for SEED in 100 200 300; do
  S=""; [ "$SEED" != "100" ] && S=_s$SEED
  M=CoTAM_HYB5_L$S
  gen_yaml $M ALLF_HYB5 allf_hyb5 hyb5 > /dev/null
  train_eval $M allf_hyb5 hyb5 $SEED
  echo "== cmp $M vs CoTAF_TW_L$S"
  ( cd $ROOT && python scripts/cmp_model.py $M CoTAF_TW_L$S 2>/dev/null )
  ( cd $ROOT && python scripts/motion_split_cmp.py $M CoTAF_TW_L$S 2>/dev/null )
done
( cd $ROOT && python scripts/conf_routing.py CoTAM_HYB5_L )
echo "$(date '+%m-%d %H:%M') CHAIN41 DONE"
} >> /tmp/chain41.log 2>&1
