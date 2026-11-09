#!/bin/bash
# 체인 41c: 신뢰도 GT 포화 수정 — DN 노이즈 강도 3단 층화(×0.15/0.4/1.0) → 비율 GT 스펙트럼 확보
# 게이트: 빌드 후 비율 분포 p25<0.8 확인 후에만 학습. 모델 CoTAM_HYB6_L 3시드.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_ADAPTIVE=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25 HYB_W_CONF=1.0
export HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb4 RPPG_CONF_DUMP=1
{
echo "$(date '+%m-%d %H:%M') CHAIN41C START"
python - <<'PYEOF'
import json
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
TIERS = [0.15, 0.4, 1.0]
SCALED = ("illum_drift", "flicker_amp", "motion_px", "rot_deg", "sensor_std", "periodic_motion_amp")
for cfg, out in (("train24dn", "TRAIN24DNT"), ("train24dn2", "TRAIN24DNT2")):
    d = json.load(open(f"{PF}/configs/{cfg}.json"))
    for i, s in enumerate(d["subjects"]):
        t = TIERS[i % 3]
        nz = s.get("noise", {})
        for k in SCALED:
            if k in nz: nz[k] = round(nz[k] * t, 4)
        if "face_frac" in nz: nz["face_frac"] = round(nz["face_frac"] * t, 4)  # 실 궤적 진폭도 층화
        s["noise_tier"] = t
    d["out"] = f"{PF}/{out}"
    json.dump(d, open(f"{PF}/configs/{cfg}_t3.json", "w"), indent=1)
    print(f"CONFIG {cfg}_t3 (tiers 0.15/0.4/1.0)")
PYEOF
for cfg in train24dn_t3 train24dn2_t3; do
  U=TRAIN24DNT; [ "$cfg" = "train24dn2_t3" ] && U=TRAIN24DNT2
  rm -rf $PFA/$U
  ( cd $ROOT && python generator/build_synth_dataset.py --config $PFA/configs/$cfg.json > /tmp/build_$cfg.log 2>&1 ) \
    && echo BUILD_${U}_OK || echo BUILD_${U}_FAIL
done
rm -rf $PFA/noise_gt_hyb4; mkdir -p $PFA/noise_gt_hyb4
for s in $PFA/TRAIN24DNT/subject* $PFA/TRAIN24DNT2/subject*; do cp "$s/noise_gt.npz" "$PFA/noise_gt_hyb4/$(basename $s).npz" 2>/dev/null; done
echo "SIDECAR4 n=$(ls $PFA/noise_gt_hyb4 | wc -l)"
# 게이트: 비율 GT 분포 변별력 확인
python - <<'PYEOF' || { echo RATIO_GATE_FAIL; exit 1; }
import glob, numpy as np, sys
rs=[]
for f in glob.glob("/SPECPHYS_ROOT/_workspace/data/physformer/noise_gt_hyb4/*.npz"):
    z=np.load(f); nd,td=z["noise_dn"],z["total_dn"]; L=160
    for k in range(len(nd)//L):
        rs.append(min(1.0,float(np.std(nd[k*L:(k+1)*L])/(np.std(td[k*L:(k+1)*L])+1e-8))))
rs=np.array(rs); q=np.percentile(rs,[5,25,50,75,95]).round(3)
print(f"RATIO_GATE p5/25/50/75/95={q}")
sys.exit(0 if q[1] < 0.8 else 1)
PYEOF
echo RATIO_GATE_OK
make_pool ALLF_HYB6 $PFA/COTRAIN_OFFICIAL $PFA/TRAIN24DNT $PFA/TRAIN24DNT2 $PFA/MMPD_TRAIN
for SEED in 100 200 300; do
  S=""; [ "$SEED" != "100" ] && S=_s$SEED
  M=CoTAM_HYB6_L$S
  gen_yaml $M ALLF_HYB6 allf_hyb6 hyb6 > /dev/null
  train_eval $M allf_hyb6 hyb6 $SEED
  echo "== cmp $M vs CoTAF_TW_L$S"
  ( cd $ROOT && python scripts/cmp_model.py $M CoTAF_TW_L$S 2>/dev/null )
  ( cd $ROOT && python scripts/motion_split_cmp.py $M CoTAF_TW_L$S 2>/dev/null )
done
( cd $ROOT && python scripts/conf_routing.py CoTAM_HYB6_L )
echo "$(date '+%m-%d %H:%M') CHAIN41C DONE"
} >> /tmp/chain41.log 2>&1
