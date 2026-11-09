#!/bin/bash
# 체인 39c (V3): SiNC를 DN 샘플에만 선택 적용 + DN 24→48 확대 (요인 분해 결론 반영)
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_DN_ONLY=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25
export HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2
{
echo "$(date '+%m-%d %H:%M') CHAIN39C WAIT chain39b"
for i in $(seq 1 720); do grep -q "CHAIN39B DONE" /tmp/chain39.log 2>/dev/null && break; sleep 60; done
echo "$(date '+%m-%d %H:%M') CHAIN39C START (V3: SiNC DN-only, DN 48)"
# DN2 24클립 (다른 시드·노이즈 draw, ids 285~308)
python - <<'PYEOF'
import json
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
d = json.load(open(f"{PF}/configs/train24dn.json"))
for i, s in enumerate(d["subjects"]):
    s["id"] = 285 + i; s["seed"] = 1900 + i; s["wave_bank_idx"] = i + 7
    s["bpm"] = max(42, min(182, s["bpm"] + 4))
    if "motion_traj" in s.get("noise", {}):
        s["noise"]["motion_traj"] = s["noise"]["motion_traj"]  # 궤적은 chain 스크립트에서 회전
json.dump({**d, "out": f"{PF}/TRAIN24DN2"}, open(f"{PF}/configs/train24dn2.json", "w"), indent=1)
print("CONFIG train24dn2 n=24")
PYEOF
python - <<'PYEOF'
import glob, json
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
trajs = sorted(glob.glob("/SPECPHYS_ROOT/_workspace/data/faces/driving_real/traj/*.npy"))
d = json.load(open(f"{PF}/configs/train24dn2.json"))
for i, s in enumerate(d["subjects"]):
    if "motion_traj" in s.get("noise", {}):
        s["noise"]["motion_traj"] = trajs[(i * 7 + 3) % len(trajs)]
json.dump(d, open(f"{PF}/configs/train24dn2.json", "w"), indent=1)
print("TRAJ rotated")
PYEOF
rm -rf $PFA/TRAIN24DN2
( cd $ROOT && python generator/build_synth_dataset.py --config $PFA/configs/train24dn2.json > /tmp/build_train24dn2.log 2>&1 ) \
  && echo BUILD_DN2_OK || echo BUILD_DN2_FAIL
rm -rf $PFA/noise_gt_hyb2; mkdir -p $PFA/noise_gt_hyb2
for s in $PFA/TRAIN24DN/subject* $PFA/TRAIN24DN2/subject*; do cp "$s/noise_gt.npy" "$PFA/noise_gt_hyb2/$(basename $s).npy" 2>/dev/null; done
echo "SIDECAR2 n=$(ls $PFA/noise_gt_hyb2 | wc -l)"
make_pool ALLF_HYB2 $PFA/COTRAIN_OFFICIAL $PFA/TRAIN24DN $PFA/TRAIN24DN2
for SEED in 100 200 300; do
  S=""; [ "$SEED" != "100" ] && S=_s$SEED
  M=CoTAF_HYB3_L$S
  gen_yaml $M ALLF_HYB2 allf_hyb2 hyb3 > /dev/null
  train_eval $M allf_hyb2 hyb3 $SEED
  echo "== cmp $M vs CoTAF_TW_L$S"
  ( cd $ROOT && python scripts/cmp_model.py $M CoTAF_TW_L$S 2>/dev/null )
  ( cd $ROOT && python scripts/motion_split_cmp.py $M CoTAF_TW_L$S 2>/dev/null )
done
echo "$(date '+%m-%d %H:%M') CHAIN39C DONE"
} >> /tmp/chain39.log 2>&1
