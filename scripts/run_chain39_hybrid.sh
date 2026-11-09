#!/bin/bash
# 체인 39 (22단계): 하이브리드 심박/노이즈 분해 학습 (설계 _workspace/24_hybrid_noise_decomp.md)
#  합성(주행·보행 노이즈 24클립, 노이즈 GT 사이드카) 이중 감독 + 실 데이터 SNR 사전(SiNC) + 탈상관
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.5
export HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb
{
echo "$(date '+%m-%d %H:%M') CHAIN39 START"

# ---------- 트랙 B 라우팅 상한 분석 (무학습) ----------
python $ROOT/scripts/routing_oracle.py

# ---------- DN 24클립 config (주행 12: 실 궤적 이식 / 보행 12: 주기 보행 진동) ----------
python - <<'PYEOF'
import glob, json
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
src = json.load(open(f"{PF}/configs/train20h_tw.json"))
trajs = sorted(glob.glob("/SPECPHYS_ROOT/_workspace/data/faces/driving_real/traj/*.npy"))
bpms = [45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160]
base = src["subjects"] + json.load(open(f"{PF}/configs/train6hi_tw.json"))["subjects"]
subs = []
for i in range(24):
    s = dict(base[i]); cid = 261 + i
    s["id"] = cid; s["bpm"] = bpms[i]; s["seed"] = 900 + cid
    s["wave_bank"] = f"{PF}/ppg_bank.npz"; s["wave_bank_nearest"] = True
    s.pop("wave_bank_idx", None); s["wave_bank_idx"] = i
    s["noise_gt"] = True; s["alpha"] = 0.008
    if i < 12:  # 주행: 실 얼굴상자 궤적 + 조명 드리프트 + 플리커
        s["noise"] = {"illum_drift": 0.2, "flicker_amp": 0.03, "flicker_hz": 1.2 + 0.1 * i,
                      "motion_px": 8, "rot_deg": 2.0, "sensor_std": 3,
                      "motion_traj": trajs[i * 7 % len(trajs)], "face_frac": 0.45}
    else:      # 보행: 주기 보행 진동(1.6~2.4Hz) + 큰 랜덤워크 + 야외 조도 변화
        s["noise"] = {"illum_drift": 0.25, "flicker_amp": 0.0, "flicker_hz": 1.2,
                      "motion_px": 10 + (i % 4) * 2, "rot_deg": 3.0, "sensor_std": 2,
                      "periodic_motion_amp": 5 + (i % 5), "periodic_motion_hz": 1.6 + 0.2 * (i % 5)}
    subs.append(s)
json.dump(dict(fps=src["fps"], size=src["size"], out=f"{PF}/TRAIN24DN", subjects=subs),
          open(f"{PF}/configs/train24dn.json", "w"), indent=1)
print(f"CONFIG train24dn n={len(subs)} (주행 12·보행 12, noise_gt on)")
PYEOF
rm -rf $PFA/TRAIN24DN
( cd $ROOT && python generator/build_synth_dataset.py --config $PFA/configs/train24dn.json > /tmp/build_train24dn.log 2>&1 ) \
  && echo BUILD_DN24_OK || echo BUILD_DN24_FAIL
grep -c "noise_gt.npy" /tmp/build_train24dn.log 2>/dev/null
ls $PFA/TRAIN24DN/subject261/noise_gt.npy > /dev/null 2>&1 && echo NGT_OK || echo NGT_FAIL

# ---------- 사이드카 조립 + 풀(OFFICIAL 203 + 24 = 227) ----------
rm -rf $PFA/noise_gt_hyb; mkdir -p $PFA/noise_gt_hyb
for s in $PFA/TRAIN24DN/subject*; do cp "$s/noise_gt.npy" "$PFA/noise_gt_hyb/$(basename $s).npy"; done
echo "SIDECAR n=$(ls $PFA/noise_gt_hyb | wc -l)"
make_pool ALLF_HYB $PFA/COTRAIN_OFFICIAL $PFA/TRAIN24DN

# ---------- CoTAF_HYB_L 3시드 학습+13축+MMPD ----------
for SEED in 100 200 300; do
  S=""; [ "$SEED" != "100" ] && S=_s$SEED
  M=CoTAF_HYB_L$S
  gen_yaml $M ALLF_HYB allf_hyb hyb > /dev/null
  train_eval $M allf_hyb hyb $SEED
done
for S in "" _s200 _s300; do
  echo "== cmp CoTAF_HYB_L$S vs CoTAF_TW_L$S"
  ( cd $ROOT && python scripts/cmp_model.py CoTAF_HYB_L$S CoTAF_TW_L$S 2>/dev/null )
done
echo "$(date '+%m-%d %H:%M') CHAIN39 DONE"
} >> /tmp/chain39.log 2>&1
