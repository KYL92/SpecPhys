#!/bin/bash
# 체인 37 (20단계): 학습된 열화 diffusion(깨끗→열화) → 저SNR 가상 클립 12개 주입 → CoTAF_TWDG_L 3시드
# 설계: _workspace/22_degradation_diffusion_design.md. 체인 36 종료 대기 후 시작.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
SCR=$ROOT/scripts/degrade_diffusion.py
{
echo "$(date '+%m-%d %H:%M') CHAIN37 START (chain36 대기)"
for i in $(seq 1 2160); do grep -q "CHAIN36 DONE" /tmp/chain36.log 2>/dev/null && break; sleep 60; done
grep -q "CHAIN36 DONE" /tmp/chain36.log && echo CHAIN36_WAIT_OK || echo "WARN chain36 미종료 — 그래도 진행"

# ---------- 열화 모델 ----------
python $SCR pairs && echo PAIRSDEG_OK || echo PAIRSDEG_FAIL
wait_gpu 8000
python $SCR train --steps 20000 && echo TRAINDEG_OK || echo TRAINDEG_FAIL
python $SCR check
python $SCR genclips --frames 1008 --id0 241 && echo GENCLIPS_OK || echo GENCLIPS_FAIL

# ---------- 12 DG 클립 빌드 + 풀(OFFICIAL 203 + 12 = 215) ----------
python - <<'PYEOF'
import json
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
src = json.load(open(f"{PF}/configs/train20h_tw.json"))
bpms = [45, 55, 65, 75, 85, 95, 105, 120, 135, 150, 165, 180]
subs = []
for i in range(12):
    s = dict(src["subjects"][i]); cid = 241 + i
    s["id"] = cid; s["bpm"] = bpms[i]; s["seed"] = 700 + cid
    for k in ("wave_bank", "wave_bank_idx", "wave_bank_nearest"): s.pop(k, None)
    s["wave_file"] = f"{PF}/degwaves/dg_{cid}.npz"
    subs.append(s)
json.dump(dict(fps=src["fps"], size=src["size"], out=f"{PF}/TRAIN12DG", subjects=subs),
          open(f"{PF}/configs/train12dg.json", "w"), indent=1)
print("CONFIG train12dg n=12")
PYEOF
rm -rf $PFA/TRAIN12DG
python $ROOT/generator/build_synth_dataset.py --config $PFA/configs/train12dg.json > /tmp/build_train12dg.log 2>&1 \
  && echo BUILD_DG_OK || echo BUILD_DG_FAIL
grep "\[build\]" /tmp/build_train12dg.log | head -12
make_pool ALLF_TWDG $PFA/COTRAIN_OFFICIAL $PFA/TRAIN12DG

# ---------- CoTAF_TWDG_L 3시드 ----------
for SEED in 100 200 300; do
  S=""; [ "$SEED" != "100" ] && S=_s$SEED
  M=CoTAF_TWDG_L$S
  gen_yaml $M ALLF_TWDG allf_twdg twdg > /dev/null
  train_eval $M allf_twdg twdg $SEED
done
for S in "" _s200 _s300; do
  echo "== cmp CoTAF_TWDG_L$S vs CoTAF_TW_L$S"
  ( cd $ROOT && python scripts/cmp_model.py CoTAF_TWDG_L$S CoTAF_TW_L$S 2>/dev/null )
done
echo "$(date '+%m-%d %H:%M') CHAIN37 DONE"
} >> /tmp/chain37.log 2>&1
