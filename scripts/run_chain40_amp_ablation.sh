#!/bin/bash
# 체인 40 (23단계): 주입 진폭 혼입 ablation (사용자 발견 2026-09-03)
#  해석적 std=1 vs TW max-abs=1(RMS 0.41) — 진폭 요인과 morphology 요인의 2셀 분리:
#  AL  = 해석적 파형 + RMS 0.414 (morphology 유지, 진폭을 TW에 정합)
#  TWH = 뱅크 파형   + RMS 1.0   (morphology 유지, 진폭을 해석적에 정합)
source /SPECPHYS_ROOT/scripts/chain_lib.sh
{
echo "$(date '+%m-%d %H:%M') CHAIN40 START"
python - <<'PYEOF'
import json
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
for tag, srcs, rms in (("al", ("train20h", "train6hi", "train6lo"), 0.414),
                       ("twh", ("train20h_tw", "train6hi_tw", "train6lo_tw"), 1.0)):
    for c in srcs:
        d = json.load(open(f"{PF}/configs/{c}.json"))
        base = c.replace("_tw", "")
        d["out"] = f"{PF}/{base.upper()}_{tag.upper()}"
        for s in d["subjects"]:
            s["wave_rms"] = rms
        json.dump(d, open(f"{PF}/configs/{base}_{tag}.json", "w"), indent=1)
        print(f"CONFIG {base}_{tag} rms={rms}")
PYEOF
for T in al twh; do
  for c in train20h train6hi train6lo; do
    U=$(echo $c | tr a-z A-Z)_$(echo $T | tr a-z A-Z); rm -rf $PFA/$U
    ( cd $ROOT && python generator/build_synth_dataset.py --config $PFA/configs/${c}_${T}.json > /tmp/build_${c}_${T}.log 2>&1 ) \
      && echo BUILD_${U}_OK || echo BUILD_${U}_FAIL
  done
  UT=$(echo $T | tr a-z A-Z)
  codec_copies TRAIN20H_$UT CMP_SYN_$UT sweep
  make_pool203 ALLF_$UT $UT
done
for T in AL TWH; do
  t=$(echo $T | tr A-Z a-z)
  for SEED in 100 200 300; do
    S=""; [ "$SEED" != "100" ] && S=_s$SEED
    M=CoTAF_${T}_L$S
    gen_yaml $M ALLF_$T allf_$t amp$t > /dev/null
    train_eval $M allf_$t amp$t $SEED
    echo "== cmp $M vs CoTAF_TW_L$S"
    ( cd $ROOT && python scripts/cmp_model.py $M CoTAF_TW_L$S 2>/dev/null | head -1 )
    echo "== cmp $M vs CoTAF_L$S"
    ( cd $ROOT && python scripts/cmp_model.py $M CoTAF_L$S 2>/dev/null | head -1 )
  done
done
echo "$(date '+%m-%d %H:%M') CHAIN40 DONE"
} >> /tmp/chain40.log 2>&1
