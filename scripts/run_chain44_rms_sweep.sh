#!/bin/bash
# 체인 44 (26단계): 주입 진폭 스윕 — 수준(R41/R30/R20) × 다양성(RND) 분리, 뱅크 morphology 고정
# 기준 CoTAF_TW_L(자연 분산, 평균 RMS 0.414). 각 3시드, 전 축 평가(무손실 검사 포함).
source /SPECPHYS_ROOT/scripts/chain_lib.sh
{
echo "$(date '+%m-%d %H:%M') CHAIN44 WAIT chain43"
for i in $(seq 1 1440); do grep -q "CHAIN43 DONE" /tmp/chain43.log 2>/dev/null && break; sleep 60; done
echo "$(date '+%m-%d %H:%M') CHAIN44 START"
python - <<'PYEOF'
import json
import numpy as np
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
VARS = {"R41": 0.414, "R30": 0.30, "R20": 0.20, "RND": None}
rng = np.random.default_rng(2600)
for tag, rms in VARS.items():
    for c in ("train20h", "train6hi", "train6lo"):
        d = json.load(open(f"{PF}/configs/{c}_tw.json"))
        d["out"] = f"{PF}/{c.upper()}_{tag}"
        for s in d["subjects"]:
            s["wave_rms"] = float(rng.uniform(0.2, 0.6)) if rms is None else rms
        json.dump(d, open(f"{PF}/configs/{c}_{tag.lower()}.json", "w"), indent=1)
    print(f"CONFIG {tag} rms={'U(0.2,0.6)' if rms is None else rms}")
PYEOF
for T in r41 r30 r20 rnd; do
  U=$(echo $T | tr a-z A-Z)
  for c in train20h train6hi train6lo; do
    CU=$(echo $c | tr a-z A-Z)_$U; rm -rf $PFA/$CU
    ( cd $ROOT && python generator/build_synth_dataset.py --config $PFA/configs/${c}_${T}.json > /tmp/build_${c}_${T}.log 2>&1 ) \
      && echo BUILD_${CU}_OK || echo BUILD_${CU}_FAIL
  done
  codec_copies TRAIN20H_$U CMP_SYN_$U sweep
  make_pool203 ALLF_$U $U
done
for U in R41 R30 R20 RND; do
  u=$(echo $U | tr A-Z a-z)
  for SEED in 100 200 300; do
    S=""; [ "$SEED" != "100" ] && S=_s$SEED
    M=CoTAF_${U}_L$S
    gen_yaml $M ALLF_$U allf_$u amp$u > /dev/null
    train_eval $M allf_$u amp$u $SEED
  done
  echo "== 3시드 요약 $U"
done
# 풀링 판정 (수준·다양성 분리)
python - <<'PYEOF'
import json, os
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"
def mm(m):
    return {(r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]): (r["err"], r["motion"])
            for r in json.load(open(f"{R}/agg_{m}_MMPD.json")) if r["subject"] >= 7}
def pooled(b):
    ds = [mm(f"{b}{s}") for s in ("", "_s200", "_s300") if os.path.exists(f"{R}/agg_{b}{s}_MMPD.json")]
    ks = sorted(set.intersection(*[set(d) for d in ds]))
    return ks, np.array([ds[0][k][1] for k in ks]), np.array([np.mean([d[k][0] for d in ds]) for k in ks]), len(ds)
ks, mo, TW, _ = pooled("CoTAF_TW_L")
def w(a, b): return wilcoxon(a, b).pvalue
print(f"=== 진폭 스윕 풀링 (vs TW 0.414 자연분산, n={len(ks)}) ===")
print(f"TW      전체 {TW.mean():.2f} | 움직임 {TW[mo>1].mean():.2f}")
for U in ("R41", "R30", "R20", "RND"):
    try:
        k2, m2, Y, n = pooled(f"CoTAF_{U}_L")
        assert k2 == ks
        print(f"{U} ({n}시드) 전체 {Y.mean():.2f} (p={w(TW, Y):.4f}) | 움직임 {Y[mo>1].mean():.2f} (p={w(TW[mo>1], Y[mo>1]):.4f}) | 정지 {Y[mo==1].mean():.2f} (p={w(TW[mo==1], Y[mo==1]):.3f})")
    except Exception as e:
        print(f"{U}: {e}")
print("RMS_SWEEP_DONE")
PYEOF
echo "$(date '+%m-%d %H:%M') CHAIN44 DONE"
} >> /tmp/chain44.log 2>&1
