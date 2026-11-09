#!/bin/bash
# 체인 43 (논문 완성도): PhysMamba + 하이브리드(V4 레시피) — 방법 일반성 1행
#  PM_CoTAM_L(동일 코퍼스 ALLF_HYB4, 하이브리드 무) vs PM_HYB4_L(하이브리드), 각 3시드, MMPD 평가만.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
gen_pm() { local M=$1 P=$2 c=$3 PRE=$4
  local Y=$(gen_yaml $M $P $c pmh)
  sed -i -E -e "s/NAME: PhysFormer/NAME: PhysMamba/" \
    -e "s#UBFC-rPPG_PhysFormer_DiffNormalized.pth#UBFC-rPPG_PhysMamba_DiffNormalized.pth#" \
    -e "s/LR: 5e-5/LR: 1e-4/" \
    -e "s/CHUNK_LENGTH: 160/CHUNK_LENGTH: 128/" \
    -e "s/EXP_DATA_NAME: real_/EXP_DATA_NAME: pmh_/" \
    -e "s/EXP_DATA_NAME: synth_/EXP_DATA_NAME: pmh_s_/" $Y/*.yaml
  for f in $Y/ev_${M}_*.yaml; do sed -i -E "s/DO_PREPROCESS: (true|false)/DO_PREPROCESS: $PRE/" "$f"; done
  python - "$Y/finetune_$M.yaml" <<'PYEOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
parts = re.split(r"(DO_PREPROCESS: \w+)", s); n = 0
for i, seg in enumerate(parts):
    if seg.startswith("DO_PREPROCESS:"):
        n += 1; parts[i] = "DO_PREPROCESS: " + ("true" if n == 1 else "true")
open(p, "w").write("".join(parts))
PYEOF
  echo "GENPM $M"; }
{
echo "$(date '+%m-%d %H:%M') CHAIN43 START"
FIRST=1
for V in CoTAM HYB4; do
  for SEED in 100 200 300; do
    S=""; [ "$SEED" != "100" ] && S=_s$SEED
    M=PM_${V}_L$S
    PRE=false; [ "$FIRST" = "1" ] && PRE=true
    c=$(echo pm_${V}_${SEED} | tr A-Z a-z)
    gen_pm $M ALLF_HYB4 $c $PRE > /dev/null
    drop_cache cotrain_$c
    if [ "$V" = "HYB4" ]; then
      ( export RPPG_NOISE_HEAD=1 RPPG_HYBRID=1 SINC_PRIOR=1 SINC_DN_ONLY=1 HYB_W_NOISE=1.0 HYB_W_DECOR=0.25
        export HYB_NOISE_GT_DIR=$PFA/noise_gt_hyb2
        RPPG_SEED=$SEED run_gated 21000 "$PFA/yaml_$M/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN )
    else
      RPPG_SEED=$SEED run_gated 21000 "$PFA/yaml_$M/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN
    fi
    drop_cache cotrain_$c
    find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
    FIRST=0
  done
done
# MMPD 평가: 배치별로 첫 실행(전처리 true)은 PM_CoTAM_L, 나머지 5개 재사용 → 드롭
for k in 1 2 3 4 5 6; do
  for V in CoTAM HYB4; do for S in "" _s200 _s300; do
    M=PM_${V}_L$S
    ENVH=""; [ "$V" = "HYB4" ] && ENVH="RPPG_NOISE_HEAD=1"
    ( [ -n "$ENVH" ] && export RPPG_NOISE_HEAD=1
      run_gated 6000 "$PFA/yaml_$M/ev_${M}_MMPD_B$k.yaml" /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k )
  done; done
  drop_cache mmpd_pmh_B$k
done
for V in CoTAM HYB4; do for S in "" _s200 _s300; do
  ( cd $ROOT && python scripts/agg_mmpd.py PM_${V}_L$S > /dev/null 2>&1 )
done; done
# 3시드 풀링 움직임 분할 판정 (PM 백본 내)
python - <<'PYEOF'
import json, numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"
def mm(m):
    return {(r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]): (r["err"], r["motion"])
            for r in json.load(open(f"{R}/agg_{m}_MMPD.json")) if r["subject"] >= 7}
tw = [mm(f"PM_CoTAM_L{s}") for s in ("", "_s200", "_s300")]
hb = [mm(f"PM_HYB4_L{s}") for s in ("", "_s200", "_s300")]
ks = sorted(set(tw[0]) & set(hb[0]))
MO = np.array([tw[0][k][1] for k in ks])
X = np.array([np.mean([d[k][0] for d in tw]) for k in ks])
Y = np.array([np.mean([d[k][0] for d in hb]) for k in ks])
def w(a, b): return wilcoxon(a, b).pvalue
print(f"=== PhysMamba 백본 3시드 풀링 (n={len(ks)}) ===")
for nm, sel in (("전체", MO > 0), ("정지", MO == 1), ("움직임", MO > 1), ("Walking", MO == 4)):
    print(f"{nm:8s} PM기준 {X[sel].mean():6.2f} -> PM+하이브리드 {Y[sel].mean():6.2f} (p={w(X[sel], Y[sel]):.4f})")
print("PMHYB_DONE")
PYEOF
echo "$(date '+%m-%d %H:%M') CHAIN43 DONE"
} >> /tmp/chain43.log 2>&1
