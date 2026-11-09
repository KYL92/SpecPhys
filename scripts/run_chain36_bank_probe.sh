#!/bin/bash
# 체인 36 (19단계): ① 파형 뱅크 v2 확장(TW2, 3시드) + ② identity-shortcut probe (설계 _workspace/19_bank_expansion_probe.md)
source /SPECPHYS_ROOT/scripts/chain_lib.sh
EXT=$ROOT/_workspace/data/ppg_ext
{
echo "$(date '+%m-%d %H:%M') CHAIN36 START"

# ---------- ② 준비: 실-only 대조 모델 CoTReal_L (161클립, 합성 0) ----------
rm -rf $PFA/COTRAIN_REALONLY; mkdir -p $PFA/COTRAIN_REALONLY
for s in $PFA/COTRAIN_ALLF/subject*; do t=$(readlink -f $s); case $t in *TRAIN20H/*|*TRAIN6HI/*|*TRAIN6LO/*|*CMP_SYN/*) ;; *) ln -sfn "$t" "$PFA/COTRAIN_REALONLY/$(basename $s)";; esac; done
echo "POOL_REALONLY n=$(ls $PFA/COTRAIN_REALONLY | wc -l)"
gen_yaml CoTReal_L REALONLY realonly rl > /dev/null
drop_cache cotrain_realonly
run_gated 21000 "$PFA/yaml_CoTReal_L/finetune_CoTReal_L.yaml" /tmp/ft_CoTReal_L.log CoTReal_L_TRAIN
drop_cache cotrain_realonly
find $PFA/runs/finetune_CoTReal_L/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null

# ---------- ② identity probe (4모델: UBFC_base / CoTReal_L / CoTAF_L / CoTAF_TW_L) ----------
wait_gpu 6000
python $ROOT/scripts/probe_identity.py && echo PROBE_OK || echo PROBE_FAIL

# ---------- ① 뱅크 v2: 다운로드 대기 → 구축 ----------
for i in $(seq 1 360); do grep -qE "DALIA_DL_(OK|FAIL)" $EXT/dl.log 2>/dev/null && break; sleep 60; done
cat $EXT/dl.log 2>/dev/null
python $ROOT/scripts/build_ppg_bank_v2.py && echo BANKV2_OK || echo BANKV2_FAIL

# ---------- ① TW2 코퍼스: config 생성 → 3세트 재주입 → 코덱 사본 → 풀 ----------
python - <<'PYEOF'
import json
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
for c in ("train20h", "train6hi", "train6lo"):
    d = json.load(open(f"{PF}/configs/{c}_tw.json"))
    d["out"] = d["out"].replace("_TW", "_TW2")
    for s in d["subjects"]:
        s["wave_bank"] = f"{PF}/ppg_bank_v2.npz"
        s["wave_bank_nearest"] = True
    json.dump(d, open(f"{PF}/configs/{c}_tw2.json", "w"), indent=1)
    print(f"CONFIG {c}_tw2 out={d['out']} n={len(d['subjects'])}")
PYEOF
for c in train20h train6hi train6lo; do
  U=$(echo $c | tr a-z A-Z)_TW2; rm -rf $PFA/$U
  python $ROOT/generator/build_synth_dataset.py --config $PFA/configs/${c}_tw2.json > /tmp/build_${c}_tw2.log 2>&1 \
    && echo BUILD_${c}_TW2_OK || echo BUILD_${c}_TW2_FAIL
done
grep -h "bank_nearest" /tmp/build_*_tw2.log | head -40
codec_copies TRAIN20H_TW2 CMP_SYN_TW2 sweep
make_pool203 ALLF_TW2 TW2

# ---------- ① CoTAF_TW2_L 3시드 학습+13축+MMPD ----------
for SEED in 100 200 300; do
  S=""; [ "$SEED" != "100" ] && S=_s$SEED
  M=CoTAF_TW2_L$S
  gen_yaml $M ALLF_TW2 allf_tw2 tw2 > /dev/null
  train_eval $M allf_tw2 tw2 $SEED
done

# ---------- 비교: vs TW(주 비교), vs 기준 CoTAF_L ----------
for S in "" _s200 _s300; do
  echo "== cmp CoTAF_TW2_L$S vs CoTAF_TW_L$S"
  ( cd $ROOT && python scripts/cmp_model.py CoTAF_TW2_L$S CoTAF_TW_L$S 2>/dev/null )
done
echo "== cmp CoTAF_TW2_L vs CoTAF_L"
( cd $ROOT && python scripts/cmp_model.py CoTAF_TW2_L CoTAF_L 2>/dev/null )
echo "$(date '+%m-%d %H:%M') CHAIN36 DONE"
} >> /tmp/chain36.log 2>&1
