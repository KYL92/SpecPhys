#!/bin/bash
# 체인 38 (21단계): 뱅크 확장 LODO-CV — 5폴드(DLCN·MMPD·NIRP·MCD·PURE), 폴드별 대상 데이터셋 GT 제외 뱅크로
# TW식 42클립 재주입 → CoTAF_LO<F>_L 학습(시드 100) → 대상 축만 평가 → 요약 표.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
{
echo "$(date '+%m-%d %H:%M') CHAIN38 START"
python $ROOT/scripts/build_ppg_bank_lodo.py || echo BANKLODO_FAIL

# 폴드별 config 생성 (TW config에서 out·wave_bank만 교체 — idx 정책은 TW와 동일, nearest 미사용)
python - <<'PYEOF'
import json
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
for fold in ("DLCN", "MMPD", "NIRP", "MCD", "PURE"):
    for c in ("train20h", "train6hi", "train6lo"):
        d = json.load(open(f"{PF}/configs/{c}_tw.json"))
        d["out"] = d["out"].replace("_TW", f"_LO{fold}")
        for s in d["subjects"]:
            s["wave_bank"] = f"{PF}/bank_lodo_{fold}.npz"
            s.pop("wave_bank_nearest", None)
        json.dump(d, open(f"{PF}/configs/{c}_lo{fold.lower()}.json", "w"), indent=1)
    print(f"CONFIG LO{fold} 3sets")
PYEOF

for F in DLCN MMPD NIRP MCD PURE; do
  f=${F,,}; M=CoTAF_LO${F}_L
  echo "$(date '+%m-%d %H:%M') FOLD_${F}_BEGIN"
  for c in train20h train6hi train6lo; do
    U=$(echo $c | tr a-z A-Z)_LO$F; rm -rf $PFA/$U
    python $ROOT/generator/build_synth_dataset.py --config $PFA/configs/${c}_lo$f.json > /tmp/build_${c}_lo$f.log 2>&1 \
      && echo BUILD_${c}_LO${F}_OK || echo BUILD_${c}_LO${F}_FAIL
  done
  codec_copies TRAIN20H_LO$F CMP_SYN_LO$F sweep
  make_pool203 ALLF_LO$F LO$F
  gen_yaml $M ALLF_LO$F allf_lo$f lo$f > /dev/null
  drop_cache cotrain_allf_lo$f
  run_gated 21000 "$PFA/yaml_$M/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN
  drop_cache cotrain_allf_lo$f
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
  case $F in
    DLCN) EVAXES="DLCN";;
    MMPD) EVAXES="";;
    NIRP) EVAXES="NIRP_STILL NIRP_SMALL NIRP_LARGE";;
    MCD)  EVAXES="MCD_A MCD_B";;
    PURE) EVAXES="PURE";;
  esac
  for A in $EVAXES; do
    run_gated 6000 "$PFA/yaml_$M/ev_${M}_$A.yaml" /tmp/ev_${M}_$A.log ${M}_$A
  done
  if [ "$F" = "MMPD" ]; then
    for k in 1 2 3 4 5 6; do
      run_gated 6000 "$PFA/yaml_$M/ev_${M}_MMPD_B$k.yaml" /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k
      drop_cache mmpd_lo${f}_B$k
    done
  fi
  ( cd $ROOT && python scripts/agg_model2.py $M > /dev/null 2>&1; [ "$F" = "MMPD" ] && python scripts/agg_mmpd.py $M > /dev/null 2>&1 )
  echo "$(date '+%m-%d %H:%M') FOLD_${F}_DONE"
done

python $ROOT/scripts/lodo_summary.py
echo "$(date '+%m-%d %H:%M') CHAIN38 DONE"
} >> /tmp/chain38.log 2>&1
