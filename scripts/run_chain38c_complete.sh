#!/bin/bash
# 체인 38c: 체인 38 잔여 인수 완주 (소유: user-bc, 사용자 지시 2026-09-02)
# ① MMPD B5/B6 재평가·재집계 ② NIRP/MCD/PURE 폴드 실행(원 설계 그대로) ③ 요약 ④ 캐시 sub-disk 전환
source /SPECPHYS_ROOT/scripts/chain_lib.sh
{
echo "$(date '+%m-%d %H:%M') CHAIN38C START (owner user-bc)"

# ---------- ① MMPD 폴드 복구 ----------
M=CoTAF_LOMMPD_L
for k in 5 6; do
  run_gated 6000 "$PFA/yaml_$M/ev_${M}_MMPD_B$k.yaml" /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k
  drop_cache mmpd_lommpd_B$k
done
( cd $ROOT && python scripts/agg_model2.py $M > /dev/null 2>&1 && python scripts/agg_mmpd.py $M > /dev/null 2>&1 ) \
  && echo MMPD_REAGG_OK || echo MMPD_REAGG_FAIL

# ---------- ② 잔여 3폴드 (원 설계: 확장 뱅크 - 대상 GT, TW식 42 재주입, 풀 203) ----------
for F in NIRP MCD PURE; do
  f=$(echo $F | tr A-Z a-z); M=CoTAF_LO${F}_L
  echo "$(date '+%m-%d %H:%M') FOLD_${F}_BEGIN"
  for c in train20h train6hi train6lo; do
    U=$(echo $c | tr a-z A-Z)_LO$F; rm -rf $PFA/$U
    ( cd $ROOT && python generator/build_synth_dataset.py --config $PFA/configs/${c}_lo$f.json > /tmp/build_${c}_lo$f.log 2>&1 ) \
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
    NIRP) EVAXES="NIRP_STILL NIRP_SMALL NIRP_LARGE";;
    MCD)  EVAXES="MCD_A MCD_B";;
    PURE) EVAXES="PURE";;
  esac
  for A in $EVAXES; do
    run_gated 6000 "$PFA/yaml_$M/ev_${M}_$A.yaml" /tmp/ev_${M}_$A.log ${M}_$A
  done
  ( cd $ROOT && python scripts/agg_model2.py $M > /dev/null 2>&1 )
  echo "$(date '+%m-%d %H:%M') FOLD_${F}_DONE"
done

# ---------- ③ 요약 ----------
python $ROOT/scripts/lodo_summary2.py

# ---------- ④ 캐시 sub-disk 전환 (모든 캐시 사용 종료 후) ----------
DST=/SPECPHYS_CACHE/vface_pf_cache
if [ ! -L "$PFA/cache" ]; then
  rsync -a --delete "$PFA/cache/" "$DST/" && mv "$PFA/cache" "$PFA/cache_localbak" \
    && ln -s "$DST" "$PFA/cache" && rm -rf "$PFA/cache_localbak" \
    && echo "CACHE_MIGRATED $(df -h / | awk 'NR==2{print $4}') free" || echo CACHE_MIGRATE_FAIL
fi
echo "$(date '+%m-%d %H:%M') CHAIN38C DONE"
} >> /tmp/chain38c.log 2>&1
