#!/bin/bash
# 체인 8 — MMPD 추가: CoTAM(=CoTAF 풀 + MMPD 피험자 1~6) 학습 → 기존 축 평가 → MMPD 6배치 × 3캐시군 평가(캐시 즉시 삭제)
set -u
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RPPG_MP_QUOTA=4
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox
PF=../../_workspace/data/physformer; Y=$PF/yaml17; PFA=/SPECPHYS_ROOT/_workspace/data/physformer
wait_gpu() { local need=$1; while true; do
  local used total; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  [ $((total-used)) -ge "$need" ] && return 0; sleep 60; done; }
run_gated() { # need yaml log tag
  for attempt in 1 2; do
    wait_gpu "$1"
    if python main.py --config_file "$2" > "$3" 2>&1; then echo "$(date +%H:%M) ${4}_OK"; return 0; fi
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$3"; then echo "$(date +%H:%M) ${4}_OOM_RETRY"; sleep 300
    else echo "$(date +%H:%M) ${4}_FAIL"; return 1; fi
  done; echo "${4}_FAIL"; return 1; }
drop_cache() { rm -rf "$PFA/cache/$1" "$PFA/cache/DataFileLists/${1}_0.0_1.0.csv"; }

# 0. MMPD 학습분할 변환 대기
until grep -q "\[mmpd-train\] done" /tmp/mmpd_convert.log 2>/dev/null; do sleep 60; done
# 1. 풀 조립 COTRAIN_ALLFM = COTRAIN_ALLF(203) + MMPD_TRAIN(120)
mkdir -p $PFA/COTRAIN_ALLFM
for s in $PFA/COTRAIN_ALLF/subject*; do t=$(readlink -f "$s"); ln -sfn "$t" "$PFA/COTRAIN_ALLFM/$(basename "$s")"; done
for s in $PFA/MMPD_TRAIN/subject*; do ln -sfn "$s" "$PFA/COTRAIN_ALLFM/$(basename "$s")"; done
echo "POOL_ALLFM n=$(ls $PFA/COTRAIN_ALLFM | wc -l)"
# 2. CoTAM 학습 (배치 4 → 21GB 필요)
drop_cache cotrain_allfm
run_gated 21000 "$Y/finetune_CoTAM.yaml" /tmp/ft_CoTAM.log CoTAM_TRAIN
drop_cache cotrain_allfm
CK=$PFA/runs/finetune_CoTAM/PreTrainedModels/SYNTH_CoTAM_physformer_Epoch9.pth
# 3. CoTAM 기존 축
if [ -f "$CK" ]; then for f in $Y/ev_CoTAM_*.yaml; do A=$(basename "$f" .yaml | sed 's/ev_CoTAM_//'); case $A in MMPD_B*) continue;; esac
  run_gated 6000 "$f" "/tmp/ev_CoTAM_${A}.log" "CoTAM_${A}"; done; else echo CoTAM_EVALS_SKIPPED; fi
# 4. MMPD 배치 평가
for k in 1 2 3 4 5 6; do
  run_gated 6000 "$Y/pre_base_MMPD_B$k.yaml" /tmp/ev_base_MMPD_B$k.log base_MMPD_B$k
  for M in CoTNH CoTA CoTAF CoTAF2 CoTAM; do
    [ "$M" = CoTAM ] && [ ! -f "$CK" ] && continue
    run_gated 6000 "$Y/ev_${M}_MMPD_B$k.yaml" /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k; done
  drop_cache mmpd_pf_B$k
  run_gated 6000 "$Y/pre_RhythmFormer_MMPD_B$k.yaml" /tmp/ev_RhythmFormer_MMPD_B$k.log RhythmFormer_MMPD_B$k
  run_gated 6000 "$Y/ev_RFA_MMPD_B$k.yaml" /tmp/ev_RFA_MMPD_B$k.log RFA_MMPD_B$k
  drop_cache mmpd_rf_B$k
  run_gated 6000 "$Y/pre_FactorizePhys_MMPD_B$k.yaml" /tmp/ev_FactorizePhys_MMPD_B$k.log FactorizePhys_MMPD_B$k
  run_gated 6000 "$Y/ev_FFA_MMPD_B$k.yaml" /tmp/ev_FFA_MMPD_B$k.log FFA_MMPD_B$k
  drop_cache mmpd_fp_B$k
  echo "$(date +%H:%M) BATCH${k}_DONE df=$(df -h / | awk 'NR==2{print $4}')"
done
echo CHAIN8_ALL_DONE; touch /tmp/chain8.done
