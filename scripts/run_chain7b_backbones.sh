#!/bin/bash
# 체인 7b — 백본 직교성: RFA(RhythmFormer)·FFA(FactorizePhys)에 동일 코퍼스 적용
set -u
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox
PF=../../_workspace/data/physformer
Y=$PF/yaml16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
wait_gpu() { local need=$1; while true; do
  local used total free
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  free=$((total-used)); [ "$free" -ge "$need" ] && return 0; sleep 60; done; }
run_gated() {
  for attempt in 1 2; do
    wait_gpu "$1"
    if python main.py --config_file "$2" > "$3" 2>&1; then echo "${4}_OK"; return 0; fi
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$3"; then echo "${4}_OOM_RETRY"; sleep 120
    else echo "${4}_FAIL"; return 1; fi
  done; echo "${4}_FAIL"; return 1; }

# 미완 RFA 캐시 잔재 정리
rm -rf $PF/cache/cotrain_allf2_rf $PF/cache/DataFileLists/cotrain_allf2_rf_0.0_1.0.csv

for M in RFA FFA; do
  run_gated 12000 "$Y/finetune_${M}.yaml" "/tmp/ft_${M}.log" "${M}_TRAIN"
  CK=$(ls $PF/runs/finetune_${M}/PreTrainedModels/*Epoch9.pth 2>/dev/null | head -1)
  if [ -n "$CK" ]; then
    for f in $Y/ev_${M}_*.yaml; do
      A=$(basename "$f" .yaml | sed "s/ev_${M}_//")
      run_gated 6000 "$f" "/tmp/ev_${M}_${A}.log" "${M}_${A}"
    done
  else echo "${M}_EVALS_SKIPPED"; fi
  # 학습 캐시는 즉시 정리 (디스크 보호)
  rm -rf $PF/cache/cotrain_allf2_rf $PF/cache/cotrain_allf2_fp
done
echo CHAIN7B_ALL_DONE; touch /tmp/chain7b.done
