#!/bin/bash
# 체인 2 — ablation(압축만 CoTNC / 파형만 CoTNW) + 실분포 고조파 CoTNH2.
# 체인 1(run_chain_gated.sh)이 끝날 때까지 대기 후, GPU 게이트로 순차 실행.
set -u
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox
Y10=../../_workspace/data/physformer/yaml10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 체인 1 종료 대기
while pgrep -f "run_chain_gated.sh" > /dev/null; do sleep 120; done
echo CHAIN1_FINISHED_DETECTED

wait_gpu() {
  local need=$1 waited=0
  while true; do
    local used total free
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    free=$((total - used))
    [ "$free" -ge "$need" ] && { echo "[gpu] free=${free}MiB >= ${need}MiB (waited ${waited}s)"; return 0; }
    sleep 60; waited=$((waited + 60))
  done
}

run_gated() {
  for attempt in 1 2; do
    wait_gpu "$1"
    if python main.py --config_file "$2" > "$3" 2>&1; then echo "${4}_OK"; return 0; fi
    if grep -q "OutOfMemoryError\|CUDA out of memory\|cudaErrorMemoryAllocation" "$3"; then
      echo "${4}_OOM_RETRY_${attempt}"; sleep 120
    else
      echo "${4}_FAIL"; return 1
    fi
  done
  echo "${4}_FAIL"; return 1
}

for N in CoTNC CoTNW CoTNH2; do
  run_gated 19500 "$Y10/finetune_${N}.yaml" "/tmp/ft_${N}.log" "${N}_TRAIN"
  if ls ../../_workspace/data/physformer/runs/finetune_${N}/PreTrainedModels/*Epoch9.pth >/dev/null 2>&1; then
    for f in $Y10/ev_${N}_*.yaml; do
      A=$(basename "$f" .yaml | sed "s/ev_${N}_//")
      run_gated 6000 "$f" "/tmp/ev_${N}_${A}.log" "${N}_${A}"
    done
  else
    echo "${N}_EVALS_SKIPPED_NO_CHECKPOINT"
  fi
done
echo CHAIN2_ALL_DONE
