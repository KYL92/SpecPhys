#!/bin/bash
# GPU 게이트 체인 — 공유 머신에서 타 사용자 GPU 점유 시 여유 메모리가 확보될 때까지
# 대기한 뒤 각 단계를 실행한다. (평가 batch1 ≈5GB, 학습 batch4 ≈19GB 필요)
set -u
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox
Y8=../../_workspace/data/physformer/yaml8
Y9=../../_workspace/data/physformer/yaml9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_gpu() { # $1 = 필요 여유 MiB
  local need=$1 waited=0
  while true; do
    local used total free
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    free=$((total - used))
    [ "$free" -ge "$need" ] && { echo "[gpu] free=${free}MiB >= ${need}MiB (waited ${waited}s)"; return 0; }
    sleep 60; waited=$((waited + 60))
    [ $((waited % 600)) -eq 0 ] && echo "[gpu] still waiting: free=${free}MiB < ${need}MiB (${waited}s)"
  done
}

run_gated() { # $1 need MiB, $2 config, $3 log, $4 marker — OOM이면 1회 재대기 후 재시도
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

# 1) PURE 제로샷 평가 (캐시 재사용, batch 1)
for M in base T2L_s100 CoTN CoTNH; do
  run_gated 6000 "$Y8/ev_${M}_PURE.yaml" "/tmp/ev_${M}_PURE.log" "${M}_PURE"
done
# 고전 방법(CPU) — GPU 게이트 불필요
python main.py --config_file "$Y8/ev_unsup_PURE.yaml" > /tmp/ev_unsup_PURE.log 2>&1 \
  && echo unsup_PURE_OK || echo unsup_PURE_FAIL
echo PURE_ALL_DONE

# 2) CoTNH-M 학습 (batch 4 ≈19GB — CoTNH와 동일 조건 유지)
run_gated 19500 "$Y9/finetune_CoTNHM.yaml" /tmp/ft_CoTNHM.log CoTNHM_TRAIN

# 3) CoTNH-M 9축 평가
if ls ../../_workspace/data/physformer/runs/finetune_CoTNHM/PreTrainedModels/*Epoch9.pth >/dev/null 2>&1; then
  for A in MCD_B MCD_A PURE REALTEST RTNI RTNC C3 C6 E3; do
    run_gated 6000 "$Y9/ev_CoTNHM_${A}.yaml" "/tmp/ev_CoTNHM_${A}.log" "CoTNHM_${A}"
  done
else
  echo CoTNHM_EVALS_SKIPPED_NO_CHECKPOINT
fi
echo CHAIN_ALL_DONE
