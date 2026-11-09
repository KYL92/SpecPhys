#!/bin/bash
# 체인 6 — 최신 아키텍처(UBFC 사전학습 공식 가중치) 실측 비교:
# RhythmFormer/FactorizePhys/EfficientPhys/PhysNet × {MCD_B,MCD_A,DLCN,NIRP×3,PURE}
set -u
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox
PF=../../_workspace/data/physformer
Y=$PF/yaml15
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
wait_gpu() {
  local need=$1
  while true; do
    local used total free
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    free=$((total - used)); [ "$free" -ge "$need" ] && return 0; sleep 60
  done
}
for A in RhythmFormer FactorizePhys EfficientPhys PhysNet; do
  for T in MCD_B MCD_A DLCN NIRP_STILL NIRP_SMALL NIRP_LARGE PURE; do
    for attempt in 1 2; do
      wait_gpu 6000
      if python main.py --config_file "$Y/ev_${A}_${T}.yaml" > "/tmp/ev_${A}_${T}.log" 2>&1; then
        echo "${A}_${T}_OK"; break
      fi
      if grep -q "OutOfMemoryError\|CUDA out of memory" "/tmp/ev_${A}_${T}.log"; then
        echo "${A}_${T}_OOM_RETRY"; sleep 120
      else
        echo "${A}_${T}_FAIL"; break
      fi
    done
  done
done
echo CHAIN6_ALL_DONE
