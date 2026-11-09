#!/bin/bash
set -u
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox
PF=../../_workspace/data/physformer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -f /tmp/chain6b.done
echo CHAIN6_FINISHED_DETECTED
# 빈 캐시/파일리스트 정리 (실패 흔적)
for T in MCD_B MCD_A DLCN NIRP_STILL NIRP_SMALL NIRP_LARGE PURE; do
  rm -rf $PF/cache/arch_FactorizePhys_${T} $PF/cache/DataFileLists/arch_FactorizePhys_${T}_0.0_1.0.csv
done
wait_gpu() { local need=$1; while true; do
  local used total free
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  free=$((total-used)); [ "$free" -ge "$need" ] && return 0; sleep 60; done; }
for T in MCD_B MCD_A DLCN NIRP_STILL NIRP_SMALL NIRP_LARGE PURE; do
  wait_gpu 6000
  if python main.py --config_file "$PF/yaml15/ev_FactorizePhys_${T}.yaml" > "/tmp/ev_FactorizePhys_${T}.log" 2>&1; then
    echo "FactorizePhys_${T}_OK"
  else
    echo "FactorizePhys_${T}_FAIL"
  fi
done
echo CHAIN6B_ALL_DONE; touch /tmp/chain6b.done
