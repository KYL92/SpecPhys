#!/bin/bash
# 체인 7 — 다음 반복: CoTAF2(운전 스트레스 합성 추가, 209클립) + RFA(RhythmFormer 백본에 동일 코퍼스)
set -u
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox
PF=../../_workspace/data/physformer
Y=$PF/yaml16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 체인 6b(FactorizePhys 재실행)까지 종료 대기
until [ -f /tmp/chain6b.done ]; do sleep 120; done
echo CHAIN6X_FINISHED_DETECTED
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

# 1) CoTAF2 학습 + 13축 평가
run_gated 12000 "$Y/finetune_CoTAF2.yaml" /tmp/ft_CoTAF2.log CoTAF2_TRAIN
if ls $PF/runs/finetune_CoTAF2/PreTrainedModels/*Epoch9.pth >/dev/null 2>&1; then
  for f in $Y/ev_CoTAF2_*.yaml; do
    A=$(basename "$f" .yaml | sed "s/ev_CoTAF2_//")
    run_gated 6000 "$f" "/tmp/ev_CoTAF2_${A}.log" "CoTAF2_${A}"
  done
else echo CoTAF2_EVALS_SKIPPED; fi

# 2) RhythmFormer 미세조정(RFA) + 7축 평가
run_gated 12000 "$Y/finetune_RFA.yaml" /tmp/ft_RFA.log RFA_TRAIN
if ls $PF/runs/finetune_RFA/PreTrainedModels/*Epoch9.pth >/dev/null 2>&1; then
  for f in $Y/ev_RFA_*.yaml; do
    A=$(basename "$f" .yaml | sed "s/ev_RFA_//")
    run_gated 6000 "$f" "/tmp/ev_RFA_${A}.log" "RFA_${A}"
  done
else echo RFA_EVALS_SKIPPED; fi
echo CHAIN7_ALL_DONE
