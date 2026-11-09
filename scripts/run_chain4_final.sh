#!/bin/bash
# 체인 4 — 최종 모델 CoTAF: 전체 데이터(UBFC+MCD+PURE 학습분할+합성) + 저심박(42~50 BPM)
# 고조파 클립까지 포함한 전 범위 풀(203클립). 체인 3 종료 후 자동 실행.
set -u
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox
PF=../../_workspace/data/physformer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while pgrep -f "run_chain3_full.sh" > /dev/null; do sleep 120; done
echo CHAIN3_FINISHED_DETECTED

wait_gpu() {
  local need=$1 waited=0
  while true; do
    local used total free
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    free=$((total - used))
    [ "$free" -ge "$need" ] && { echo "[gpu] free=${free}MiB (waited ${waited}s)"; return 0; }
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

# COTRAIN_ALLF 조립 = COTRAIN_ALL(197) + TRAIN6LO(6, subject231-236)
if [ ! -d "$PF/COTRAIN_ALL" ]; then echo COTRAIN_ALL_MISSING; exit 1; fi
rm -rf $PF/COTRAIN_ALLF; mkdir -p $PF/COTRAIN_ALLF
for s in $PF/COTRAIN_ALL/subject*; do ln -s $(readlink -f "$s") $PF/COTRAIN_ALLF/$(basename "$s"); done
for i in 1 2 3 4 5 6; do ln -s $(readlink -f $PF/TRAIN6LO/subject$i) $PF/COTRAIN_ALLF/subject$((230+i)); done
echo "COTRAIN_ALLF clips: $(ls $PF/COTRAIN_ALLF | wc -l)"

run_gated 19500 "$PF/yaml12/finetune_CoTAF.yaml" /tmp/ft_CoTAF.log CoTAF_TRAIN
if ls $PF/runs/finetune_CoTAF/PreTrainedModels/*Epoch9.pth >/dev/null 2>&1; then
  for f in $PF/yaml12/ev_CoTAF_*.yaml; do
    A=$(basename "$f" .yaml | sed "s/ev_CoTAF_//")
    run_gated 6000 "$f" "/tmp/ev_CoTAF_${A}.log" "CoTAF_${A}"
  done
else
  echo CoTAF_EVALS_SKIPPED_NO_CHECKPOINT
fi
echo CHAIN4_ALL_DONE
