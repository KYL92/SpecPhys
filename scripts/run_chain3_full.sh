#!/bin/bash
# 체인 3 — 통합 실행:
#  1) CoTNH-M (고심박+MCD in-domain) 학습 재시도(HR 클램프 패치 후) + 9축 평가
#  2) CoTA (전체 데이터: UBFC+MCD+PURE 학습분할+합성) 조립→학습→9축 평가
#  3) ablation: CoTNC(압축만) / CoTNW(파형만) / CoTNH2(실분포 고조파) 학습+평가
set -u
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox
PF=../../_workspace/data/physformer
Y9=$PF/yaml9; Y10=$PF/yaml10; Y11=$PF/yaml11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
train_and_eval() { # $1=name $2=finetune_yaml $3=eval_dir_glob_prefix
  run_gated 19500 "$2" "/tmp/ft_$1.log" "$1_TRAIN"
  if ls $PF/runs/finetune_$1/PreTrainedModels/*Epoch9.pth >/dev/null 2>&1; then
    for f in $3/ev_$1_*.yaml; do
      A=$(basename "$f" .yaml | sed "s/ev_$1_//")
      run_gated 6000 "$f" "/tmp/ev_$1_${A}.log" "$1_${A}"
    done
  else
    echo "$1_EVALS_SKIPPED_NO_CHECKPOINT"
  fi
}

# 1) CoTNH-M
train_and_eval CoTNHM "$Y9/finetune_CoTNHM.yaml" "$Y9"

# 2) CoTA — PURE 변환 완료 대기 후 조립
while pgrep -f convert_pure_train.py > /dev/null; do sleep 60; done
NP=$(ls -d $PF/PURE_TRAIN/subject* 2>/dev/null | wc -l)
echo "PURE_TRAIN clips: $NP"
if [ "$NP" -ge 30 ]; then
  rm -rf $PF/COTRAIN_ALL; mkdir -p $PF/COTRAIN_ALL
  for s in $PF/COTRAIN_HM/subject*; do ln -s $(readlink -f "$s") $PF/COTRAIN_ALL/$(basename "$s"); done
  for s in $PF/PURE_TRAIN/subject*; do ln -s $(readlink -f "$s") $PF/COTRAIN_ALL/$(basename "$s"); done
  echo "COTRAIN_ALL clips: $(ls $PF/COTRAIN_ALL | wc -l)"
  train_and_eval CoTA "$Y11/finetune_CoTA.yaml" "$Y11"
else
  echo CoTA_SKIPPED_PURE_TRAIN_INCOMPLETE
fi

# 3) ablations
for N in CoTNC CoTNW CoTNH2; do
  train_and_eval $N "$Y10/finetune_${N}.yaml" "$Y10"
done
echo CHAIN3_ALL_DONE
