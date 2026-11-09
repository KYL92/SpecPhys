#!/bin/bash
# 체인 5 — 신규 데이터셋(DLCN, MR-NIRP Car 3조건) 제로샷 평가.
# 변환 완료를 기다린 뒤, 세트별로 base(전처리 겸)→나머지 4모델→고전(POS/CHROM) 순 실행.
set -u
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox
PF=../../_workspace/data/physformer
Y=$PF/yaml14
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while pgrep -f "convert_nirp.py|convert_dlcn.py" > /dev/null; do sleep 120; done
echo CONVERTERS_FINISHED

wait_gpu() {
  local need=$1
  while true; do
    local used total free
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    free=$((total - used))
    [ "$free" -ge "$need" ] && return 0
    sleep 60
  done
}
run_gated() {
  for attempt in 1 2; do
    wait_gpu "$1"
    if python main.py --config_file "$2" > "$3" 2>&1; then echo "${4}_OK"; return 0; fi
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$3"; then echo "${4}_OOM_RETRY"; sleep 120
    else echo "${4}_FAIL"; return 1; fi
  done
  echo "${4}_FAIL"; return 1
}

for T in DLCN NIRP_STILL NIRP_SMALL NIRP_LARGE; do
  DIRN=$T; [ "$T" = "DLCN" ] && DIRN=DLCN_TEST
  NCLIP=$(ls -d $PF/$DIRN/subject* 2>/dev/null | wc -l)
  echo "SET $T clips=$NCLIP"
  if [ "$NCLIP" -lt 5 ]; then echo "${T}_SKIPPED_TOO_FEW"; continue; fi
  for M in base CoTNH CoTNHM CoTA CoTAF; do
    run_gated 6000 "$Y/ev_${M}_${T}.yaml" "/tmp/ev_${M}_${T}.log" "${M}_${T}"
  done
  python main.py --config_file "$Y/ev_unsup_${T}.yaml" > "/tmp/ev_unsup_${T}.log" 2>&1 \
    && echo "unsup_${T}_OK" || echo "unsup_${T}_FAIL"
done
echo CHAIN5_ALL_DONE
