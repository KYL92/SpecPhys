#!/bin/bash
# 체인 8b — CoTAM 학습(재시도) → 기존 축 평가 → 메인 체인(chain8) 종료 후 누락된 MMPD 배치 CoTAM 보충 평가
set -u
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RPPG_MP_QUOTA=4
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox
Y=../../_workspace/data/physformer/yaml17; PFA=/SPECPHYS_ROOT/_workspace/data/physformer
wait_gpu() { local need=$1; while true; do
  local used total; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  [ $((total-used)) -ge "$need" ] && return 0; sleep 60; done; }
run_gated() { for attempt in 1 2; do wait_gpu "$1"
    if python main.py --config_file "$2" > "$3" 2>&1; then echo "$(date +%H:%M) ${4}_OK"; return 0; fi
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$3"; then echo "$(date +%H:%M) ${4}_OOM_RETRY"; sleep 300
    else echo "$(date +%H:%M) ${4}_FAIL"; return 1; fi; done; return 1; }
drop_cache() { rm -rf "$PFA/cache/$1" "$PFA/cache/DataFileLists/${1}_0.0_1.0.csv"; }
CK=$PFA/runs/finetune_CoTAM/PreTrainedModels/SYNTH_CoTAM_physformer_Epoch9.pth
if [ ! -f "$CK" ]; then
  drop_cache cotrain_allfm
  run_gated 21000 "$Y/finetune_CoTAM.yaml" /tmp/ft_CoTAM.log CoTAM_TRAIN
  drop_cache cotrain_allfm
fi
[ -f "$CK" ] || { echo CoTAM_NO_CKPT; touch /tmp/chain8b.done; exit 1; }
for f in $Y/ev_CoTAM_*.yaml; do A=$(basename "$f" .yaml | sed 's/ev_CoTAM_//'); case $A in MMPD_B*) continue;; esac
  run_gated 6000 "$f" "/tmp/ev_CoTAM_${A}.log" "CoTAM_${A}"; done
until [ -f /tmp/chain8.done ]; do sleep 120; done
for k in 1 2 3 4 5 6; do
  if ls $PFA/runs/ev_CoTAM_MMPD_B$k/*/saved_test_outputs/*.pickle >/dev/null 2>&1; then echo "CoTAM_MMPD_B${k}_EXISTS"; continue; fi
  run_gated 6000 "$Y/pre_CoTAM_MMPD_B$k.yaml" /tmp/ev_CoTAM_MMPD_B$k.log CoTAM_MMPD_B$k
  drop_cache mmpd_pfm_B$k
done
echo CHAIN8B_ALL_DONE; touch /tmp/chain8b.done
