#!/bin/bash
# 체인 8c — 라벨 수정(RPPG_HR_LABEL=cumsum) 재학습: CoTAF_L(풀 ALLF) · CoTAM_L(풀 ALLFM) → 전 축 평가 → chain8b 종료 후 MMPD 6배치
set -u
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RPPG_MP_QUOTA=4 RPPG_HR_LABEL=cumsum
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox
Y=../../_workspace/data/physformer/yaml17; PFA=/SPECPHYS_ROOT/_workspace/data/physformer
wait_gpu() { local need=$1; while true; do local used total; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits); [ $((total-used)) -ge "$need" ] && return 0; sleep 60; done; }
run_gated() { for attempt in 1 2; do wait_gpu "$1"
    if python main.py --config_file "$2" > "$3" 2>&1; then echo "$(date +%H:%M) ${4}_OK"; return 0; fi
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$3"; then echo "$(date +%H:%M) ${4}_OOM_RETRY"; sleep 300
    else echo "$(date +%H:%M) ${4}_FAIL"; return 1; fi; done; return 1; }
drop_cache() { rm -rf "$PFA/cache/$1" "$PFA/cache/DataFileLists/${1}_0.0_1.0.csv"; }
# chain8b의 CoTAM 학습(cotrain_allfm 캐시 사용)과 충돌 방지: 8b 학습 종료 대기
until grep -q "CoTAM_TRAIN_OK\|CoTAM_NO_CKPT\|CoTAM_TRAIN_FAIL" /tmp/chain8b.log 2>/dev/null || [ -f /tmp/chain8b.done ]; do sleep 60; done
for M in CoTAF_L CoTAM_L; do
  case $M in CoTAF_L) C=cotrain_allf;; *) C=cotrain_allfm;; esac
  CK=$PFA/runs/finetune_${M}/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  if [ ! -f "$CK" ]; then drop_cache $C; run_gated 21000 "$Y/finetune_${M}.yaml" /tmp/ft_${M}.log ${M}_TRAIN; drop_cache $C; fi
  [ -f "$CK" ] || { echo ${M}_NO_CKPT; continue; }
  for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*) continue;; esac
    run_gated 6000 "$f" "/tmp/ev_${M}_${A}.log" "${M}_${A}"; done
done
until [ -f /tmp/chain8b.done ]; do sleep 120; done
for k in 1 2 3 4 5 6; do
  run_gated 6000 "$Y/ev_CoTAF_L_MMPD_B$k.yaml" /tmp/ev_CoTAF_L_MMPD_B$k.log CoTAF_L_MMPD_B$k
  run_gated 6000 "$Y/ev_CoTAM_L_MMPD_B$k.yaml" /tmp/ev_CoTAM_L_MMPD_B$k.log CoTAM_L_MMPD_B$k
  drop_cache mmpd_pfl_B$k; echo "$(date +%H:%M) LBATCH${k}_DONE"
done
echo CHAIN8C_ALL_DONE; touch /tmp/chain8c.done
