#!/bin/bash
# 체인 14 — 합성만 학습: SYNALL_FT(UBFC 초기화, 10ep) · SYNALL_SCR(무작위 초기화, 30ep) → 13축 + MMPD 6배치(캐시 공유)
set -u; export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RPPG_MP_QUOTA=6 RPPG_HR_LABEL=cumsum
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox; Y=../../_workspace/data/physformer/yaml23; PFA=/SPECPHYS_ROOT/_workspace/data/physformer
wait_gpu() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits); [ $((t-u)) -ge "$1" ] && return 0; sleep 60; done; }
run_gated() { wait_gpu "$1"; if python main.py --config_file "$2" > "$3" 2>&1; then echo "$(date +%H:%M) ${4}_OK"; else echo "$(date +%H:%M) ${4}_FAIL"; fi; }
drop_cache() { rm -rf "$PFA/cache/$1" "$PFA/cache/DataFileLists/${1}_0.0_1.0.csv"; }
drop_cache cotrain_synall
for M in SYNALL_FT SYNALL_SCR; do run_gated 21000 "$Y/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" | sort -V | head -n -1 | xargs -r rm -f
  for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*) continue;; esac; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done; done
drop_cache cotrain_synall
for k in 1 2 3 4 5 6; do for M in SYNALL_FT SYNALL_SCR; do run_gated 6000 "$Y/ev_${M}_MMPD_B$k.yaml" /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k; done; drop_cache mmpd_pfs_B$k; done
cd /SPECPHYS_ROOT; python scripts/agg_model2.py SYNALL_FT SYNALL_SCR > /tmp/agg_chain14.log 2>&1; python scripts/agg_mmpd.py SYNALL_FT SYNALL_SCR >> /tmp/agg_chain14.log 2>&1
echo CHAIN14_DONE; touch /tmp/chain14.done
