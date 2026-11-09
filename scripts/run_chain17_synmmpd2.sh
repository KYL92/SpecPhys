#!/bin/bash
# 체인 17 — 도메인 특화 합성 재설계(2요인 분리): SYN_MMPD_A(외형만) / SYN_MMPD_B(측정 물리+피부 색 통계) 각 24클립 + CoTAF 풀 → CoTAF_DM2A_L / CoTAF_DM2B_L
set -u; export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RPPG_MP_QUOTA=6 RPPG_HR_LABEL=cumsum
cd /SPECPHYS_ROOT; PFA=/SPECPHYS_ROOT/_workspace/data/physformer
wait_gpu() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits); [ $((t-u)) -ge "$1" ] && return 0; sleep 60; done; }
run_gated() { wait_gpu "$1"; if python main.py --config_file "$2" > "$3" 2>&1; then echo "$(date +%H:%M) ${4}_OK"; else echo "$(date +%H:%M) ${4}_FAIL"; fi; }
drop_cache() { rm -rf "$PFA/cache/$1" "$PFA/cache/DataFileLists/${1}_0.0_1.0.csv"; }
for V in A B; do
  v=$(echo $V | tr A-Z a-z); M=CoTAF_DM2${V}_L
  rm -rf $PFA/SYN_MMPD_$V; python generator/build_synth_dataset.py --config $PFA/configs/syn_mmpd_$v.json > /tmp/build_syn_mmpd_$v.log 2>&1 || { echo BUILD_${V}_FAIL; exit 1; }
  rm -rf $PFA/COTRAIN_ALLFDM2$V; mkdir -p $PFA/COTRAIN_ALLFDM2$V; for s in $PFA/COTRAIN_ALLF/subject* $PFA/SYN_MMPD_$V/subject*; do ln -sfn "$(readlink -f $s)" "$PFA/COTRAIN_ALLFDM2$V/$(basename $s)"; done; echo "POOL_ALLFDM2$V n=$(ls $PFA/COTRAIN_ALLFDM2$V | wc -l)"
  cd repos/rPPG-Toolbox; Y=../../_workspace/data/physformer/yaml26
  drop_cache cotrain_allfdm2$v; run_gated 21000 "$Y/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN; drop_cache cotrain_allfdm2$v
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
  for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*) continue;; esac; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
  for k in 1 2 3 4 5 6; do run_gated 6000 "$Y/ev_${M}_MMPD_B$k.yaml" /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k; drop_cache mmpd_pfdm2${v}_B$k; done
  cd /SPECPHYS_ROOT; python scripts/agg_model2.py $M > /tmp/agg_chain17_$v.log 2>&1; python scripts/agg_mmpd.py $M >> /tmp/agg_chain17_$v.log 2>&1; echo "$(date +%H:%M) AGG_${V}_DONE"
done
echo CHAIN17_DONE; touch /tmp/chain17.done
