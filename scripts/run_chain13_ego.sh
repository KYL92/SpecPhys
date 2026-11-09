#!/bin/bash
# 체인 13 — CoTAF3_L: CoTAF 풀 + EGO6(카메라 ego-motion 합성 6클립), cumsum 라벨
set -u; export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RPPG_MP_QUOTA=6 RPPG_HR_LABEL=cumsum
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox; Y=../../_workspace/data/physformer/yaml22; PFA=/SPECPHYS_ROOT/_workspace/data/physformer; M=CoTAF3_L
wait_gpu() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits); [ $((t-u)) -ge "$1" ] && return 0; sleep 60; done; }
run_gated() { wait_gpu "$1"; if python main.py --config_file "$2" > "$3" 2>&1; then echo "$(date +%H:%M) ${4}_OK"; else echo "$(date +%H:%M) ${4}_FAIL"; fi; }
drop_cache() { rm -rf "$PFA/cache/$1" "$PFA/cache/DataFileLists/${1}_0.0_1.0.csv"; }
CK=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
if [ ! -f "$CK" ]; then drop_cache cotrain_allfe; run_gated 21000 "$Y/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN; drop_cache cotrain_allfe; find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete; fi
[ -f "$CK" ] || { echo NO_CKPT; touch /tmp/chain13.done; exit 1; }
for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*) continue;; esac; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
for k in 1 2 3 4 5 6; do run_gated 6000 "$Y/ev_${M}_MMPD_B$k.yaml" /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k; drop_cache mmpd_pfe_B$k; done
cd /SPECPHYS_ROOT; python scripts/agg_model2.py $M > /tmp/agg_chain13.log 2>&1; python scripts/agg_mmpd.py $M >> /tmp/agg_chain13.log 2>&1
echo CHAIN11_DONE; touch /tmp/chain13.done
