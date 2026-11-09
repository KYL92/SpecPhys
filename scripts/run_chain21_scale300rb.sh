#!/bin/bash
# 체인 21 — 희석 가설 검정: S300R 풀 + 실 클립 93개 복제(실 비율 23% → 38%) → CoTAF_S300RB_L. 합성 내용이 아니라 실 비율이 원인이면 회복해야 함
set -u; export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RPPG_MP_QUOTA=6 RPPG_HR_LABEL=cumsum
cd /SPECPHYS_ROOT; PFA=/SPECPHYS_ROOT/_workspace/data/physformer
until [ -f /tmp/chain20.done ]; do sleep 60; done
wait_gpu() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits); [ $((t-u)) -ge "$1" ] && return 0; sleep 60; done; }
run_gated() { wait_gpu "$1"; if python main.py --config_file "$2" > "$3" 2>&1; then echo "$(date +%H:%M) ${4}_OK"; else echo "$(date +%H:%M) ${4}_FAIL"; fi; }
drop_cache() { rm -rf "$PFA/cache/$1" "$PFA/cache/DataFileLists/${1}_0.0_1.0.csv"; }
M=CoTAF_S300RB_L; P=ALLFS300RB; rm -rf $PFA/COTRAIN_$P; mkdir -p $PFA/COTRAIN_$P
for s in $PFA/COTRAIN_ALLF/subject* $PFA/SYN_S300R/subject*; do ln -sfn "$(readlink -f $s)" "$PFA/COTRAIN_$P/$(basename $s)"; done
for s in $PFA/COTRAIN_ALLF/subject*; do id=${s##*subject}; if [ $id -lt 100 ] || { [ $id -ge 700 ] && [ $id -lt 800 ]; } || { [ $id -ge 90000 ] && [ $id -lt 91000 ]; }; then ln -sfn "$(readlink -f $s)" "$PFA/COTRAIN_$P/subject$((80000+id))"; fi; done
echo "POOL_$P n=$(ls $PFA/COTRAIN_$P | wc -l)"
cd repos/rPPG-Toolbox; Y=../../_workspace/data/physformer/yaml30
drop_cache cotrain_allfs300rb; run_gated 21000 "$Y/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN; drop_cache cotrain_allfs300rb
find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*) continue;; esac; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
for k in 1 2 3 4 5 6; do run_gated 6000 "$Y/ev_${M}_MMPD_B$k.yaml" /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k; drop_cache mmpd_pfs300rb_B$k; done
cd /SPECPHYS_ROOT; python scripts/agg_model2.py $M > /tmp/agg_chain21.log 2>&1; python scripts/agg_mmpd.py $M >> /tmp/agg_chain21.log 2>&1; echo "$(date +%H:%M) AGG_${M}_DONE"
echo CHAIN21_DONE; touch /tmp/chain21.done
