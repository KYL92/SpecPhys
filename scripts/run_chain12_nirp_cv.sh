#!/bin/bash
# 체인 12 — MR-NIRP 피험자 분리 3-fold CV: CoTANcv{k}_L (CoTAF 풀 + MR-NIRP 12~13명 3조건), cumsum 라벨 → NIRP 3조건 + 주요 축 평가
set -u; export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RPPG_MP_QUOTA=6 RPPG_HR_LABEL=cumsum
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox; Y=../../_workspace/data/physformer/yaml21; PFA=/SPECPHYS_ROOT/_workspace/data/physformer
wait_gpu() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits); [ $((t-u)) -ge "$1" ] && return 0; sleep 60; done; }
run_gated() { wait_gpu "$1"; if python main.py --config_file "$2" > "$3" 2>&1; then echo "$(date +%H:%M) ${4}_OK"; else echo "$(date +%H:%M) ${4}_FAIL"; fi; }
drop_cache() { rm -rf "$PFA/cache/$1" "$PFA/cache/DataFileLists/${1}_0.0_1.0.csv"; }
for k in 1 2 3; do M=CoTANcv${k}_L; CK=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  if [ ! -f "$CK" ]; then drop_cache cotrain_allfn_f$k; run_gated 21000 "$Y/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN; drop_cache cotrain_allfn_f$k; find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete; fi
  [ -f "$CK" ] || { echo ${M}_NO_CKPT; continue; }
  for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
  echo "$(date +%H:%M) FOLD${k}_DONE"
done
cd /SPECPHYS_ROOT; python scripts/agg_model2.py CoTANcv1_L CoTANcv2_L CoTANcv3_L > /tmp/agg_chain12.log 2>&1
echo CHAIN12_DONE; touch /tmp/chain12.done
