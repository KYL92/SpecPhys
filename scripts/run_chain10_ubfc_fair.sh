#!/bin/bash
# 체인 10 — UBFC 공정 비교표용 누락 평가 (PURE 사전학습 3백본 zero-shot, FactorizePhys 실노이즈 축, FFA, T2L REALTEST)
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RPPG_MP_QUOTA=6
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox; Y=../../_workspace/data/physformer/yaml19
for tag in $(cat $Y/order.txt); do
  if python main.py --config_file "$Y/ev_${tag}.yaml" > /tmp/ev_${tag}.log 2>&1; then echo "$(date +%H:%M) ${tag}_OK"; else echo "$(date +%H:%M) ${tag}_FAIL"; fi
done
cd /SPECPHYS_ROOT; python scripts/agg_model2.py PFpure RFpure FPpure FactorizePhys_arch RhythmFormer_arch FFA RFA T2L_s100 base CoTAF CoTAF2 > /tmp/agg_chain10.log 2>&1
echo CHAIN10_DONE; touch /tmp/chain10.done
