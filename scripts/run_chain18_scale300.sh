#!/bin/bash
# 체인 18 — SCALE300: 300 층화 가상얼굴 합성(F 고정 주입 / P 피부톤 의존 주입) + CoTAF 풀 → CoTAF_S300F_L / CoTAF_S300P_L, 13축 + MMPD
set -u; export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RPPG_MP_QUOTA=6 RPPG_HR_LABEL=cumsum
cd /SPECPHYS_ROOT; PFA=/SPECPHYS_ROOT/_workspace/data/physformer
until grep -q ANIM_DONE /tmp/anim_scale300.log 2>/dev/null; do sleep 60; done
wait_gpu() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits); [ $((t-u)) -ge "$1" ] && return 0; sleep 60; done; }
run_gated() { wait_gpu "$1"; if python main.py --config_file "$2" > "$3" 2>&1; then echo "$(date +%H:%M) ${4}_OK"; else echo "$(date +%H:%M) ${4}_FAIL"; fi; }
drop_cache() { rm -rf "$PFA/cache/$1" "$PFA/cache/DataFileLists/${1}_0.0_1.0.csv"; }
for V in P F; do
  v=$(echo $V | tr A-Z a-z); M=CoTAF_S300${V}_L
  python scripts/make_syn_scale_config.py $V || { echo CONFIG_${V}_FAIL; exit 1; }
  rm -rf $PFA/SYN_S300$V; python generator/build_synth_dataset.py --config $PFA/configs/syn_s300$v.json > /tmp/build_syn_s300$v.log 2>&1 || { echo BUILD_${V}_FAIL; exit 1; }
  rm -rf $PFA/COTRAIN_ALLFS300$V; mkdir -p $PFA/COTRAIN_ALLFS300$V; for s in $PFA/COTRAIN_ALLF/subject* $PFA/SYN_S300$V/subject*; do ln -sfn "$(readlink -f $s)" "$PFA/COTRAIN_ALLFS300$V/$(basename $s)"; done; echo "POOL_S300$V n=$(ls $PFA/COTRAIN_ALLFS300$V | wc -l)"
  cd repos/rPPG-Toolbox; Y=../../_workspace/data/physformer/yaml27
  drop_cache cotrain_allfs300$v; run_gated 21000 "$Y/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN; drop_cache cotrain_allfs300$v
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
  for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*) continue;; esac; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
  for k in 1 2 3 4 5 6; do run_gated 6000 "$Y/ev_${M}_MMPD_B$k.yaml" /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k; drop_cache mmpd_pfs300${v}_B$k; done
  cd /SPECPHYS_ROOT; python scripts/agg_model2.py $M > /tmp/agg_chain18_$v.log 2>&1; python scripts/agg_mmpd.py $M >> /tmp/agg_chain18_$v.log 2>&1; echo "$(date +%H:%M) AGG_${V}_DONE"
done
echo CHAIN18_DONE; touch /tmp/chain18.done
