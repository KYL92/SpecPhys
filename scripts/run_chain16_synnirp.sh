#!/bin/bash
# 체인 16 — 도메인 특화 합성 SYN_NIRP(MR-NIRP 1~6 측정 + 정체성) + CoTAF 풀 → CoTAF_DN_L
set -u; export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RPPG_MP_QUOTA=6 RPPG_HR_LABEL=cumsum
cd /SPECPHYS_ROOT; PFA=/SPECPHYS_ROOT/_workspace/data/physformer
until [ -s $PFA/../results/domain_stats_NIRP.json ] && grep -q LP_NIRP_DONE /tmp/lp_nirp.log; do sleep 30; done
for s in 1 2 3 4 5 6; do d=/SPECPHYS_ROOT/_workspace/data/faces/nirp_s${s}_frames; [ -d $d ] || { mkdir -p $d; ffmpeg -y -loglevel error -i /SPECPHYS_ROOT/_workspace/data/faces/animations_nirp/nirp_s$s--d6.mp4 -vf scale=512:512 $d/%06d.png; }; done; echo FRAMES_OK
python scripts/make_syn_domain_config.py NIRP || { echo CONFIG_FAIL; exit 1; }
rm -rf $PFA/SYN_NIRP; python generator/build_synth_dataset.py --config $PFA/configs/syn_nirp.json > /tmp/build_syn_nirp.log 2>&1 || { echo BUILD_FAIL; exit 1; }
mkdir -p $PFA/COTRAIN_ALLFDN; for s in $PFA/COTRAIN_ALLF/subject*; do ln -sfn "$(readlink -f $s)" "$PFA/COTRAIN_ALLFDN/$(basename $s)"; done; for s in $PFA/SYN_NIRP/subject*; do ln -sfn "$(readlink -f $s)" "$PFA/COTRAIN_ALLFDN/$(basename $s)"; done; echo "POOL_ALLFDN n=$(ls $PFA/COTRAIN_ALLFDN | wc -l)"
cd repos/rPPG-Toolbox; Y=../../_workspace/data/physformer/yaml25; M=CoTAF_DN_L
wait_gpu() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits); [ $((t-u)) -ge "$1" ] && return 0; sleep 60; done; }
run_gated() { wait_gpu "$1"; if python main.py --config_file "$2" > "$3" 2>&1; then echo "$(date +%H:%M) ${4}_OK"; else echo "$(date +%H:%M) ${4}_FAIL"; fi; }
drop_cache() { rm -rf "$PFA/cache/$1" "$PFA/cache/DataFileLists/${1}_0.0_1.0.csv"; }
drop_cache cotrain_allfdn; run_gated 21000 "$Y/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN; drop_cache cotrain_allfdn
find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*) continue;; esac; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
for k in 1 2 3 4 5 6; do run_gated 6000 "$Y/ev_${M}_MMPD_B$k.yaml" /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k; drop_cache mmpd_pfdn_B$k; done
cd /SPECPHYS_ROOT; python scripts/agg_model2.py $M > /tmp/agg_chain16.log 2>&1; python scripts/agg_mmpd.py $M >> /tmp/agg_chain16.log 2>&1
echo CHAIN16_DONE; touch /tmp/chain16.done
