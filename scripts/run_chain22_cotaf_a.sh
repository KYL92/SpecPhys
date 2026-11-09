#!/bin/bash
# 체인 22 — CoTAF-A: 채택 코퍼스의 가상얼굴 42클립(TRAIN20H 20 + 6HI 6 + 6LO 6 + CMP_SYN 10)만 해부학 A(x,y)로 재렌더, 실 161클립 동일 → CoTAF_A_L (CoTAF 레시피 + A(x,y) 단일 요인)
set -u; cd /SPECPHYS_ROOT; PFA=/SPECPHYS_ROOT/_workspace/data/physformer
export PATH=/SPECPHYS_CONDA/envs/vface-gen/bin:$PATH
for c in train20h train6hi train6lo; do U=$(echo $c | tr a-z A-Z); python scripts/anatomical_amplitude_map.py --config $PFA/configs/$c.json $PFA/configs/${c}_a.json $PFA/amap_base/$U _workspace/data/physformer/${U}_A > /tmp/amap_$c.log 2>&1 || { echo AMAPCFG_${c}_FAIL; exit 1; }; done; echo AMAP_BASE_DONE
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RPPG_MP_QUOTA=6 RPPG_HR_LABEL=cumsum
for c in train20h train6hi train6lo; do U=$(echo $c | tr a-z A-Z); rm -rf $PFA/${U}_A; python generator/build_synth_dataset.py --config $PFA/configs/${c}_a.json > /tmp/build_${c}_a.log 2>&1 || { echo BUILD_${c}_FAIL; exit 1; }; done
rm -rf $PFA/CMP_SYN_A; python - <<PY
import sys; sys.path.insert(0,"scripts"); from compress_aug import compress_copy
for k in range(1,20,2): compress_copy(f"$PFA/TRAIN20H_A/subject{k}", f"$PFA/CMP_SYN_A/subject{600+(k+1)//2}", 6)
PY
echo "CMP_SYN_A n=$(ls $PFA/CMP_SYN_A | wc -l)"
P=ALLF_A; rm -rf $PFA/COTRAIN_$P; mkdir -p $PFA/COTRAIN_$P
for s in $PFA/COTRAIN_ALLF/subject*; do t=$(readlink -f $s); case $t in *TRAIN20H*|*TRAIN6HI*|*TRAIN6LO*|*CMP_SYN/*) ;; *) ln -sfn "$t" "$PFA/COTRAIN_$P/$(basename $s)";; esac; done
for k in $(seq 1 20); do ln -sfn $PFA/TRAIN20H_A/subject$k $PFA/COTRAIN_$P/subject$((200+k)); done
for k in $(seq 1 6); do ln -sfn $PFA/TRAIN6HI_A/subject$k $PFA/COTRAIN_$P/subject$((220+k)); ln -sfn $PFA/TRAIN6LO_A/subject$k $PFA/COTRAIN_$P/subject$((230+k)); done
for k in $(seq 1 10); do ln -sfn $PFA/CMP_SYN_A/subject$((600+k)) $PFA/COTRAIN_$P/subject$((600+k)); done
echo "POOL_$P n=$(ls $PFA/COTRAIN_$P | wc -l) broken=$(find $PFA/COTRAIN_$P -xtype l | wc -l)"
until [ -f /tmp/chain21.done ]; do sleep 60; done
wait_gpu() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits); [ $((t-u)) -ge "$1" ] && return 0; sleep 60; done; }
run_gated() { wait_gpu "$1"; if python main.py --config_file "$2" > "$3" 2>&1; then echo "$(date +%H:%M) ${4}_OK"; else echo "$(date +%H:%M) ${4}_FAIL"; fi; }
drop_cache() { rm -rf "$PFA/cache/$1" "$PFA/cache/DataFileLists/${1}_0.0_1.0.csv"; }
M=CoTAF_A_L; cd repos/rPPG-Toolbox; Y=../../_workspace/data/physformer/yaml31
drop_cache cotrain_allf_a; run_gated 21000 "$Y/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN; drop_cache cotrain_allf_a
find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*) continue;; esac; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
for k in 1 2 3 4 5 6; do run_gated 6000 "$Y/ev_${M}_MMPD_B$k.yaml" /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k; drop_cache mmpd_pfa_B$k; done
cd /SPECPHYS_ROOT; python scripts/agg_model2.py $M > /tmp/agg_chain22.log 2>&1; python scripts/agg_mmpd.py $M >> /tmp/agg_chain22.log 2>&1; echo "$(date +%H:%M) AGG_${M}_DONE"
echo CHAIN22_DONE; touch /tmp/chain22.done
