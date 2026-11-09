#!/bin/bash
# 체인 23 — 디지털 트윈 v1: 가상얼굴 32클립을 실 PPG 파형 뱅크 + 정반사 분리로 재렌더 + 코덱 스윕 사본 10(mpeg4 q6 ×4, h264 crf23 ×3, crf37 ×3) → 풀 203 유지 → CoTAF_T_L (기준 CoTAF-L)
set -u; export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RPPG_MP_QUOTA=6 RPPG_HR_LABEL=cumsum
cd /SPECPHYS_ROOT; PFA=/SPECPHYS_ROOT/_workspace/data/physformer
wait_gpu() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits); [ $((t-u)) -ge "$1" ] && return 0; sleep 60; done; }
run_gated() { wait_gpu "$1"; if python main.py --config_file "$2" > "$3" 2>&1; then echo "$(date +%H:%M) ${4}_OK"; else echo "$(date +%H:%M) ${4}_FAIL"; fi; }
drop_cache() { rm -rf "$PFA/cache/$1" "$PFA/cache/DataFileLists/${1}_0.0_1.0.csv"; }
for c in train20h train6hi train6lo; do U=$(echo $c | tr a-z A-Z); rm -rf $PFA/${U}_T; python generator/build_synth_dataset.py --config $PFA/configs/${c}_t.json > /tmp/build_${c}_t.log 2>&1 || { echo BUILD_${c}_FAIL; exit 1; }; done; echo TWIN_BUILD_DONE
rm -rf $PFA/CMP_SYN_T; mkdir -p $PFA/CMP_SYN_T
i=0; for k in 1 3 5 7 9 11 13 15 17 19; do i=$((i+1)); d=$PFA/CMP_SYN_T/subject$((600+i)); mkdir -p $d
  case $i in 1|2|3|4) ffmpeg -v error -y -i $PFA/TRAIN20H_T/subject$k/vid.avi -c:v mpeg4 -q:v 6 $d/vid.avi;; 5|6|7) ffmpeg -v error -y -i $PFA/TRAIN20H_T/subject$k/vid.avi -c:v libx264 -crf 23 -pix_fmt yuv420p $d/vid.avi;; *) ffmpeg -v error -y -i $PFA/TRAIN20H_T/subject$k/vid.avi -c:v libx264 -crf 37 -pix_fmt yuv420p $d/vid.avi;; esac
  cp $PFA/TRAIN20H_T/subject$k/ground_truth.txt $d/; done
echo "CMP_SYN_T n=$(ls $PFA/CMP_SYN_T | wc -l)"
P=ALLF_T; rm -rf $PFA/COTRAIN_$P; mkdir -p $PFA/COTRAIN_$P
for s in $PFA/COTRAIN_ALLF/subject*; do t=$(readlink -f $s); case $t in *TRAIN20H/*|*TRAIN6HI/*|*TRAIN6LO/*|*CMP_SYN/*) ;; *) ln -sfn "$t" "$PFA/COTRAIN_$P/$(basename $s)";; esac; done
for k in $(seq 1 20); do ln -sfn $PFA/TRAIN20H_T/subject$k $PFA/COTRAIN_$P/subject$((200+k)); done
for k in $(seq 1 6); do ln -sfn $PFA/TRAIN6HI_T/subject$k $PFA/COTRAIN_$P/subject$((220+k)); ln -sfn $PFA/TRAIN6LO_T/subject$k $PFA/COTRAIN_$P/subject$((230+k)); done
for k in $(seq 1 10); do ln -sfn $PFA/CMP_SYN_T/subject$((600+k)) $PFA/COTRAIN_$P/subject$((600+k)); done
echo "POOL_$P n=$(ls $PFA/COTRAIN_$P | wc -l) broken=$(find $PFA/COTRAIN_$P -xtype l | wc -l)"
M=CoTAF_T_L; cd repos/rPPG-Toolbox; Y=../../_workspace/data/physformer/yaml32
drop_cache cotrain_allf_t; run_gated 21000 "$Y/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN; drop_cache cotrain_allf_t
find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*) continue;; esac; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
for k in 1 2 3 4 5 6; do run_gated 6000 "$Y/ev_${M}_MMPD_B$k.yaml" /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k; drop_cache mmpd_pft_B$k; done
cd /SPECPHYS_ROOT; python scripts/agg_model2.py $M > /tmp/agg_chain23.log 2>&1; python scripts/agg_mmpd.py $M >> /tmp/agg_chain23.log 2>&1; echo "$(date +%H:%M) AGG_${M}_DONE"
echo CHAIN23_DONE; touch /tmp/chain23.done
