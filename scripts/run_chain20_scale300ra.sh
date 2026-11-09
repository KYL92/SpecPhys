#!/bin/bash
# 체인 20 — SCALE300-RA: 실 움직임(체인 19와 동일 프레임) + 해부학 기반 A(x,y)(얼굴 파싱, 사람별 랜덤 유효 맥동 공간) → SYN_S300RA; CoTAF_S300RA_L / CoTAF_S300RAM_L(+MMPD 1~6 실 라벨)
set -u; cd /SPECPHYS_ROOT; PFA=/SPECPHYS_ROOT/_workspace/data/physformer
until grep -q ANIMR_DONE /tmp/anim_real.log 2>/dev/null; do sleep 60; done
export PATH=/SPECPHYS_CONDA/envs/vface-gen/bin:$PATH
python scripts/anatomical_amplitude_map.py --batch _workspace/data/faces/scale300r_pairs.json _workspace/data/faces/scale300r_amap > /tmp/amap.log 2>&1 || { echo AMAP_FAIL; exit 1; }; echo "$(date +%H:%M) $(tail -1 /tmp/amap.log)"
until [ -f /tmp/chain19.done ]; do sleep 60; done
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RPPG_MP_QUOTA=6 RPPG_HR_LABEL=cumsum
wait_gpu() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits); [ $((t-u)) -ge "$1" ] && return 0; sleep 60; done; }
run_gated() { wait_gpu "$1"; if python main.py --config_file "$2" > "$3" 2>&1; then echo "$(date +%H:%M) ${4}_OK"; else echo "$(date +%H:%M) ${4}_FAIL"; fi; }
drop_cache() { rm -rf "$PFA/cache/$1" "$PFA/cache/DataFileLists/${1}_0.0_1.0.csv"; }
python scripts/make_syn_scale_config.py RA || { echo CONFIG_RA_FAIL; exit 1; }
rm -rf $PFA/SYN_S300RA; python generator/build_synth_dataset.py --config $PFA/configs/syn_s300ra.json > /tmp/build_syn_s300ra.log 2>&1 || { echo BUILD_RA_FAIL; exit 1; }
for M in CoTAF_S300RA_L CoTAF_S300RAM_L; do
  case $M in CoTAF_S300RA_L) P=ALLFS300RA; c=allfs300ra; e=pfs300ra; EXTRA="";; *) P=ALLFS300RAM; c=allfs300ram; e=pfs300ram; EXTRA="$PFA/MMPD_TRAIN/subject*";; esac
  rm -rf $PFA/COTRAIN_$P; mkdir -p $PFA/COTRAIN_$P; for s in $PFA/COTRAIN_ALLF/subject* $PFA/SYN_S300RA/subject* $EXTRA; do ln -sfn "$(readlink -f $s)" "$PFA/COTRAIN_$P/$(basename $s)"; done; echo "POOL_$P n=$(ls $PFA/COTRAIN_$P | wc -l)"
  cd repos/rPPG-Toolbox; Y=../../_workspace/data/physformer/yaml29
  drop_cache cotrain_$c; run_gated 21000 "$Y/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN; drop_cache cotrain_$c
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null
  for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*) continue;; esac; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
  for k in 1 2 3 4 5 6; do run_gated 6000 "$Y/ev_${M}_MMPD_B$k.yaml" /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k; drop_cache mmpd_${e}_B$k; done
  cd /SPECPHYS_ROOT; python scripts/agg_model2.py $M > /tmp/agg_chain20_$M.log 2>&1; python scripts/agg_mmpd.py $M >> /tmp/agg_chain20_$M.log 2>&1; echo "$(date +%H:%M) AGG_${M}_DONE"
done
echo CHAIN20_DONE; touch /tmp/chain20.done
