#!/bin/bash
# 체인 9 — PhysFormer 계열 10모델을 RPPG_HR_LABEL=cumsum 라벨로 재학습(batch 4 통일) → 13축 평가 → MMPD 6배치 전 모델 평가
set -u
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RPPG_MP_QUOTA=6 RPPG_HR_LABEL=cumsum
cd /SPECPHYS_ROOT/repos/rPPG-Toolbox
Y=../../_workspace/data/physformer/yaml18; PFA=/SPECPHYS_ROOT/_workspace/data/physformer
MODELS="CoTN CoTNH CoTNHM CoTA CoTAF2 CoTNC CoTNW CoTNH2 CoTB CoTBH"
wait_gpu() { local need=$1; while true; do local used total; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits); [ $((total-used)) -ge "$need" ] && return 0; sleep 60; done; }
run_gated() { for attempt in 1 2; do wait_gpu "$1"
    if python main.py --config_file "$2" > "$3" 2>&1; then echo "$(date +%m-%d\ %H:%M) ${4}_OK"; return 0; fi
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$3"; then echo "$(date +%H:%M) ${4}_OOM_RETRY"; sleep 300
    else echo "$(date +%m-%d\ %H:%M) ${4}_FAIL"; return 1; fi; done; return 1; }
drop_cache() { rm -rf "$PFA/cache/$1" "$PFA/cache/DataFileLists/${1}_0.0_1.0.csv"; }
cache_of() { grep -m1 EXP_DATA_NAME "$Y/finetune_${1}_L.yaml" | awk '{print $2}' | tr -d '"'; }
# Phase A
for M in $MODELS; do L=${M}_L; C=$(cache_of $M); CK=$PFA/runs/finetune_${L}/PreTrainedModels/SYNTH_${L}_physformer_Epoch9.pth
  if [ ! -f "$CK" ]; then drop_cache $C
    if [ "$M" = CoTBH ]; then export PF_HARM_LOSS=1 PF_HARM_W=1.0 PF_HARM_TAU=0.3; else unset PF_HARM_LOSS PF_HARM_W PF_HARM_TAU; fi
    run_gated 21000 "$Y/finetune_${L}.yaml" /tmp/ft_${L}.log ${L}_TRAIN; unset PF_HARM_LOSS PF_HARM_W PF_HARM_TAU; drop_cache $C
    find $PFA/runs/finetune_${L}/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; fi
  [ -f "$CK" ] || { echo ${L}_NO_CKPT; continue; }
  for f in $Y/ev_${L}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${L}_//"); case $A in MMPD_B*) continue;; esac
    run_gated 6000 "$f" "/tmp/ev_${L}_${A}.log" "${L}_${A}"; done
  echo "$(date +%m-%d\ %H:%M) MODEL_${L}_DONE df=$(df -h / | awk 'NR==2{print $4}')"
done
# Phase B: MMPD 6배치, 배치당 캐시 1회
for k in 1 2 3 4 5 6; do
  run_gated 6000 "$Y/pre_CoTN_L_MMPD_B$k.yaml" /tmp/ev_CoTN_L_MMPD_B$k.log CoTN_L_MMPD_B$k
  for M in $MODELS; do [ "$M" = CoTN ] && continue; L=${M}_L
    [ -f $PFA/runs/finetune_${L}/PreTrainedModels/SYNTH_${L}_physformer_Epoch9.pth ] || continue
    run_gated 6000 "$Y/ev_${L}_MMPD_B$k.yaml" /tmp/ev_${L}_MMPD_B$k.log ${L}_MMPD_B$k; done
  drop_cache mmpd_pfl2_B$k; echo "$(date +%m-%d\ %H:%M) MBATCH${k}_DONE"
done
cd /SPECPHYS_ROOT
LS=""; for M in $MODELS; do LS="$LS ${M}_L"; done
python scripts/agg_model2.py $LS > /tmp/agg_chain9.log 2>&1; python scripts/agg_mmpd.py $LS >> /tmp/agg_chain9.log 2>&1
python scripts/paper_tables_v2.py > /dev/null 2>&1 && echo TABLES_V2_REGENERATED
echo CHAIN9_ALL_DONE; touch /tmp/chain9.done
