# 공용 체인 함수 (source 해서 사용). 전제: vface-rppg env, cumsum 라벨
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RPPG_MP_QUOTA=${RPPG_MP_QUOTA:-6} RPPG_HR_LABEL=cumsum RPPG_NUM_WORKERS=${RPPG_NUM_WORKERS:-6}
ROOT=/SPECPHYS_ROOT; PFA=$ROOT/_workspace/data/physformer; TPL=$PFA/yaml24
# [10-31, 54 문서] 다중 GPU 박스(PRO 6000)에서도 동작: CUDA_VISIBLE_DEVICES 의 첫 장치(없으면 첫 줄)만 본다
wait_gpu() { local q="nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i ${CUDA_VISIBLE_DEVICES%%,*}}"; while true; do u=$($q --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1); t=$($q --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1); [ $((t-u)) -ge "$1" ] && return 0; sleep 60; done; }
# 디스크 가드(2026-09-02 ENOSPC 사고 재발 방지): 캐시 파일시스템·루트 중 하나라도 여유 < VF_DISK_MIN_GB(기본 60G)면
# 대기(캐시 증가 중단). 툴박스 BaseLoader의 하드 가드(RPPG_CACHE_MIN_FREE_GB=40)보다 먼저 발동하는 1차 방벽.
wait_disk() { local MIN=$((${VF_DISK_MIN_GB:-60}*1024*1024)); while true; do
  local ok=1 p; for p in "$(readlink -f $PFA/cache)" /; do
    [ "$(df -k --output=avail "$p" 2>/dev/null | tail -1)" -lt "$MIN" ] 2>/dev/null && { ok=0; echo "$(date +%H:%M) DISK_WAIT $p < ${VF_DISK_MIN_GB:-60}G"; }; done
  [ $ok = 1 ] && return 0; sleep 300; done; }
run_gated() { wait_disk; wait_gpu "$1"; ( cd $ROOT/repos/rPPG-Toolbox && python main.py --config_file "$2" > "$3" 2>&1 ) && echo "$(date +%H:%M) ${4}_OK" || echo "$(date +%H:%M) ${4}_FAIL"; }
drop_cache() { rm -rf "$PFA/cache/$1" "$PFA/cache/DataFileLists/${1}_0.0_1.0.csv"; }
# gen_yaml M POOL cache mmpdtag [MODEL_SRC=M] [DYN=0]  → $PFA/yaml_$M/
gen_yaml() { local M=$1 P=$2 c=$3 e=$4 SRC=${5:-$1} DYN=${6:-0}; local Y=$PFA/yaml_$M; mkdir -p $Y
  for f in $TPL/*.yaml; do b=$(basename $f); sed -e "s/CoTAF_DM_L/$M/g" -e "s/COTRAIN_ALLFDM/COTRAIN_$P/g" -e "s/cotrain_allfdm/cotrain_$c/g" -e "s/mmpd_pfdm_/mmpd_${e}_/g" $f > $Y/${b//CoTAF_DM_L/$M}; done
  [ "$SRC" != "$M" ] && sed -i "s#runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer#runs/finetune_$SRC/PreTrainedModels/SYNTH_${SRC}_physformer#g" $Y/ev_${M}_*.yaml
  if [ "$DYN" = "1" ]; then sed -i -e "s/DO_DYNAMIC_DETECTION: false/DO_DYNAMIC_DETECTION: true/" -e "s/EXP_DATA_NAME: real_/EXP_DATA_NAME: dyn_/" $Y/ev_${M}_*.yaml; fi
  echo $Y; }
# make_pool POOL src_dir... (subject* 심볼릭 링크 병합)
make_pool() { local P=$1; shift; rm -rf $PFA/COTRAIN_$P; mkdir -p $PFA/COTRAIN_$P; for d in "$@"; do for s in $d/subject*; do ln -sfn "$(readlink -f $s)" "$PFA/COTRAIN_$P/$(basename $s)"; done; done; echo "POOL_$P n=$(ls $PFA/COTRAIN_$P | wc -l) broken=$(find $PFA/COTRAIN_$P -xtype l | wc -l)"; }
# 실 161 + 지정 가상 세트(TRAIN20H_x, TRAIN6HI_x, TRAIN6LO_x, CMP_SYN_x)로 203 풀 구성: make_pool203 POOL SUFFIX
make_pool203() { local P=$1 X=$2; rm -rf $PFA/COTRAIN_$P; mkdir -p $PFA/COTRAIN_$P
  for s in $PFA/COTRAIN_ALLF/subject*; do t=$(readlink -f $s); case $t in *TRAIN20H/*|*TRAIN6HI/*|*TRAIN6LO/*|*CMP_SYN/*) ;; *) ln -sfn "$t" "$PFA/COTRAIN_$P/$(basename $s)";; esac; done
  for k in $(seq 1 20); do ln -sfn $PFA/TRAIN20H_$X/subject$k $PFA/COTRAIN_$P/subject$((200+k)); done
  for k in $(seq 1 6); do ln -sfn $PFA/TRAIN6HI_$X/subject$k $PFA/COTRAIN_$P/subject$((220+k)); ln -sfn $PFA/TRAIN6LO_$X/subject$k $PFA/COTRAIN_$P/subject$((230+k)); done
  for k in $(seq 1 10); do ln -sfn $PFA/CMP_SYN_$X/subject$((600+k)) $PFA/COTRAIN_$P/subject$((600+k)); done
  echo "POOL_$P n=$(ls $PFA/COTRAIN_$P | wc -l) broken=$(find $PFA/COTRAIN_$P -xtype l | wc -l)"; }
# codec_copies SRCSET DSTSET MODE(sweep|mpeg4): TRAIN20H_x 홀수 10클립 → CMP_SYN_x
codec_copies() { local S=$1 D=$2 MODE=$3; rm -rf $PFA/$D; mkdir -p $PFA/$D; local i=0; for k in 1 3 5 7 9 11 13 15 17 19; do i=$((i+1)); local d=$PFA/$D/subject$((600+i)); mkdir -p $d
  if [ "$MODE" = "sweep" ]; then case $i in 1|2|3|4) ffmpeg -v error -y -i $PFA/$S/subject$k/vid.avi -c:v mpeg4 -q:v 6 $d/vid.avi;; 5|6|7) ffmpeg -v error -y -i $PFA/$S/subject$k/vid.avi -c:v libx264 -crf 23 -pix_fmt yuv420p $d/vid.avi;; *) ffmpeg -v error -y -i $PFA/$S/subject$k/vid.avi -c:v libx264 -crf 37 -pix_fmt yuv420p $d/vid.avi;; esac
  else ffmpeg -v error -y -i $PFA/$S/subject$k/vid.avi -c:v mpeg4 -q:v 6 $d/vid.avi; fi; cp $PFA/$S/subject$k/ground_truth.txt $d/; done; echo "$D n=$(ls $PFA/$D | wc -l)"; }
# train_eval M cache mmpdtag [SEED=100]
train_eval() { local M=$1 c=$2 e=$3 SEED=${4:-100}; local Y=$PFA/yaml_$M
  drop_cache cotrain_$c; RPPG_SEED=$SEED run_gated 21000 "$Y/finetune_$M.yaml" /tmp/ft_$M.log ${M}_TRAIN; drop_cache cotrain_$c
  find $PFA/runs/finetune_$M/PreTrainedModels -name "*.pth" -not -name "*Epoch9.pth" -delete 2>/dev/null; eval_only $M $c $e; }
eval_only() { local M=$1 c=$2 e=$3; local Y=$PFA/yaml_$M
  for f in $Y/ev_${M}_*.yaml; do A=$(basename "$f" .yaml | sed "s/ev_${M}_//"); case $A in MMPD_B*) continue;; esac; run_gated 6000 "$f" /tmp/ev_${M}_$A.log ${M}_$A; done
  for k in 1 2 3 4 5 6; do run_gated 6000 "$Y/ev_${M}_MMPD_B$k.yaml" /tmp/ev_${M}_MMPD_B$k.log ${M}_MMPD_B$k; drop_cache mmpd_${e}_B$k; done
  ( cd $ROOT && python scripts/agg_model2.py $M > /tmp/agg_$M.log 2>&1; python scripts/agg_mmpd.py $M >> /tmp/agg_$M.log 2>&1 ); echo "$(date +%H:%M) AGG_${M}_DONE"; }
