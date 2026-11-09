#!/bin/bash
# Phase 6b ablation chain — CoTNC / CoTNW / CoTNH2 학습(3) → 8세트 평가(24).
# GPU 게이트: 각 단계 시작 전 nvidia-smi 여유 VRAM을 확인하고 부족하면 60초 폴링 대기.
#   기본 게이트 8 GiB (요청 사양). 학습(batch 4)은 과거 실측 ≈19 GB가 필요하므로
#   TRAIN_GATE_MIB로 별도 상향 가능 (기본 19000). 평가(batch 4)는 EVAL_GATE_MIB (기본 8192).
# 로그: _workspace/data/physformer/runs/<name>/stdout.log
# 사용:  bash scripts/run_phase6b.sh            # 전체
#        ONLY_EVAL=1 bash scripts/run_phase6b.sh # 학습 건너뛰고 평가만 (체크포인트 존재 시)
#        MODELS="CoTNC" bash scripts/run_phase6b.sh
set -u
ROOT=/SPECPHYS_ROOT
TB=$ROOT/repos/rPPG-Toolbox
PF=$ROOT/_workspace/data/physformer
Y=$PF/yaml10
RUNS=$PF/runs
PY=${PY:-$HOME/miniforge3/envs/vface-rppg/bin/python}
MODELS=${MODELS:-"CoTNC CoTNW CoTNH2"}
SETS=${SETS:-"MCD_A MCD_B REALTEST RTNC RTNI C3 C6 E3"}
EVAL_GATE_MIB=${EVAL_GATE_MIB:-8192}
TRAIN_GATE_MIB=${TRAIN_GATE_MIB:-19000}
ONLY_EVAL=${ONLY_EVAL:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$TB" || { echo "toolbox dir missing: $TB"; exit 1; }
MASTER=$RUNS/phase6b_chain.log
mkdir -p "$RUNS"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER"; }

wait_gpu() { # $1 = 필요 여유 MiB — 확보될 때까지 60초 폴링
  local need=$1 waited=0 used total free
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    free=$((total - used))
    if [ "$free" -ge "$need" ]; then log "[gpu] free=${free}MiB >= ${need}MiB (waited ${waited}s)"; return 0; fi
    [ $((waited % 600)) -eq 0 ] && log "[gpu] waiting: free=${free}MiB < ${need}MiB (${waited}s)"
    sleep 60; waited=$((waited + 60))
  done
}

run_step() { # $1 name, $2 yaml, $3 gate MiB — OOM이면 1회 재대기 후 재시도
  local name=$1 cfg=$2 need=$3 logf=$RUNS/$1/stdout.log attempt
  mkdir -p "$RUNS/$name"
  [ -f "$cfg" ] || { log "$name MISSING_YAML $cfg"; return 1; }
  for attempt in 1 2; do
    wait_gpu "$need"
    log "$name START (attempt $attempt) $cfg"
    if "$PY" main.py --config_file "$cfg" > "$logf" 2>&1; then log "$name OK"; return 0; fi
    if grep -q "OutOfMemoryError\|CUDA out of memory\|cudaErrorMemoryAllocation" "$logf"; then
      log "$name OOM_RETRY_$attempt"; sleep 120
    else
      log "$name FAIL (see $logf)"; return 1
    fi
  done
  log "$name FAIL_OOM"; return 1
}

log "===== phase6b start: MODELS=[$MODELS] SETS=[$SETS] ====="
# 1) 학습 (순차)
for M in $MODELS; do
  CKPT=$RUNS/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  if [ "$ONLY_EVAL" = "1" ] || [ -f "$CKPT" ]; then log "finetune_$M SKIP (ckpt exists or ONLY_EVAL)"; continue; fi
  run_step "finetune_$M" "$Y/finetune_$M.yaml" "$TRAIN_GATE_MIB" || log "finetune_$M failed — evals for $M will be skipped"
done
# 2) 평가 (순차, Epoch9 체크포인트가 있는 모델만)
for M in $MODELS; do
  CKPT=$RUNS/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  [ -f "$CKPT" ] || { log "ev_$M SKIP_ALL (no ckpt $CKPT)"; continue; }
  for S in $SETS; do
    if ls "$RUNS/ev_${M}_$S"/*/saved_test_outputs/*.pickle >/dev/null 2>&1; then
      log "ev_${M}_$S SKIP (outputs exist)"; continue; fi
    run_step "ev_${M}_$S" "$Y/ev_${M}_$S.yaml" "$EVAL_GATE_MIB"
  done
done
log "===== phase6b done ====="
