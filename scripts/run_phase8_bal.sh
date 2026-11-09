#!/bin/bash
# Phase 8 — 균형 학습 풀(COTRAIN_BAL) 체인: CoTB(기본 손실) / CoTBH(+고조파 억제 손실) 학습(2) → 평가(2×10).
# GPU 게이트: 각 단계 전 nvidia-smi 여유 VRAM 확인, 부족하면 60초 폴링(학습 19000 MiB, 평가 8192 MiB).
# CoTBH 는 트레이너 패치 scripts/patches/physformer_harm_loss.patch 가 적용돼 있어야 한다 —
#   적용 여부를 grep 으로 확인하고 없으면 자동 적용(REVERT_PATCH=1 이면 체인 종료 시 복원).
#   손실 게이트는 환경변수: PF_HARM_LOSS=1 PF_HARM_W(기본 1.0) PF_HARM_TAU(힌지 기본 0.3) PF_HARM_SIGMA_HZ(기본 0.1).
# 로그: _workspace/data/physformer/runs/<name>/stdout.log, 마스터 runs/phase8_bal_chain.log
# 사용:  bash scripts/run_phase8_bal.sh                 # 전체 (CoTB → CoTBH → 평가 20)
#        MODELS="CoTB" bash scripts/run_phase8_bal.sh    # 한 모델만
#        ONLY_EVAL=1 bash scripts/run_phase8_bal.sh      # 학습 건너뛰고 평가만
#        PF_HARM_W=0.5 bash scripts/run_phase8_bal.sh    # 고조파 손실 가중치 변경(CoTBH만 영향)
# 분석: python scripts/analyze_train_pool_balance.py (PURE held-out 07~10 은 vid id>=700 필터로 보고)
set -u
ROOT=/SPECPHYS_ROOT
TB=$ROOT/repos/rPPG-Toolbox
PF=$ROOT/_workspace/data/physformer
Y=$PF/yaml13
RUNS=$PF/runs
PATCH=$ROOT/scripts/patches/physformer_harm_loss.patch
PY=${PY:-$HOME/miniforge3/envs/vface-rppg/bin/python}
MODELS=${MODELS:-"CoTB CoTBH"}
SETS=${SETS:-"MCD_A MCD_B REALTEST RTNC RTNI C1 C3 C6 E3 PURE"}
EVAL_GATE_MIB=${EVAL_GATE_MIB:-8192}
TRAIN_GATE_MIB=${TRAIN_GATE_MIB:-19000}
ONLY_EVAL=${ONLY_EVAL:-0}
REVERT_PATCH=${REVERT_PATCH:-0}
PF_HARM_W=${PF_HARM_W:-1.0}
PF_HARM_TAU=${PF_HARM_TAU:-0.3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$TB" || { echo "toolbox dir missing: $TB"; exit 1; }
MASTER=$RUNS/phase8_bal_chain.log
mkdir -p "$RUNS"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER"; }

[ -d "$PF/COTRAIN_BAL" ] || { log "COTRAIN_BAL missing — run scripts/assemble_cotrain_bal.py first"; exit 1; }
# v2(+TRAIN_BAL_LOA2, 221 링크) 확인 — 캐시 cotrain_bal 이 이미 있는데 링크 수가 다르면 옛 캐시(v1 209)일 수 있으므로 중단
NLINK=$(ls -d "$PF"/COTRAIN_BAL/subject* | wc -l)
[ "$NLINK" -eq "${EXPECT_LINKS:-221}" ] || { log "COTRAIN_BAL has $NLINK links (expected ${EXPECT_LINKS:-221}) — re-run scripts/assemble_cotrain_bal.py"; exit 1; }
if [ -d "$PF/cache/cotrain_bal" ]; then
  NCACHE=$(ls "$PF/cache/cotrain_bal" | sed 's/_input.*//' | sort -u | wc -l)
  [ "$NCACHE" -eq "$NLINK" ] || { log "cache/cotrain_bal has $NCACHE subjects != $NLINK links — stale cache; rename EXP_DATA_NAME to cotrain_bal_v2 or delete cache"; exit 1; }
fi

ensure_patch() { # CoTBH 학습 전: 패치 적용 확인/적용
  if grep -q "PF_HARM_LOSS" neural_methods/trainer/PhysFormerTrainer.py; then log "[patch] harm-loss patch already applied"; return 0; fi
  if patch -p1 --dry-run < "$PATCH" >/dev/null 2>&1; then patch -p1 < "$PATCH" && log "[patch] applied $PATCH"; return 0; fi
  log "[patch] FAILED to apply $PATCH (manual merge needed)"; return 1
}
revert_patch() {
  if [ "$REVERT_PATCH" = "1" ] && grep -q "PF_HARM_LOSS" neural_methods/trainer/PhysFormerTrainer.py; then
    patch -R -p1 < "$PATCH" && log "[patch] reverted"; fi
}
trap revert_patch EXIT

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

run_step() { # $1 name, $2 yaml, $3 gate MiB, $4.. extra env (KEY=VAL) — OOM 이면 1회 재대기 후 재시도
  local name=$1 cfg=$2 need=$3; shift 3
  local logf=$RUNS/$name/stdout.log attempt
  mkdir -p "$RUNS/$name"
  [ -f "$cfg" ] || { log "$name MISSING_YAML $cfg"; return 1; }
  for attempt in 1 2; do
    wait_gpu "$need"
    log "$name START (attempt $attempt) $cfg env=[$*]"
    if env "$@" "$PY" main.py --config_file "$cfg" > "$logf" 2>&1; then log "$name OK"; return 0; fi
    if grep -q "OutOfMemoryError\|CUDA out of memory\|cudaErrorMemoryAllocation" "$logf"; then
      log "$name OOM_RETRY_$attempt"; sleep 120
    else
      log "$name FAIL (see $logf)"; return 1
    fi
  done
  log "$name FAIL_OOM"; return 1
}

log "===== phase8 start: MODELS=[$MODELS] SETS=[$SETS] PF_HARM_W=$PF_HARM_W ====="
# 1) 학습 (순차). 전처리 캐시 cotrain_bal 은 첫 학습이 생성(DO_PREPROCESS: True); 두 번째는 재사용.
for M in $MODELS; do
  CKPT=$RUNS/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  if [ "$ONLY_EVAL" = "1" ] || [ -f "$CKPT" ]; then log "finetune_$M SKIP (ckpt exists or ONLY_EVAL)"; continue; fi
  CFG=$Y/finetune_$M.yaml
  if [ -d "$PF/cache/cotrain_bal" ] && grep -q "DO_PREPROCESS: True" "$CFG"; then
    sed -i '0,/DO_PREPROCESS: True/s//DO_PREPROCESS: False/' "$CFG"; log "finetune_$M: cache cotrain_bal exists -> DO_PREPROCESS False"; fi
  if [ "$M" = "CoTBH" ]; then
    ensure_patch || { log "finetune_$M SKIP (patch failed)"; continue; }
    run_step "finetune_$M" "$CFG" "$TRAIN_GATE_MIB" PF_HARM_LOSS=1 PF_HARM_W=$PF_HARM_W PF_HARM_TAU=$PF_HARM_TAU || log "finetune_$M failed"
  else
    run_step "finetune_$M" "$CFG" "$TRAIN_GATE_MIB" PF_HARM_LOSS=0 || log "finetune_$M failed"
  fi
done
# 2) 평가 (순차, Epoch9 체크포인트가 있는 모델만; 평가에는 손실 게이트 무관)
for M in $MODELS; do
  CKPT=$RUNS/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  [ -f "$CKPT" ] || { log "ev_$M SKIP_ALL (no ckpt $CKPT)"; continue; }
  for S in $SETS; do
    if ls "$RUNS/ev_${M}_$S"/*/saved_test_outputs/*.pickle >/dev/null 2>&1; then
      log "ev_${M}_$S SKIP (outputs exist)"; continue; fi
    run_step "ev_${M}_$S" "$Y/ev_${M}_$S.yaml" "$EVAL_GATE_MIB"
  done
done
log "===== phase8 done ====="
