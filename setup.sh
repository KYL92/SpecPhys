#!/usr/bin/env bash
# =============================================================================
# Virtual-Face rPPG 파이프라인 — 재현 가능한 개발환경 구축
#
# 스택: SFHQ 얼굴 → LivePortrait 영상화 → Pbv 픽셀 심박 주입 → rPPG-Toolbox 측정
# 대상 하드웨어: NVIDIA RTX 5090 (Blackwell, sm_120) → PyTorch cu128+ 필수
#
# 사용법:  bash setup.sh            # 전체 구축
#          bash setup.sh gen        # 얼굴 생성 환경만
#          bash setup.sh rppg       # rPPG 측정 환경만
#          bash setup.sh verify     # 검증 smoke test만
# 원칙: 버전 고정 · 환경 격리(gen/rppg 분리) · Blackwell 호환 torch · 무손실 데이터
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOS="$ROOT/repos"; WEIGHTS="$ROOT/weights"
CONDA_HOME="${CONDA_HOME:-$HOME/miniforge3}"
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu128"   # Blackwell sm_120 지원

log(){ echo -e "\n\033[1;36m[setup] $*\033[0m"; }

# --- 0. Miniforge (conda 미설치 시) -----------------------------------------
ensure_conda(){
  if [ -x "$CONDA_HOME/bin/conda" ]; then log "conda 존재: $CONDA_HOME"; return; fi
  if command -v conda >/dev/null 2>&1; then CONDA_HOME="$(conda info --base)"; log "시스템 conda 사용: $CONDA_HOME"; return; fi
  log "Miniforge 설치 → $CONDA_HOME"
  curl -L -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$CONDA_HOME"
}
conda_run(){ "$CONDA_HOME/bin/conda" run -n "$1" --no-capture-output bash -c "$2"; }
env_exists(){ "$CONDA_HOME/bin/conda" env list | grep -qE "^\s*$1\s"; }

# --- 1. 얼굴 생성 환경 (vface-gen) ------------------------------------------
setup_gen(){
  log "환경 vface-gen (Python 3.10) — 얼굴 생성/영상화"
  env_exists vface-gen || "$CONDA_HOME/bin/conda" create -n vface-gen python=3.10 -y
  conda_run vface-gen "pip install --upgrade pip && \
    pip install torch torchvision --index-url $TORCH_CUDA_INDEX && \
    pip install numpy scipy opencv-python neurokit2 ffmpeg-python huggingface_hub"
  log "LivePortrait 클론 (MIT)"
  [ -d "$REPOS/LivePortrait" ] || git clone https://github.com/KwaiVGI/LivePortrait.git "$REPOS/LivePortrait"
  # requirements의 torch 핀은 Blackwell과 충돌 → 제거 후 설치
  if [ -f "$REPOS/LivePortrait/requirements.txt" ]; then
    grep -viE '^torch(vision)?([=<>].*)?$' "$REPOS/LivePortrait/requirements.txt" > /tmp/lp_req.txt || true
    conda_run vface-gen "pip install -r /tmp/lp_req.txt" || log "LivePortrait 일부 의존 실패 — 로그 확인"
  fi
  log "LivePortrait 가중치 다운로드 (HuggingFace, 게이팅 없음)"
  # huggingface_hub>=1.0: 'huggingface-cli'는 'hf'로 대체됨
  conda_run vface-gen "hf download KwaiVGI/LivePortrait \
    --local-dir '$REPOS/LivePortrait/pretrained_weights'" || log "가중치 수동 확보 필요"
}

# --- 2. rPPG 측정 환경 (vface-rppg) -----------------------------------------
setup_rppg(){
  log "환경 vface-rppg (Python 3.10) — rPPG 측정/평가"
  env_exists vface-rppg || "$CONDA_HOME/bin/conda" create -n vface-rppg python=3.10 -y
  conda_run vface-rppg "pip install --upgrade pip && \
    pip install torch torchvision --index-url $TORCH_CUDA_INDEX && \
    pip install numpy scipy opencv-python neurokit2 pandas matplotlib h5py"
  log "rPPG-Toolbox 클론 (비지도 방법은 GPU 불필요)"
  [ -d "$REPOS/rPPG-Toolbox" ] || git clone https://github.com/ubicomplab/rPPG-Toolbox.git "$REPOS/rPPG-Toolbox"
  if [ -f "$REPOS/rPPG-Toolbox/requirements.txt" ]; then
    grep -viE '^torch(vision|audio)?([=<>].*)?$' "$REPOS/rPPG-Toolbox/requirements.txt" > /tmp/rppg_req.txt || true
    conda_run vface-rppg "pip install -r /tmp/rppg_req.txt" || log "rPPG-Toolbox 일부 의존 실패 — 로그 확인"
  fi
  # rPPG-Toolbox 비지도 방법이 요구하는 추가 의존 (requirements 누락분)
  conda_run vface-rppg "pip install scikit-image mat73"
  conda_run vface-rppg "pip install open-rppg" || log "open-rppg 설치 실패(선택 항목)"
}

# --- 3. 검증 ----------------------------------------------------------------
verify(){
  log "검증: GPU 아키텍처 + 실제 커널 실행"
  for E in vface-gen vface-rppg; do
    env_exists "$E" || continue
    echo "== $E =="
    conda_run "$E" "python '$ROOT/verify_env.py'" || log "$E 검증 실패"
  done
}

case "${1:-all}" in
  gen)    ensure_conda; setup_gen ;;
  rppg)   ensure_conda; setup_rppg ;;
  verify) ensure_conda; verify ;;
  all)    ensure_conda; setup_gen; setup_rppg; verify ;;
  *) echo "usage: bash setup.sh [all|gen|rppg|verify]"; exit 1 ;;
esac
log "완료. 활성화:  source $CONDA_HOME/bin/activate && conda activate vface-gen"
