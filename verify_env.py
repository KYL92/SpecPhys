#!/usr/bin/env python3
"""환경 검증 smoke test — GPU 아키텍처 호환과 실제 CUDA 커널 실행을 확인한다.
Blackwell(RTX 50xx, sm_120)에서 구버전 torch 휠은 여기서 'no kernel image'로 실패한다."""
import sys

def main():
    ok = True
    try:
        import torch
    except Exception as e:
        print(f"[FAIL] torch import: {e}"); return 1
    print(f"[info] torch {torch.__version__}")
    print(f"[info] cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[FAIL] CUDA 미인식 — 드라이버/휠(cuXXX) 불일치 의심"); ok = False
    else:
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        print(f"[info] device: {name}  capability sm_{cap[0]}{cap[1]}")
        if cap[0] >= 12:
            print("[info] Blackwell 감지 — cu128+ torch 필요")
        # 실제 커널 실행 (no kernel image 판정)
        try:
            x = torch.randn(2048, 2048, device="cuda")
            y = (x @ x).sum().item()
            print(f"[ OK ] CUDA 커널 실행 성공 (matmul sum={y:.1f})")
        except Exception as e:
            print(f"[FAIL] CUDA 커널 실행: {e}"); ok = False
    # 주입 파이프라인 공통 의존
    for m in ("numpy", "cv2", "scipy"):
        try:
            __import__(m); print(f"[ OK ] {m}")
        except Exception as e:
            print(f"[WARN] {m} 미설치: {e}")
    try:
        import neurokit2  # noqa
        print("[ OK ] neurokit2 (현실적 PPG 파형)")
    except Exception:
        print("[WARN] neurokit2 미설치 — 주입은 정현파 폴백 사용")
    print("[done]", "PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
