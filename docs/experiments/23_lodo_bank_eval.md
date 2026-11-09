# 21단계 — 뱅크 확장의 Leave-One-Dataset-Out CV (체인 38, 2026-09-02)

사용자 지시: "뱅크 확장을 DLCN_face, MMPD, MR-NIRP, mcd_rppg, PURE 전부에서 평가 — 데이터셋 단위 LODO-CV".

## 세션 충돌 기록 (2026-09-02 10:54~11:05)
같은 지시가 두 세션(user-8b, user-bc)에 내려져 동일 경로(scripts/run_chain38_lodo.sh, /tmp/chain38.log)로
이중 기동됨. user-bc(본 세션)가 스크립트를 덮어썼으나 user-8b의 bash는 삭제된 원본 inode를 계속 읽어
**오염 없음**을 확인, 원본을 /proc fd에서 복구하고 본 세션 인스턴스는 종료·철수. **실행 소유: user-8b.**
본 세션의 대안 설계(v1 vs v2 외부 소스 LODO)는 scripts/run_chain38_bc_lodo.sh.parked 로 보존(미실행).

## 실행 중인 설계 (user-8b)
- **뱅크 확장** = 5개 데이터셋 학습분할 GT PPG로 뱅크 확장(build_ppg_bank_lodo.py), 폴드별로
  **대상(held-out) 데이터셋 GT를 제외**한 bank_lodo_{FOLD}.npz — 파형 수준 누수 차단.
- 학습 풀은 기존 203 구성 유지(실 161 불변, 가상 42만 폴드 뱅크로 재주입 — TW와 동일 idx 정책, nearest 미사용)
  → 순수 "파형 뱅크 소스" 요인 분리. 모델 CoTAF_LO{FOLD}_L, 시드 100, 폴드 축만 평가.
- 폴드 순서 DLCN→MMPD→NIRP→MCD→PURE, 요약 scripts/lodo_summary.py, 로그 /tmp/chain38.log.
- 해석 시 주의: 비교 기준이 CoTAF_TW_L일 경우 PURE 폴드는 TW(v1 뱅크에 PURE 포함) 쪽에 누수 이점 —
  PURE 폴드만은 LOPURE vs "UBFC-only 뱅크판"이 공정(요약 스크립트 확인 필요).

체인 36 교훈(모달리티 정합)과 부합하는 확장 방향: BIDMC/DaLiA(외부 모달리티)가 아니라
**같은 rPPG 벤치마크들의 접촉식 GT PPG**로 확장 — 얼굴 rPPG와 정합하는 소스만 뱅크에 유효하다는 가설의 직접 검정.

## 최종 결과 (2026-09-02 21:32, 체인 38c 완주 — user-bc 인수)

인수 경위: user-8b 체인이 루트 ENOSPC 여파로 NIRP 폴드 중 비정상 종료 → 사용자 지시로 본 세션이
MMPD B5/B6 재평가·재집계 + NIRP/MCD/PURE 폴드(원 설계 그대로) + 요약을 완주. DLCN 폴드와 MMPD 학습은
user-8b 산출물을 그대로 사용.

| 폴드 (미학습 데이터셋) | n | TW(뱅크 v1) → LO(확장 뱅크) | p |
|---|---|---|---|
| DLCN | 20 | 1.71 → 1.85 | .85 |
| MMPD (전 클립) | 655 | 10.17 → 10.71 | .20 (held-out 537: 10.43→11.16, p=.14) |
| MR-NIRP | 57 | 9.62 → 8.79 | .69 (small 10.36→7.59 비유의, n=19) |
| MCD | 40 | 1.80 → 2.46 | .86 |
| PURE | 59 | 0.34 → 1.34 | .29 (비대칭: TW 뱅크에 PURE 포함. 보조 CoTAF_L 0.43→1.34 p=.72) |
| **풀링** | **831** | **8.83 → 9.30** | **.19 (개선 176 / 악화 182)** |

**판정: 확장 뱅크 채택 불가.** 5폴드 전부 비유의, 방향은 중립~악화(풀링 +0.48). PURE 폴드는 누수 비대칭을
보정한 보조 비교(CoTAF_L 대비)에서도 악화 방향 — "타 rPPG 데이터셋 GT 추가"는 미학습 데이터셋 일반화에
이득이 없고 오히려 대상-정합 파형(UBFC/PURE fingertip)의 사전을 희석한다.

### 뱅크 확장 축 종결 (19단계 + 21단계 종합)
① v2(외부 모달리티 BIDMC/DaLiA): MMPD 중심 평가에서 유의 악화(19 문서). ② LODO(rPPG-정합 5셋 GT,
파형 누수 차단): 전 폴드 비유의·풀링 악화 방향. → **파형 뱅크는 v1(131세그, UBFC/PURE 학습분할) 최종 확정.**
TW의 이득은 "많은 파형"이 아니라 "적더라도 대상 모달리티·수집 조건에 정합하는 파형"에서 나온다 — 논문
Discussion의 뱅크 스케일링 절 소재(2회의 확장 실패가 ablation).

부속: 캐시 sub-disk 이전 완료(cache→/SPECPHYS_CACHE/vface_pf_cache 심볼릭 링크, 루트 535G 확보),
디스크 가드 2층 상시화. 시드 100 1개 한계는 사전 등록대로 병기.
