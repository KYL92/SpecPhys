# 19단계 — 파형 뱅크 확장(TW2) + identity-shortcut probe (체인 36, 2026-09-01)

사용자 승인 방향: "검증된 축(waveform morphology)의 직접 확장 + 발표 주장('identity shortcut 방지')의 실증".

## ① TW2: 실 PPG 파형 뱅크 v2

- 동기: TW(뱅크 131세그, UBFC/PURE 학습분할)의 개선 메커니즘은 실 파형 형태 분포. 현재는 목표 BPM을 리샘플로 만들며(형태-HR 결합 왜곡), 저/고심박의 실 형태가 뱅크에 없음.
- 확장 소스(영상 불필요, GT PPG만): **BIDMC**(PhysioNet ICU fingertip, 임상 다양성) + **PPG-DaLiA**(UCI 손목 BVP, 운동 구간 실 고심박). v1과 동일 품질 게이트(기본파 지배·2f 제외) + **HR 라벨 일치 게이트(±8 BPM)**, 소스별 상한(BIDMC 8/기록, DaLiA 60/피험자).
- 주입 개선: `wave_bank_nearest` — 목표 BPM에 f0가 가까운 상위 20세그 중 선택(로그비 최소화) → 리샘플 왜곡 최소화, 실 HR-형태 결합 보존.
- 산출: TRAIN20H/6HI/6LO_TW2 재주입 + CMP_SYN_TW2 코덱 사본 → COTRAIN_ALLF_TW2(203) → **CoTAF_TW2_L** 3시드.
- 판정(사전 등록, §5 관행 동일): 주 비교 = **vs CoTAF_TW_L** — MMPD held-out Wilcoxon p<.05 개선 + 타 축 무손실 + 3시드 방향 일치면 승격. 미달이어도 "뱅크 규모/커버리지 스케일링" 서술 확보. 주의: 손목 BVP(DaLiA)는 얼굴 rPPG와 형태가 다를 수 있음 — 실패 시 제1 용의자로 기록.

## ② identity-shortcut probe

- 주장(발표 중 제기): "합성 클립은 identity와 주입 HR이 무상관이라 identity 기반 예측 지름길을 끊는다" — 현재 미검증 추론.
- 설계: PURE 학습분할(6 피험자 × ~6 태스크, 35클립)에서 PhysFormer **인코더 특징**(transformer3 토큰 mean+std 풀링, 192-d)으로 선형 probe:
  - identity 복호: 태스크 01~04로 학습, 05~06으로 평가(video held-out), chance 1/6.
  - 통제: 같은 특징으로 chunk HR Ridge 회귀(R²) — HR 정보는 유지되어야 함.
- 4모델 비교: UBFC_base(합성 0·다양성 0) / **CoTReal_L(실 161만, 신규 학습 — 합성 유무 격리 대조군)** / CoTAF_L(+해석적 42) / CoTAF_TW_L(+뱅크 42).
- 예측: 합성 포함 모델(CoTAF_L·TW)의 identity acc < CoTReal_L, HR R²는 동등 이상이면 주장 지지. 반대면 주장 철회(정직 보고).

코드: scripts/build_ppg_bank_v2.py, scripts/probe_identity.py, builder wave_bank_nearest 패치, scripts/run_chain36_bank_probe.sh (/tmp/chain36.log)

## 결과 (2026-09-02 01:20, 체인 36 완료)

### ① TW2 — 판정: 채택 불가 (TW 유지)
bank v2 = 555세그(v1 131 + BIDMC 278 + DaLiA 146; 단 150 BPM 이상 0개 — DaLiA 고심박이 품질 게이트에서 대부분 탈락).

| 시드 | TW → TW2 (MMPD held-out) | p | halving |
|---|---|---|---|
| 100 | 10.51 → 10.84 | .43 | 14 |
| 200 | 10.06 → **11.74** | **<.0001 (유의 악화)** | 22 |
| 300 | 10.37 → 11.04 | .10 | 11 |

기준 CoTAF_L 대비로는 여전히 개선(10.84 vs 12.19, p=.009)이나 **TW보다 일관 열세** + PURE 0.34→1.29, NIRP still 8.51→11.56 후퇴, C3/C6에서 시드 의존 doubling(1/4클립 70→140). 해석(사전 등록 용의자 순):
① **소스 도메인 격차** — 손목 BVP(DaLiA)·ICU fingertip(BIDMC)의 파형 형태가 얼굴 rPPG와 정합하는 UBFC/PURE fingertip과 달라 형태 사전을 희석. ② nearest 선택이 클립당 형태 다양성을 축소. ③ 고심박 구간은 결국 124 BPM 세그 리샘플이라 v1 대비 이득 없음. **교훈: 뱅크는 "많이"가 아니라 "대상 모달리티와 정합하게" — v1(131, rPPG-정합)이 최적.**

### ② identity probe — "identity shortcut 방지" 주장 실증 실패 (문구 교체 권고)
| 모델 | identity acc (chance 16.7%) | HR R² |
|---|---|---|
| UBFC_base | 77.1% | 0.04 |
| CoTReal_L (실 161만) | 85.4% | 0.50 |
| CoTAF_L | 81.3% | 0.53 |
| CoTAF_TW_L | 84.0% | **0.61** |

합성 포함 모델의 identity 복호율이 실-only와 동등(n=144, 유의차 없음) → 주장 기각. 대신 **특징의 HR 정보량이 합성 추가로 단조 증가**(R² 0.50→0.53→0.61) — 발표·논문 문구는 "identity shortcut 차단"이 아니라 "생리신호 정보량 증대"로. (주의: PURE 6명 선형 probe, 얼굴 외형 특징이 사소하게 identity를 담는 것은 모든 비디오 모델의 공통 성질로 보임.)
