# 20단계 — 학습된 열화 diffusion 주입 (체인 37, 2026-09-01 사전 등록)

사용자 제안: "실 PPG의 깨끗한 파형을 rPPG의 약화된 파형으로 diffusion하는 모듈로 얼굴에 합성" — 두 병목
(체인 34: 실 저SNR 쌍 부족→암기, 체인 35: 손수 설계 잡음의 도메인 격차)의 동시 공략.

## 선행 대비 위치
- Dual-GAN(CVPR'21): GT BVP+잡음→합성 STMap 생성으로 잡음 분포를 학습 (GAN·STMap·백본 내장).
- DiffPhys('24)/Diffusion-Phys: diffusion이지만 역방향(SNR 향상) — 우리 refinement 10구성과 같은 방향(전부 실패).
- UCLA CVPR'22·SCAMPS·본 연구 TW: 주입 계열이나 깨끗한 파형만 주입.
→ "실 PPG→열화 rPPG를 diffusion으로 학습해 픽셀 주입 코퍼스로 산출"은 직접 선행 없음. DSYN·DLS 실패가 필요성의 ablation.

## 설계
1. **열화 모델**: 조건부 ε-U-Net(V1 아키텍처 재사용, CFG), cond=깨끗한 GT 창(pulse space 300), target=CoTAF-TW-L backbone 출력 창.
   쌍 = TRAINPOOL 203 + MMPD 1~6(×4 oversample; 실 저SNR 열화의 주 공급원). 평가 축 불사용(누수 차단).
2. **사전 게이트(check)**: 검증 창에서 생성 열화의 spec-SNR·|ΔHR| 분포를 실측 열화와 대조 — DSYN을 기각시킨
   지표로 "학습된 열화가 hand-crafted보다 실측에 가까운가"를 코퍼스 구축 전에 확인.
3. **주입**: 신규 12클립(TRAIN12DG, 기존 TRAIN20H 얼굴 1~12 재사용, BPM 45~180 커버).
   clean = bank v2 nearest 세그 → diffusion 열화(창 overlap-add, mean spec-SNR ∈ [−9,0] dB 거부 샘플링 — 비디오
   교란과의 이중 열화 방지) → **주입은 열화판, GT 라벨은 clean** (builder wave_file 패치) = 열화→복원 감독.
4. **코퍼스**: COTRAIN_OFFICIAL(203, TW) + 12 = 215 → CoTAF_TWDG_L, 3시드. 기존 203은 불변(커버리지 추가 원칙 — 체인 34 교훈).
5. **판정(§5 관행)**: 주 비교 vs CoTAF_TW_L — MMPD held-out Wilcoxon p<.05 + 타 축 무손실 + 3시드 방향 일치.
   부 지표: halving/doubling, Walking·Skin 6, 저/고심박 축(PURE low-HR).

## 사전 등록 리스크
- 열화 모델이 backbone 출력 열화를 배움 → 픽셀 신호의 광학적 열화와 다를 수 있음(POS trace 쌍은 후속 확장).
- MMPD 1~6 1192창이 열화 다양성의 상한 — 생성이 MMPD풍으로 collapse하면 NIR 등 타 도메인 비용 감시(체인 33 교훈).
- 12/215 클립이라 효과 크기가 작을 수 있음 — null이어도 "학습된 열화 주입의 첫 통제 실험"으로 기록.

코드: scripts/degrade_diffusion.py, builder wave_file 패치, scripts/run_chain37_degradation.sh (/tmp/chain37.log, 체인 36 종료 대기)

## 결과 (2026-09-02 04:55, 체인 37 완료) — 판정: 채택 불가

**열화 모델 자체는 성공**: DEGCHECK에서 생성 열화의 spec-SNR 분포가 실측과 정합(gen p10/50/90 = −3.8/1.7/8.6 vs real −4.5/1.3/8.0) — DSYN(hand-crafted)이 실패했던 사실성 게이트를 처음 통과. 단 |ΔHR| p90 6.6 vs 실측 13.5로 **실측보다 온건한 열화**(HR 파괴 성분 과소생성 — conditional diffusion의 mode-averaging 의심).

**코퍼스 검정은 실패**:
| 시드 | TW → TWDG (MMPD held-out) | p | halving |
|---|---|---|---|
| 100 | 10.51 → 11.36 | .067 | 17 |
| 200 | 10.06 → 10.52 | .092 | 17 |
| 300 | 10.37 → **11.39** | **.023 (유의 악화)** | 23 |

3시드 일관 악화 + halving 12→17~23 + C3/C6 시드 의존 doubling. 해석: 열화 파형 주입(12/215클립)이 파형-라벨 결합을 약화시켜 **harmonic locking 자체를 불안정화** — 저SNR 학습 신호는 1-D refiner(체인 34·35)에 이어 **영상 주입 채널에서도 유해**. 온건한 열화조차 해로우므로 "실측 수준 열화"는 더 해로울 가능성이 큼.

### 20단계 종합
"저SNR을 학습 분포에 넣는" 세 경로(실 쌍 재학습 DLS / 합성 잡음 DSYN / 학습된 열화 주입 TWDG) 전부 기각 —
**깨끗한 파형 형태 다양성(TW)만이 유효**하며, 저SNR 강건성은 이 데이터 규모·주입 채널로는 주입되지 않는다.
학습된 열화 diffusion의 사실성 통과(DEGCHECK)는 방법론적 기여로 서술 가치(Dual-GAN류 대비 diffusion 열화 모델의 첫 통제 실험).
