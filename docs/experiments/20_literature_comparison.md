# 문헌 비교 정본 (2026-08-25) — PURE/UBFC/MR-NIRP/MCD 사용 최신 논문 수치

조사: 본 세션 리서치 에이전트(원문 PDF 확인 기반). 프로토콜 경고: rPPG-Toolbox 계열(영상단위 HR)과 SiNC 계열(10초 슬라이딩 윈도우)은 같은 표에 섞지 말 것 (동일 모델·방향에서 0.98 vs 7.02까지 벌어짐).

## A. PURE/UBFC (조사 에이전트 1 — 전체 원문은 세션 기록)

### 핵심 요약
- PhysFormer(CVPR'22)·PhysFormer++(IJCV'23) 원 논문 모두 PURE/UBFC 수치 없음 → 제3자 재학습치만 존재
- rPPG-Toolbox(NeurIPS'23 D&B, arXiv:2210.00716, 부록 Table 7): PhysFormer UBFC→PURE **12.92±2.69**, PURE→UBFC **1.44±0.54** / POS→PURE 3.67, POS→UBFC 4.08 / PhysNet UBFC→PURE 8.06 / TS-CAN 3.69 / DeepPhys 5.54 / EfficientPhys-C 5.47
- 최신 SOTA 교차 (rPPG-Toolbox 프로토콜, UBFC→PURE / PURE→UBFC):
  - RhythmFormer (PR 2025, arXiv:2402.12788): **0.97 / 0.89**
  - PhysMamba (CCBR 2024, arXiv:2409.12031): 1.20 / 0.97
  - RhythmMamba (AAAI 2025, arXiv:2404.06483): 1.98 / 0.95
  - Spiking-PhysFormer (Neural Networks 2025, arXiv:2402.04798): 3.83 / 2.80 (PhysMamba가 인용한 2.70/5.25는 구버전 — 각주 처리)
- SiNC(CVPR'23) 인트라: UBFC 0.59 / PURE 0.61; 교차 UBFC→PURE 4.02 (SiNC+ TSCAN 백본 1.15)
- Contrast-Phys+(TPAMI'24) 인트라: UBFC 0.21 / PURE 0.48 (100% 라벨); PURE↔UBFC 교차는 원저자 미보고(SiNC 재현치 19.61/10.22, 귀속 필수)
- 우리 파이프라인 앵커: base(UBFC 사전학습 PhysFormer)→PURE 정본 10.67 vs toolbox 보고 12.92 — 프로토콜 차이(밴드 0.6-3.3 vs 0.75-2.5, detrend) 감안 시 정합

### 프로토콜 충돌 주의
1. Spiking-PhysFormer 수치 2종 존재(3.83/2.80 저널판 채택)
2. rPPG-Toolbox 본문 Table2 vs 부록 Table7 소수점 차이(부록 값 채택)
3. PhysNet PURE→UBFC 0.98(toolbox) vs 7.02(SiNC) — 계열 분리 제시 필수

## B. MR-NIRP Car (조사 에이전트 2)

- 원 논문 계열(Nowara T-ITS 2022, AutoSparsePPG)은 MAE 미보고(RMSE/PTE6): driving NIR RMSE 11.6, RGB >15; garage RGB 2.9. RGB는 야간 주행 붕괴, 차고에선 NIR보다 우수.
- **RGB driving MAE의 유일한 포괄 벤치마크: Bouraffa et al., WACVW 2025** (5-fold subject-exclusive = in-domain 학습): Driving Still — PhysNet 4.37/PhysFormer 6.35/POS 8.14; Driving Small — 5.80/7.88/8.54; Driving Large — 7.04/9.17/**POS 9.16**.
- **외부 검증 앵커**: 우리 POS 측정(정지 15.08/소 13.04/대 9.48)과 Bouraffa POS(8.14/8.54/9.16) — 대동작은 거의 일치(9.48 vs 9.16). 정지/소동작 차이는 세션 선택(975nm 세션만)·60초 절단·평가 인원 차이로 설명 가능, 논문엔 대동작 앵커만 강조.
- WACV 2026 FREC(RGB+NIR 융합): 전체 7.51, PhysFormer(RGB 재학습) 6.98. MS-rPPG(arXiv 2606.21115): **PhysFormer NIR large-motion 5.36→34.83 붕괴** — 우리 MCD 고조파 배가 붕괴와 동일 실패 양식(병기 가치).
- 우리 제로샷 CoTAF(7.49/9.11/10.78)는 **in-domain 학습된** Bouraffa PhysFormer(6.35/7.88/9.17)에 근접 — 제로샷 대 in-domain임을 명기하면 강한 결과.

## C. MCD-rPPG (원 논문 Egorov et al. ACM MM 2025, arXiv:2508.17924)

- Table 3 (정면): POS 3.80(우리 4.75와 동일 자릿수 — 앵커), **UBFC 학습 PhysFormer 43.65±3.71 (원저자도 붕괴 관측!)**, 인트라 PhysFormer 4.08, **RhythmFormer 인트라 2.82(최고)**, iBVPNet 4.83.
- 우리 CoTAF(정체성 분리 in-domain) 0.48은 보고된 모든 수치(인트라 RhythmFormer 2.82 포함)보다 낮음 — 단 평가 부분집합(20명 정면 60초) 차이 명기 필수.
- 후속 사용 논문은 MVRD-Bench(arXiv:2603.22826) 1편뿐: SiNC→MCD 2.69, MVRD 0.69. 우리 작업이 사실상 2번째 외부 사용.

## D. DLCN 출처 특정 (요청 항목 해결)

**Li et al., "Exploring Remote Physiological Signal Measurement under Dynamic Lighting Conditions at Night: Dataset, Experiment, and Analysis", arXiv:2507.04306 (2025)** — 98명(남55/여43), 784개 1분 영상, 640×480@30fps→128² 크롭, CMS50E PPG, HR 45~154, 야간 동적 조명(FI/VI×FP/VP)+rest/exercise. 게재처 미정(arXiv only) — 스펙이 우리 확보분과 일치. **논문의 "출처 확인 중" 익명 처리를 이 인용으로 교체할 것.**
- 인트라(전 조건 학습): PhysFormer rest 1.699/exercise 2.482/최난도 VI&VP 5.343; PhysNet 1.982; POS 11.016 (동적 조명에서 고전 붕괴 — 우리 플리커 락인 발견의 실데이터 인용처)
- 교차(UBFC→DLCN): PhysFormer FI&FP 3.944 (우리 base 3.69와 정합 앵커), VI&VP 14.611
- 우리 제로샷 CoTAF 1.36은 **인트라 학습된 PhysFormer(rest 1.699)보다 낮음** — 단 우리 평가는 20명 첫 트라이얼 부분집합(조건 라벨 미구분)임을 명기.

## MMPD 문헌 수치 (2026-10-17 검증, RhythmFormer 논문 arXiv 2402.12788 Table 1·3 — mini-MMPD, 저자 툴박스 재구현)
- **인트라(MMPD 학습, 순차 7:1:2 분할·피험자 분리 여부 불명)**: RhythmFormer **3.07**/6.81/r .86, PhysFormer++ 4.88/7.62, PhysNet 4.80/11.80, Dual-GAN 4.93/7.68, CVD 5.02/7.97, RhythmNet 5.30/8.14, **PhysFormer 11.99**/18.41/r .18, EfficientPhys 13.47, TS-CAN 9.71, DeepPhys 22.27.
- **교차(MMPD 전체를 테스트)**: POS 12.34/17.70, CHROM 13.63, PhysFormer UBFC→MMPD **12.10**/17.79 (PURE→ 14.57), RhythmFormer UBFC→MMPD **9.08**/15.07/r .53 (PURE→ 8.98), PhysNet UBFC→ 10.24, TS-CAN 14.01, EfficientPhys 13.78, DeepPhys 17.50.
- 프로토콜: 160 프레임, 첫 프레임 얼굴 박스 고정 128×128, Butterworth 0.75–2.5 Hz 후 Welch, 30 epoch batch 4; **시간 증강**(HR>90 bpm 영상 무작위 다운샘플, <75 bpm 업샘플 + 좌우 반전) 사용 — 32 문서(체인 50) 시간 배속 증강의 문헌 근거 추가.
- 우리 수치와의 관계(정본 프로토콜, MMPD held-out 피험자 7~33, n=540): 툴박스 공식 UBFC 가중치 zero-shot PhysFormer 14.39 / RhythmFormer 15.26 / POS 12.42 ↔ 문헌 교차 12.10 / 9.08 / 12.34 — POS·PhysFormer 는 일치 범위, RhythmFormer 는 문헌 자체 학습 가중치(9.08)가 툴박스 배포 가중치(15.26)보다 크게 낫다(가중치·테스트 집합·mini-MMPD 차이). 우리 공식 모델 CoTAF_TW_L 10.84(5시드)·하이브리드 9.69(5시드, MMPD 1~6 포함)·앙상블 9.10 → **PhysFormer 백본 기준 문헌 교차 12.10 대비 우위, RhythmFormer 교차 9.08 과 동등 수준(프로토콜 상이), 문헌 인트라 3.07 과는 비교 불가(MMPD 학습 vs 거의 zero-shot)**.
