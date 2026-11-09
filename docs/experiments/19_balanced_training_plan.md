# 19단계: 균형 학습 풀(COTRAIN_BAL) 설계·조립 + PURE 안정부 오차 진단 + 고조파 억제 손실 준비

작성: physformer-training-engineer (2026-08-25). **GPU 실행 없음** — 진단(기존 피클 재분석)·합성 생성(CPU)·풀 조립·패치·yaml·실행 스크립트까지. 학습/평가는 `scripts/run_phase8_bal.sh`로 GPU 여유 시 실행.

문제 제기: "PURE 안정(steady) 구간에서 PhysFormer 계열 성능이 낮다. 합성 데이터가 노이즈에 편중된 탓 아닌가. 생성+실제의 안정/노이즈 상황과 전 심박 범위를 고루 학습하라."

## 0. 요약

1. **PURE steady 저성능의 원인은 저심박 배가(×2)이지 안정 조건 자체가 아니다.** steady 10 트라이얼 중 GT<55 BPM인 3건(05-01 45.9, 08-01 46.5, 09-01 45.2)이 정확히 2배로 예측되어 MAE 13.8을 만들고, 나머지 7건(59~125 BPM)은 MAE 0.2다. 같은 저심박 피험자(5·8·9)는 talking/translation/rotation 태스크에서도 똑같이 배가된다 — 태스크 무관, **주파수 커버리지 문제**. steady가 유독 나빠 보인 이유는 steady 세트에 59 트라이얼 중 가장 낮은 GT 3건이 몰려 있기 때문(steady GT 하한 47.3 vs 다른 태스크 49.7~51.4)이다.
2. 합성 풀이 노이즈에 편중됐다는 지적은 **사실이다**: 기존 풀(COTRAIN_ALL(+TRAIN6LO)=ALLF)의 합성 42클립은 **clean 0개**, 전부 노이즈/압축. 다만 실데이터 117클립은 전부 clean이라 풀 전체로는 clean 117 : noisy 86. 즉 "안정·clean 조건은 실데이터에서만, 저심박·고심박 극단은 합성에서만" 배우는 구조였고, 그 결과 저심박(<55) 학습 클립이 실 13 + 합성 12개뿐이었다.
3. 이미 돌아간 **CoTA(PURE 01~06 포함)**의 PURE 피클을 재분석하니 held-out 07~10에서 MAE 0.49, 배가 0/24, GT<55 19건 MAE 0.09 — PURE 정체성(05번, 45.9 BPM)을 학습에 넣자 held-out 08·09의 배가가 사라졌다. 이는 "저심박 커버리지"가 원인이라는 결론을 in-domain 방식으로 확인한 것이다. 단 CoTA도 01-03(GT 75.9, 학습 피험자)에서 배가 1건이 남아 고조파 모호성 자체는 완전히 해소되지 않았다.
4. **COTRAIN_BAL** 209클립: 실 161(동일 분할) + 합성 48(TRAIN_BAL 36: BPM 40~160 10 BPM 빈마다 3클립, clean:light:medium 12:12:12, 실분포 정합 파형 + clean 12클립의 q6 압축 사본 12). 합성 BPM 히스토그램 전 빈 4로 평탄, held-out 누설 0. **v2(§3.5, 현재 디렉토리 상태)**: 저 a2(0~0.3) 보강 12클립(TRAIN_BAL_LOA2)을 더해 **221클립**, 합성 60(clean 20 : noisy 40), BPM 전 빈 5, a2 0~0.85 전 구간 커버.
5. 손실 개선안: GT f0 대비 2f0·0.5f0 대역 파워 비율 힌지 벌점 `L_h = relu(ratio − τ)` (`PF_HARM_LOSS=1` 게이트, λ_h=`PF_HARM_W`, τ=`PF_HARM_TAU` 기본 0.3). 패치 `scripts/patches/physformer_harm_loss.patch` (dry-run 적용 확인, 미적용 상태).

## 1. 학습 풀 구성 진단 (`scripts/analyze_train_pool_balance.py`)

클립별 전체표: `_workspace/data/results/train_pool_clips.md` (출처/도메인/조건/파형/a2/정체성/GT BPM/링크 대상), JSON `_workspace/data/results/train_pool_balance.json`, 그림 `train_pool_balance.png`.

대상 풀: `COTRAIN_ALL`(197, 다른 세션이 CoTA에 사용) + `TRAIN6LO`(6, 42~50 BPM; CoTAF용 COTRAIN_ALLF 디렉토리는 아직 없어 가상 결합) = **ALLF 203**. `readlink`로 확인한 링크 구성: subject1~41(34)→UBFC_dataset, 201~220→TRAIN20H(해석 파형 a2 그리드 0~0.75, **TRAIN20H2 아님**), 221~226→TRAIN6HI, 301~341→COTRAIN_NOISY_REAL, 501~510→CMP_REAL, 601~610→CMP_SYN(TRAIN20H 홀수 압축), 701~724/801~824→MCD_TRAIN(정체성 21~44 안정/운동직후), 9SSTT→PURE_TRAIN(01~06, 35 트라이얼).

### 1.1 요약 (기존 ALLF vs 신규 BAL)

| 항목 | ALLF (ALL+TRAIN6LO) | **COTRAIN_BAL** |
|---|---|---|
| 클립 수 | 203 | 209 |
| 실 : 합성 | 161 : 42 (3.8:1) | 161 : 48 (3.4:1) |
| 실 내 clean : 증강(noisy-real 34 + 압축 10) | 117 : 44 | 117 : 44 (동일; 상한 117 이내) |
| **합성 내 clean : noisy** | **0 : 42** | **12 : 36** (clean 12 / light 12 / medium 12 / 압축 12) |
| 풀 전체 clean : noisy | 117 : 86 | 129 : 80 |
| 도메인 | UBFC 78, MCD 48, PURE 35, SFHQ 42 | UBFC 78, MCD 48, PURE 35, SFHQ 48 |
| 파형 | real 161 / analytic 42 (TRAIN20H 그리드 a2 0~0.75: a2<0.3 이 11/20) | real 161 / analytic 48 (실분포 정합 a2 0.34~0.84) |
| 합성 BPM 범위 | 42~152 (52~118 집중, 90~100 빈 2개, 140+ 빈 1개씩) | **42~158, 12빈 × 4 평탄** |
| 합성 노이즈 강도 | 기존 그리드(중~강: motion 6~12px, illum 0.1~0.18, sensor 2~3) 만 | clean / 경(motion 3~4px, illum 0.05~0.06) / 중(기존 그리드 중앙값) |

### 1.2 BPM 히스토그램 (10 BPM 빈) — PURE steady 와 겹침

| BPM 빈 | PURE steady (10) | PURE 전체 (59) | ALLF 전체 / 합성 | **BAL 전체 / 합성** |
|---|---|---|---|---|
| 40-50 | **3** | 6 | 9 / 5 | 8 / 4 |
| 50-60 | 1 | 16 | 20 / 7 | 17 / 4 |
| 60-70 | 3 | 8 | 29 / 4 | 29 / 4 |
| 70-80 | 1 | 19 | 25 / 6 | 23 / 4 |
| 80-90 | 1 | 3 | 19 / 3 | 20 / 4 |
| 90-100 | 0 | 0 | 19 / 2 | 21 / 4 |
| 100-110 | 0 | 1 | 29 / 6 | 27 / 4 |
| 110-120 | 0 | 0 | 34 / 3 | 35 / 4 |
| 120-130 | 0 | 0 | 12 / 2 | 14 / 4 |
| 130-140 | 1 | 5 | 4 / 2 | 6 / 4 |
| 140-150 | 0 | 1 | 1 / 1 | 4 / 4 |
| 150-160 | 0 | 0 | 1 / 1 | 4 / 4 |

- PURE steady GT: 47.3, 45.9, 46.5, 52.9, 59.5, 63.3, 65.7, 68.9, 78.8, 125.3 (하위 3건이 40~50 빈).
- 미커버 정량화(±3 BPM 이내 학습 클립 없음): ALLF·BAL 모두 PURE 59/steady 10 트라이얼 전부 커버(합성만으로도 0건 미커버). **즉 "빈 자체가 비어 있어서" 실패한 것은 아니다** — ALLF의 40~50 빈에는 실 4(PURE 05번·09번 아님; UBFC/MCD)+합성 5(TRAIN6LO)가 있다. 문제는 그 빈의 **밀도**(전체의 4%)와, 저심박에서 2차 고조파(90~100 BPM)가 학습 분포의 최빈 구간(100~120 빈 63클립)과 정확히 겹쳐 사전학습 편향을 이기지 못한다는 점이다. BAL은 합성만으로 40~50 빈을 4개(+실 4)로 유지하면서 100~120 빈의 합성 편중(9→8)을 줄였다. 실데이터는 여전히 60~120에 몰려 있으므로(변경 불가), **합성이 BPM 밀도를 평탄화하는 유일한 수단**이다.

주의: 풀 전체 histogram의 40-50 빈이 ALLF 9 → BAL 8로 1 줄어든 것은 TRAIN6LO(42~50 5클립 + 37.5 BPM 1클립)를 BAL의 40~50 빈 4클립(42/45/48 + 압축 사본)으로 대체했기 때문. 더 두텁게 하려면 §3.4 옵션 참조.

## 2. PURE 안정부 오차 원인 진단 (기존 피클 재분석, 모델 재실행 없음)

피클: `runs/ev_{base,T2L_s100,CoTN,CoTNH,CoTNHM,CoTA}_PURE/real_PURE/saved_test_outputs/*.pickle` (59 트라이얼, 청크 160). 태스크는 PURE 파일명 `xx-0y`의 y(1 steady, 2 talking, 3 slow-tr, 4 fast-tr, 5 small-rot, 6 medium-rot). 두 지표: **tb** = toolbox 프로토콜(cumsum→detrend→butter 0.6–3.3→periodogram, 청크 결합), **hi** = 영상 단위 고해상 FFT(n×8 zero-pad, 0.7–4 Hz; 16 보고서 정본).

### 2.1 steady 10 트라이얼: 예측 BPM (hi-res)

| 트라이얼 | 분할 | GT | base | T2L | CoTN | CoTNH | CoTNH-M | CoTA |
|---|---|---|---|---|---|---|---|---|
| 01-01 | train | 65.7 | 65.9 | 65.9 | 66.0 | 65.9 | 65.9 | 65.9 |
| 02-01 | train | 68.9 | 68.9 | 69.0 | 68.9 | 68.9 | 68.9 | 68.9 |
| 03-01 | train | 52.9 | 53.0 | 53.0 | **104.5 ×2** | 53.0 | 53.0 | 53.0 |
| 04-01 | train | 59.5 | 59.0 | 59.0 | 59.1 | 59.0 | 59.0 | 59.0 |
| 05-01 | train | **45.9** | **90.3 ×2** | **90.2 ×2** | **93.3 ×2** | **90.5 ×2** | **90.5 ×2** | 45.8 |
| 06-01 | train | 63.3 | 63.5 | 63.0 | 62.8 | 62.9 | 62.8 | 62.8 |
| 07-01 | held-out | 125.3 | 125.3 | 125.4 | 125.4 | 125.3 | 125.3 | 125.3 |
| 08-01 | held-out | **46.5** | **93.5 ×2** | **91.2 ×2** | **91.3 ×2** | **93.6 ×2** | **93.6 ×2** | 46.4 |
| 09-01 | held-out | **45.2** | **90.5 ×2** | **90.6 ×2** | **90.3 ×2** | **90.5 ×2** | 45.2 | 45.2 |
| 10-01 | held-out | 78.8 | 78.6 | 81.9 | 84.6 | 78.6 | 78.6 | 84.5 |

("train"은 CoTA만 해당 피험자를 학습에 사용; 나머지 모델에는 59건 전부 제로샷.)

### 2.2 모델별 요약

| 모델 | 59 hi-MAE | 59 tb-MAE | 배가 | steady hi/tb MAE | steady 배가 | held-out 24 hi/tb | held-out 배가 | **GT<55 (19건) MAE / 배가** | **GT≥55 (40건) MAE** |
|---|---|---|---|---|---|---|---|---|---|
| base | 8.11 | 10.67 | 9/59 | 13.78 / 13.8 | 3/10 | 10.75 / 16.5 | 5/24 | 23.41 / 9 | 0.84 |
| T2L_s100 | 8.46 | 13.85 | 9/59 | 13.88 / 13.97 | 3/10 | 14.73 / 20.87 | 7/24 | 24.40 / 9 | 0.89 |
| CoTN | 12.70 | 13.70 | 14/59 | 19.61 / 19.25 | 4/10 | 21.13 / 22.39 | 10/24 | 37.78 / 14 | 0.79 |
| CoTNH | 10.80 | 13.83 | 9/59 | 13.82 / 19.25 | 3/10 | 19.18 / 21.62 | 7/24 | 24.64 / 8 | 4.23 |
| CoTNH-M | 6.56 | 10.64 | 6/59 | 9.31 / 13.8 | 2/10 | 10.85 / 17.05 | 5/24 | 18.19 / 6 | 1.03 |
| CoTA (PURE 01~06 학습) | 1.54 | 0.48 | 1/59 | 0.73 / 0.18 | 0/10 | **0.49 / 0.48** | **0/24** | **0.09 / 0** | 2.24 |

태스크별 hi-MAE(배가 수): base steady 13.78(3) / talking 11.84(2) / slow-tr 5.36(1) / fast-tr 7.84(1) / small-rot 5.44(1) / medium-rot 4.78(1). 태스크별 GT 하한: steady 47.3, talking 51.4, slow 49.8, fast 50.1, small-rot 49.7, medium-rot 50.9.

### 2.3 결론

- **(a) 저심박 배가가 원인이다.** 모든 제로샷 모델에서 GT≥55 BPM 40건의 MAE는 0.8~1.0(CoTNH만 4.2)이며 steady의 GT≥55 7건은 MAE 0.2 이하. 오차는 GT 45~48 BPM 피험자 5·8·9(그리고 일부 모델에서 50~55 BPM 피험자 3)의 정확한 ×2 예측에 전부 몰려 있고, 배가 트라이얼의 2f0/f0 파워비는 1.0~3.2(정상 트라이얼 0.05~0.5). 안정 조건의 시각 특성(정면·균일 조명)은 문제가 아니다 — 같은 피험자가 움직임 태스크에서도 동일하게 배가된다.
- **(b) 다른 원인은 부차적**: 10-01(78.8)에서 CoTN/T2L/CoTA가 +3~6 BPM 상향(고조파 아님, 스펙트럼 확산), CoTA의 01-03(75.9→151.5) 1건 배가(학습 피험자인데도 발생 → 커버리지만으론 모호성이 완전히 안 사라짐 → 손실 개선 근거). tb 지표는 hi보다 크게 나오는데(예: CoTNH-M 10.64 vs 6.56) 이는 160프레임 청크 FFT 양자화(11.25 BPM 빈)와 청크별 배가 혼입 때문이며, 09-01의 CoTNH-M처럼 hi는 정답(45.2)인데 tb는 90.5로 나오는 사례가 있다 → 보고 정본은 hi, tb는 병기.
- 커버리지 가설의 직접 검증: CoTA(PURE 05번 45.9 BPM 클립 6개 학습)는 학습에 없는 08·09번(46.5/45.2)도 정답 → 저심박 예시가 충분하면 배가가 사라진다. CoTNH-M→CoTA 사이에 TRAIN6LO는 없고 PURE_TRAIN 35클립만 추가됐으므로 개선은 PURE in-domain 데이터(그중 저심박 6클립) 덕이다. BAL은 이를 도메인 무관 합성으로 채우는 것이 목표.

## 3. COTRAIN_BAL 설계·조립

### 3.1 합성 TRAIN_BAL (`configs/train_bal.json`, 생성기 `scripts/gen_train_bal_config.py`)

- 12빈(40~160) × 3클립, BPM = 빈하한 + {2,5,8}; 조건 = (클립순번 + 빈번호) mod 3 → 각 빈에 clean/light/medium 하나씩, BPM 오프셋과 조건이 교락되지 않음. 총 36 = clean 12 / light 12 / medium 12.
- 정체성 SFHQ 5명(01,02,03,05,06) 순환 — 평가 C/E 세트(sfhq04·diff_* 정체성, 시드 301~424)와 분리. 시드 1001~1036(기존 101~120/501~566, 평가 300~400대, 실노이즈 800대와 분리). 공간 uniform/patches(coverage 0.4~0.6) 교대 18:18.
- 파형(실분포 정합, `train_h2_proposal.json` 규약): a2 = 실 PPG 분위수 함수(p5 0.342 … p95 0.842)의 36 균등 분위를 셔플, φ2~N(0.084,0.415), a3~N(0.293,0.091) clip[0,0.5], φ3~N(−0.279,0.602); rng seed 19.
- 경노이즈 3종 순환: {illum 0.06+flicker 0.012@1.2Hz} / {motion 4px, rot 0.8°} / {illum 0.05, flicker 0.01, motion 3px, rot 0.6°, sensor 1}. 중노이즈 3종: {illum 0.15, flicker 0.03@1.1, sensor 2} / {motion 10px, rot 2°, sensor 2} / {illum 0.12, flicker 0.025@0.95, motion 8px, rot 1.5°, sensor 2}. clean은 noise 없음(주입만).
- 생성: `build_synth_dataset.py`, CPU nice 10, **36클립 / 약 6분(13:14→13:20)**, 2.4 GB FFV1(1008프레임 256², 클립당 ~67 MB). 로그 `runs/build_TRAIN_BAL/stdout.log`.
- 압축 사본 `CMP_SYN_BAL`(9.7 MB): TRAIN_BAL clean 12클립의 mpeg4 q6(`compress_aug.compress_copy`, 기존 CMP_SYN* 와 동일 규약) → 합성 조건 4종 12:12:12:12.

### 3.2 TRAIN_BAL 검증 (`_workspace/data/results/cotrain_bal_manifest.json`; a2 적합 = GT 스펙트럼 √(P(2f0)/P(f0)))

| id | 링크 | 목표 | achieved | peak=f0 | 조건 | a2 요청 | a2 적합 | a3 요청 | a3 적합 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1001 | 42 | 42.19 | Y | clean | 0.498 | 0.500 | 0.343 | 0.340 |
| 2 | 1002 | 45 | 45.09 | Y | light | 0.392 | 0.402 | 0.335 | 0.342 |
| 3 | 1003 | 48 | 47.99 | Y | medium | 0.382 | 0.376 | 0.252 | 0.246 |
| 4 | 1004 | 52 | 52.01 | Y | light | 0.534 | 0.538 | 0.181 | 0.187 |
| 5 | 1005 | 55 | 54.91 | Y | medium | 0.439 | 0.434 | 0.379 | 0.375 |
| 6 | 1006 | 58 | 58.04 | Y | clean | 0.456 | 0.455 | 0.348 | 0.342 |
| 7 | 1007 | 62 | 62.05 | Y | medium | 0.676 | 0.670 | 0.127 | 0.132 |
| 8 | 1008 | 65 | 64.96 | Y | clean | 0.372 | 0.372 | 0.228 | 0.225 |
| 9 | 1009 | 68 | 68.08 | Y | light | 0.704 | 0.707 | 0.384 | 0.385 |
| 10 | 1010 | 72 | 72.10 | Y | clean | 0.586 | 0.592 | 0.390 | 0.390 |
| 11 | 1011 | 75 | 75.00 | Y | light | 0.560 | 0.560 | 0.235 | 0.235 |
| 12 | 1012 | 78 | 77.90 | Y | medium | 0.464 | 0.463 | 0.498 | 0.502 |
| 13 | 1013 | 82 | 81.92 | Y | light | 0.732 | 0.731 | 0.421 | – |
| 14 | 1014 | 85 | 85.04 | Y | medium | 0.814 | 0.814 | 0.380 | – |
| 15 | 1015 | 88 | 87.95 | Y | clean | 0.447 | 0.447 | 0.281 | – |
| 16 | 1016 | 92 | 91.96 | Y | medium | 0.489 | 0.486 | 0.166 | – |
| 17 | 1017 | 95 | 95.09 | Y | clean | 0.547 | 0.552 | 0.367 | – |
| 18 | 1018 | 98 | 97.99 | Y | light | 0.472 | 0.472 | 0.455 | – |
| 19 | 1019 | 102 | 102.01 | Y | clean | 0.649 | 0.652 | 0.376 | – |
| 20 | 1020 | 105 | 104.91 | Y | light | 0.430 | 0.429 | 0.195 | – |
| 21 | 1021 | 108 | 108.04 | Y | medium | 0.787 | 0.786 | 0.397 | – |
| 22 | 1022 | 112 | 112.05 | Y | light | 0.521 | 0.517 | 0.232 | – |
| 23 | 1023 | 115 | 114.96 | Y | medium | 0.599 | 0.596 | 0.093 | – |
| 24 | 1024 | 118 | 118.08 | Y | clean | 0.508 | 0.511 | 0.322 | – |
| 25 | 1025 | 122 | 122.10 | Y | medium | 0.402 | 0.404 | 0.046 | – |
| 26 | 1026 | 125 | 125.00 | Y | clean | 0.573 | 0.573 | 0.303 | – |
| 27 | 1027 | 128 | 127.90 | Y | light | 0.352 | 0.350 | 0.385 | – |
| 28 | 1028 | 132 | 131.92 | Y | clean | 0.422 | 0.422 | 0.387 | – |
| 29 | 1029 | 135 | 135.04 | Y | light | 0.612 | 0.611 | 0.500 | – |
| 30 | 1030 | 138 | 137.95 | Y | medium | 0.362 | 0.362 | 0.354 | – |
| 31 | 1031 | 142 | 141.96 | Y | light | 0.412 | 0.410 | 0.204 | – |
| 32 | 1032 | 145 | 145.09 | Y | medium | 0.342 | 0.346 | 0.349 | – |
| 33 | 1033 | 148 | 147.99 | Y | clean | 0.759 | 0.758 | 0.355 | – |
| 34 | 1034 | 152 | 152.01 | Y | medium | 0.842 | 0.843 | 0.240 | – |
| 35 | 1035 | 155 | 154.91 | Y | clean | 0.481 | 0.481 | 0.134 | – |
| 36 | 1036 | 158 | 158.04 | Y | light | 0.624 | 0.623 | 0.259 | – |

achieved BPM 오차 ≤0.2, 36/36 스펙트럼 피크 = 기본파, a2 적합 오차 ≤0.01, 전 클립 ffv1 1008프레임. (a3 적합은 3f0>240 BPM이면 산출 안 함.)

### 3.3 조립 (`scripts/assemble_cotrain_bal.py`, 전부 심볼릭링크, 기존 디렉토리 무수정)

| 링크 범위 | 대상 | 수 | 비고 |
|---|---|---|---|
| subject1~41 | `UBFC_dataset/DATASET_2` | 34 | COTRAIN_ALL과 동일 34명(2,6,7,19,21,28,29 제외) |
| subject301~341 | `COTRAIN_NOISY_REAL` | 34 | 실+노이즈 증강, 정답 불변 |
| subject501~510 | `CMP_REAL` | 10 | UBFC 1,3,4,5,8,9,10,11,12,13 q6 |
| subject601~612 | `CMP_SYN_BAL` (신규) | 12 | TRAIN_BAL clean 12 q6 |
| subject701~724 / 801~824 | `MCD_TRAIN` | 48 | 정체성 21~44 안정/운동직후 (COTRAIN_ALL과 동일) |
| subject9SSTT | `PURE_TRAIN` | 35 | 피험자 01~06 전 태스크(06-02 원본 결측) — 다른 세션 분할 그대로 |
| subject1001~1036 | `TRAIN_BAL` (신규) | 36 | |
| **합계 (v1)** | | **209** | 무결성 209/209 (vid+GT) |
| subject1201~1212 | `TRAIN_BAL_LOA2` (v2, §3.5) | 12 | 저 a2 0~0.3 보강 |
| **합계 (v2, 현재)** | | **221** | 무결성 221/221 |

실 증강 상한: 실 clean 117 ≥ 증강 44(noisy-real 34 + 압축 10) → 조건 충족.

### 3.4 누설 검증

학습 정체성 vs held-out 교집합(manifest `leak_check`): UBFC ∅ (학습 34명 vs 42~49), MCD ∅ (21~44 vs 평가 20명 1020…1275), PURE ∅ (01~06 vs 07~10), 합성 얼굴 ∅ (sfhq_01/02/03/05/06 vs sfhq04·diff_*), 합성 시드 ∅ (1001~1036 vs 301~424). VALID는 기존 yaml과 같이 TRAIN20 마지막 10%(subject19–20, 학습 풀에 없음)이고 `USE_LAST_EPOCH: True`라 모델 선택에 관여하지 않는다.

옵션(미적용): 40~50 빈을 더 두텁게 하려면 `TRAIN6LO`(42~50, 6클립, 시드 561~566, TRAIN20H 파형 규약)를 `subject1101~1106`으로 추가 링크. 합성 BPM 평탄성은 깨지지만 PURE 서맥 편중에 대한 가중이 된다. CoTB 결과에서 GT<55 배가가 남으면 CoTB2로 시도.

### 3.5 저 a2 보강(v2) — TRAIN_BAL_LOA2 (2026-08-25 추가, CPU 빌드만)

**근거.** 17 보고서 §6c: 실분포 정합 파형(a2 0.34~0.85)만 쓴 CoTNH2는 안정 MCD에서 CoTNH(a2 0~0.75 그리드, a2<0.3 클립 11/20)보다 유의하게 악화(MAE 18.90 vs 5.19, Wilcoxon p=.049, 배가 재출현 2건). 해석: 저 a2(정현파에 가까운) 클립은 "고조파가 약할 때 1f가 정답"이라는 기본파–고조파 대비를 모호하지 않게 가르쳐 2f 락인을 억제한다 — 분포 폭·대비가 실분포 밀도 정합보다 중요. v1 TRAIN_BAL(§3.1)은 실분포 정합 규약이라 a2<0.34 구간이 **0클립**이었으므로(CoTNH2와 같은 결함), 그 구간을 보강한다. 이는 §6c 가 요구한 "a2 구간별 통제 실험(a2<0.3 만 추가)"의 학습 풀에 해당한다: CoTB(v1 209) 대비 CoTB(v2 221)의 차이는 저 a2 12클립뿐.

**구성** (`scripts/gen_train_bal_loa2_config.py` → `configs/train_bal_loa2.json`, 빌드 로그 `runs/build_TRAIN_BAL_LOA2/stdout.log`, 12클립 674 MB FFV1, 약 2분).

- 12클립 = TRAIN_BAL 과 같은 12빈(40~160) 하나씩; BPM = 빈하한 + {4,6,3,7} 순환(TRAIN_BAL 의 +2/+5/+8 과 빈 내 불일치 → 44,56,63,77,84,96,103,117,124,136,143,157).
- 파형: a2 ∈ {0.0, 0.1, 0.2, 0.3} 순환(각 3클립, 빈과 교락 없음), φ2 ~ N(0.084, 0.415) 실분포 표본, a3 ∈ {0, 0.1} 교대, φ3 ~ N(−0.279, 0.602); rng seed 23.
- 조건 **clean 8 : light 4** (k%3==1 → light, 경노이즈 3종 순환; light 4클립의 a2는 0.1/0.0/0.3/0.2로 a2–조건 교락 없음). medium·압축 없음 — v1에서 합성 clean:noisy 12:36 이던 편중을 20:40 으로 완화.
- 정체성 SFHQ 5명(01,02,03,05,06) 순환(3/3/2/2/2), 공간 uniform/patches 교대 6:6, α 0.01.
- 시드 **1201~1212**: 기존 configs/*.json 전수(101~120, 301~354, 401~424, 501~566, 1001~1036; 124개) 및 실노이즈 700·800대와 불일치(생성기가 assert 로 확인). 링크 번호 subject1201~1212 (1101~1106 은 §3.4 TRAIN6LO 옵션용으로 비워 둠).

**검증** (`scripts/assemble_cotrain_bal.py` 재실행, manifest `train_bal_verification` src=TRAIN_BAL_LOA2; a2 적합 = GT 스펙트럼 √(P(2f0)/P(f0)))

| id | 링크 | 목표 | achieved | peak=f0 | 조건 | 노이즈 | a2 요청 | a2 적합 | a3 요청 | a3 적합 | 정체성 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1201 | 44 | 43.97 | Y | clean | – | 0.0 | 0.012 | 0.0 | 0.006 | sfhq_01 |
| 2 | 1202 | 56 | 56.03 | Y | light | illum 0.06 + flicker 0.012@1.2 | 0.1 | 0.101 | 0.1 | 0.098 | sfhq_02 |
| 3 | 1203 | 63 | 62.95 | Y | clean | – | 0.2 | 0.203 | 0.0 | 0.005 | sfhq_03 |
| 4 | 1204 | 77 | 77.01 | Y | clean | – | 0.3 | 0.301 | 0.1 | 0.103 | sfhq_05 |
| 5 | 1205 | 84 | 83.93 | Y | light | motion 4px, rot 0.8° | 0.0 | 0.005 | 0.0 | – | sfhq_06 |
| 6 | 1206 | 96 | 95.98 | Y | clean | – | 0.1 | 0.098 | 0.1 | – | sfhq_01 |
| 7 | 1207 | 103 | 102.90 | Y | clean | – | 0.2 | 0.197 | 0.0 | – | sfhq_02 |
| 8 | 1208 | 117 | 116.96 | Y | light | illum 0.05, flicker 0.01, motion 3px, rot 0.6°, sensor 1 | 0.3 | 0.299 | 0.1 | – | sfhq_03 |
| 9 | 1209 | 124 | 124.11 | Y | clean | – | 0.0 | 0.003 | 0.0 | – | sfhq_05 |
| 10 | 1210 | 136 | 135.94 | Y | clean | – | 0.1 | 0.102 | 0.1 | – | sfhq_06 |
| 11 | 1211 | 143 | 143.08 | Y | light | illum 0.06 + flicker 0.012@1.2 | 0.2 | 0.201 | 0.0 | – | sfhq_01 |
| 12 | 1212 | 157 | 156.92 | Y | clean | – | 0.3 | 0.300 | 0.1 | – | sfhq_02 |

achieved BPM 오차 ≤0.11, 12/12 스펙트럼 피크 = 기본파(a2=0 클립에서도 2f0 잔여 파워 ≤0.012 → 순수 정현파 확인), a2 적합 오차 ≤0.003, 전 클립 ffv1 1008프레임.

**COTRAIN_BAL v2 균형 요약** (manifest `balance`; 링크 221/221 무결)

| 항목 | v1 (209) | **v2 (221)** |
|---|---|---|
| 실 : 합성 | 161 : 48 (3.4:1) | 161 : 60 (2.7:1) |
| 실 clean : 증강 | 117 : 44 | 117 : 44 (불변) |
| 합성 clean : noisy | 12 : 36 (clean 12 / light 12 / medium 12 / 압축 12) | **20 : 40** (clean 20 / light 16 / medium 12 / 압축 12) |
| 풀 전체 clean : noisy | 129 : 80 | 137 : 84 |
| 합성 BPM 히스토그램(10 BPM 빈, 40~160) | 12빈 × 4 | **12빈 × 5** (평탄 유지) |
| 합성 a2 히스토그램(0.1 폭) | 0.3–0.4: 10, 0.4–0.5: 17, 0.5–0.6: 12, 0.6–0.7: 5, 0.7–0.8: 5, 0.8–0.9: 2 (**<0.3: 0**) | **0.0–0.1: 3, 0.1–0.2: 3, 0.2–0.3: 3**, 0.3–0.4: 10, 0.4–0.5: 17, 0.5–0.6: 12, 0.6–0.7: 5, 0.7–0.8: 5, 0.8–0.9: 2 → 0~0.85 전 구간 커버 |
| 합성 정체성 | SFHQ 5명 | SFHQ 5명(동일) |

합성 내 a2<0.3 비율은 9/60 = 15%(CoTNH 풀의 55%보다 낮음) — 실분포 밀도(0.34~0.84)를 주로 유지하면서 대비 학습용 저 a2 구간만 채우는 절충. 효과가 없으면 저 a2 클립 수를 늘리는 것이 다음 단계(TRAIN_BAL_LOA2 config 의 a2 순환을 확장).

**누설 재검증**: UBFC ∅, MCD ∅, PURE ∅, 합성 얼굴 ∅(sfhq_01/02/03/05/06 vs sfhq04·diff_*), 합성 시드 ∅(1001~1036, 1201~1212 vs 평가 301~424). 기존 링크(1~41, 301~341, 501~510, 601~612, 701~824, 9SSTT, 1001~1036)는 재실행 시 realpath 동일 확인 후 그대로 두었고 CMP_SYN_BAL 사본도 재생성하지 않았다(기존 데이터셋 무수정). `--no-loa2` 로 실행하면 subject12xx 링크를 제거해 v1(209)로 되돌릴 수 있다.

**yaml/캐시 확인**: `yaml13/finetune_CoTB*.yaml` 은 DATA_PATH=COTRAIN_BAL, EXP_DATA_NAME=cotrain_bal 이며 `cache/cotrain_bal`·`cache/DataFileLists/*cotrain_bal*` 모두 **아직 없음**(runs/finetune_CoTB 도 없음) → 첫 학습이 v2 221클립으로 전처리하므로 이름 변경 불필요(그대로 유효). `run_phase8_bal.sh` 에 안전장치 추가: 시작 시 링크 수가 `EXPECT_LINKS`(기본 221)와 다르거나, `cache/cotrain_bal` 이 이미 있는데 캐시 subject 수 ≠ 링크 수(옛 v1 캐시)이면 중단하고 `cotrain_bal_v2` 로 이름 변경 또는 캐시 삭제를 안내한다. 예상 학습 시간은 221/209 = +6%(≈1,326 청크/epoch).

## 4. 고조파 모호성 억제 손실 (패치 파일만, **미적용**)

파일: `scripts/patches/physformer_harm_loss.patch` (unified diff, `repos/rPPG-Toolbox` 루트에서 `-p1`). 기존 `[Virtual-Face patch]` SiNC 블록(`SINC_PRIOR` 환경변수 게이트) 바로 뒤에 같은 스타일로 삽입한다. `patch --dry-run` 통과, 패치본 `py_compile` 통과 확인.

### 4.1 기존 손실 (PhysFormerTrainer.train)

`loss = a·NegPearson(rPPG, label) + b·(CE_powerspectrum + KL_DLDL)`; a=1, b=1(epoch≤10, 10 epoch 학습이므로 항상). 주파수 항은 라벨 청크 welch 피크 HR(`get_hr`, 30~180 BPM, 40~179 클램프)을 클래스 타깃으로 하는 CE+DLDL — 즉 **이미 f0를 맞추도록 감독**하지만, 예측 스펙트럼이 f0와 2f0에 파워를 나눠 가질 때 CE는 소프트맥스 확률로만 벌하고 2f0 파워 자체를 명시적으로 줄이지는 않는다(하위 배가 청크의 2f0 확률이 f0 확률보다 조금만 낮아도 손실은 작다). SiNC 블록(옵션)은 대역 밖 파워와 피크 비희소성을 벌하되 라벨을 안 쓴다 → 배가 피크도 "희소"하면 통과.

### 4.2 신규 항

```
psd   = |rfft(rPPG)|²                       (B, F)
f0    = hr_label / 60                       (B,)  — 트레이너의 기존 hr (welch 피크, 클램프 후)
P(fc) = Σ_f psd · exp(−(f−fc)²/2σ²) · 1[0.5 ≤ fc ≤ 4.0]     σ = PF_HARM_SIGMA_HZ (기본 0.1 Hz)
ratio = (P(2f0) + P(f0/2)) / (P(f0) + P(2f0) + P(f0/2) + ε)   ∈ [0,1]
L_h   = mean_b relu(ratio − τ)              τ = PF_HARM_TAU (기본 0.3)
loss += λ_h · L_h                            λ_h = PF_HARM_W (기본 1.0), PF_HARM_LOSS=1 일 때만
```

- 정규화를 세 대역 파워 합으로만 하므로 다른 주파수 성분에는 압력이 없고 **기본파 vs 고조파/서브하모닉의 상대 비율**만 조정한다(NegPearson·CE와 충돌 최소).
- 힌지 τ=0.3인 이유: 실 PPG는 정상적으로 2차 고조파를 가진다(a2 중앙값 0.50 → 파워비 0.25, p75 0.63 → 0.40·(1+...)). CPU 수치검증: f0 정현파 ratio 0.001, 2f0 정현파 0.998, 0.5f0 0.995, f0+0.6·2f0 혼합 0.223(τ 이하 → 벌점 0), f0=150(2f0=5 Hz 대역 밖) 0.004. 배가된 예측(ratio≈1)에는 0.7의 벌점, 정상 파형에는 0.
- 2f0 창이 4 Hz를 넘거나 f0/2 창이 0.5 Hz 아래면 해당 항 0 → 고심박·저심박 끝단에서 잘못된 벌점 없음.
- 한계: hr 라벨이 청크 welch 피크라 라벨 자체가 배가된 청크(실 PPG에 드묾; 클램프 로직 주석 참조)에서는 f0가 틀린 채로 벌한다 — 비율 기반이라 그 경우 손실은 "라벨 피크를 기본파로" 밀며 CE와 같은 방향이므로 새 오류원은 아니다. λ_h 튜닝 없이 1.0 고정 제안, 변화가 없으면 2.0.

### 4.3 적용/복원

```
cd repos/rPPG-Toolbox
patch -p1 --dry-run < ../../scripts/patches/physformer_harm_loss.patch   # 확인
patch -p1            < ../../scripts/patches/physformer_harm_loss.patch   # 적용
patch -R -p1         < ../../scripts/patches/physformer_harm_loss.patch   # 복원
```
환경변수 미설정 시 패치가 있어도 손실은 완전히 0(기존 학습과 동일 코드 경로). `run_phase8_bal.sh`는 CoTBH 학습 전 `grep PF_HARM_LOSS`로 적용 여부를 확인하고 없으면 자동 적용하며, `REVERT_PATCH=1`이면 종료 시 복원한다.

## 5. 실행 계획 (`yaml13/`, `scripts/run_phase8_bal.sh` — 실행하지 않음)

| 단계 | 산출 | 예상 소요 (RTX 5090, 과거 실측 기준) |
|---|---|---|
| 전처리 cotrain_bal 캐시 (첫 학습에서 `DO_PREPROCESS: True`; 두 번째 학습은 스크립트가 False로 전환) | `cache/cotrain_bal` | 221클립(v2) HC crop+DiffNorm ≈ 25~35분 |
| finetune_CoTB (COTRAIN_BAL, 손실 기본, LR 5e-5, batch 4, 10 epoch, 사전학습 UBFC PhysFormer 초기화) | `runs/finetune_CoTB/PreTrainedModels/SYNTH_CoTB_physformer_Epoch9.pth` | ≈1,254 청크 × 10 epoch ≈ 60~80분 (CoTA 197클립 대비 +6%) |
| finetune_CoTBH (동일 + `PF_HARM_LOSS=1 PF_HARM_W=1.0 PF_HARM_TAU=0.3`) | `runs/finetune_CoTBH/...Epoch9.pth` | 동일 + FFT 오버헤드 미미 |
| 평가 2모델 × 10세트 (MCD_A, MCD_B, REALTEST, RTNC, RTNI, C1, C3, C6, E3, PURE) | `runs/ev_{CoTB,CoTBH}_*/…/saved_test_outputs/*.pickle` | 세트당 1~4분, 총 ≈ 40분 |
| 분석 | `python scripts/analyze_train_pool_balance.py` → `pure_model_summary`에 CoTB/CoTBH 자동 포함; MCD/UBFC 세트는 `analyze_phase6.py --models` 확장 | CPU 수분 |

- PURE 평가 세트: 다른 세션과 동일하게 `PURE_RAW` 전체(59, 캐시 `real_PURE` 재사용)로 평가하고 **held-out 07~10(vid≥700, 24 트라이얼)** 을 정본으로, 01~06은 "학습 피험자"로 구분 보고한다(별도 PURE_TEST 디렉토리 없음 — `ev_*_PURE.yaml`).
- GPU 게이트: 학습 19,000 MiB, 평가 8,192 MiB 여유 확인 후 시작, 60초 폴링; OOM 시 1회 재시도.
- 명령:
```
bash scripts/run_phase8_bal.sh                      # CoTB → CoTBH → 평가 20건
MODELS="CoTB" bash scripts/run_phase8_bal.sh        # 기본 손실만
ONLY_EVAL=1 bash scripts/run_phase8_bal.sh          # 체크포인트 있을 때 평가만
PF_HARM_W=2.0 bash scripts/run_phase8_bal.sh        # 손실 가중 변경(CoTBH)
```
- 성공 기준: PURE held-out 07~10 hi-MAE < 1(CoTA 0.49 수준)이면서 **CoTA가 못 한 것** — 배가 0/59, MCD_B/MCD_A 3.4/6.0 유지, UBFC held-out ≤1.7, 합성 C/E ≈0. CoTBH − CoTB 차이는 배가 카운트와 2f0/f0 파워비 분포(`pow_2f0_over_f0`)로 본다.
- 정직한 한계: (1) BAL의 실:합성 3.4:1은 실데이터를 줄이지 않는 한 1:1로 못 만든다 — 실 BPM 편중(60~120)은 그대로이며 합성만 평탄. (2) 합성 정체성 5명·1008프레임이라 합성 다양성은 여전히 작다. (3) CoTA와 BAL의 차이는 합성 42→48 교체 하나이므로(실 부분 동일) CoTB−CoTA 비교가 "합성 균형화" 단독 효과를 준다 — 단 CoTA는 MCD 정체성·PURE 01~06 in-domain이므로 그 열들은 제로샷이 아님을 계속 구분 표기.

## 6. 산출물

- 스크립트: `scripts/analyze_train_pool_balance.py`(진단 1·2), `scripts/gen_train_bal_config.py`, `scripts/gen_train_bal_loa2_config.py`(v2), `scripts/assemble_cotrain_bal.py`(v2: LOA2 링크·균형표, `--no-loa2`), `scripts/run_phase8_bal.sh`, `scripts/patches/physformer_harm_loss.patch`
- 데이터: `configs/train_bal.json`, `TRAIN_BAL/`(36, 2.4 GB), `CMP_SYN_BAL/`(12, 9.7 MB), `configs/train_bal_loa2.json`, `TRAIN_BAL_LOA2/`(12, 674 MB), `COTRAIN_BAL/`(**221 링크, v2**), `runs/build_TRAIN_BAL/stdout.log`, `runs/build_TRAIN_BAL_LOA2/{stdout.log,assemble_v2.log}`
- 결과: `_workspace/data/results/train_pool_balance.{png,json}`, `train_pool_clips.md`, `cotrain_bal_manifest.json`
- yaml: `yaml13/finetune_CoTB.yaml`, `finetune_CoTBH.yaml`, `ev_{CoTB,CoTBH}_{MCD_A,MCD_B,REALTEST,RTNC,RTNI,C1,C3,C6,E3,PURE}.yaml` (22개)
