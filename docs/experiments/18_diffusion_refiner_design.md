# 18단계 — Diffusion Waveform Refinement Head 설계서 (체인 31)

> 2026-08-31 시작. 합성 데이터 라인은 17단계에서 종료(최종 채택 CoTAF-TW-L). 18단계부터는 **모델 측 고도화**.
> 핵심 가설: TW가 데이터 경로로 증명한 "실 맥파 morphology를 알려주면 halving이 고쳐진다"를, **추론 경로의 학습된 사전(conditional diffusion prior)**으로 재주입하면 잔여 실패(halving 12건, MMPD 10.31 vs CV 상한 9.26의 격차 1.05)를 추가로 줄일 수 있다.

## 1. 구조

- **Backbone 동결**: CoTAF-TW-L(Epoch9, 시드 100/200/300) 가중치·평가 pickle을 그대로 사용. head는 backbone의 noisy BVP를 조건으로 파형을 재구성하는 후단 모듈.
- **평가는 전부 post-hoc**: 기존 19개 평가 run(13축 + MMPD B1~B6)의 saved_test_outputs pickle에 head를 적용해 refined pickle을 만들고, 기존 `agg_model2.py`/`agg_mmpd.py`/`cmp_model.py`를 모델명 `CoTAF_TW_DIFF_L{,_s200,_s300}`으로 그대로 실행 — GPU 재평가·캐시 재생성 불필요, 프로토콜 변화 없음.

## 2. 신호 공간 ("pulse space")

pred/label 모두 DiffNormalized(도함수) chunk → full-video 연결 후:
`cumsum → detrend(λ=100, sparse O(N)) → butter1 0.6–3.3 Hz(filtfilt) → 창 단위 z-score`
(정본 HR 프로토콜의 전처리와 동일 계열 — head 출력이 곧바로 정본 프로토콜에 들어가는 형태가 되도록). refined full 신호는 `np.diff(prepend)`로 도함수 공간으로 되돌려 원본과 동일한 160-샘플 chunk 구조로 저장.

- 창: **L=300 (10 s @30 Hz)**, stride 150, 추론 시 Hann overlap-add(꼬리는 끝-정렬 창 추가).

## 3. 학습 쌍

- 소스: **공식 코퍼스 COTRAIN_ALLF_TW(203클립, 학습 분할만)**에 backbone only_test 1회 실행(DO_PREPROCESS=true, 완료 후 캐시 즉시 삭제) → (pred, GT) full 파형 쌍 덤프. held-out·zero-shot 축(UBFC 42–49, PURE 07–10, MMPD 7–33, DLCN, MR-NIRP)은 **어떤 형태로도 불사용**. MMPD 1–6도 1차 head에는 넣지 않음(backbone과 동일 분포 유지 — TW의 zero-shot 성격 보존).
- 게이트(타깃 창): fundamental-dominant — f0 ∈ [0.75, 3.5] Hz & p(2f) ≤ 0.9·p_max (`build_ppg_bank.py`와 동일 규칙, 라벨 안전).
- 분할: vid 번호 %10==0 → 내부 검증(≈10%, 하이퍼파라미터 선택 전용; 성능 보고에는 불사용).

## 4. 모델·diffusion 사양

| 항목 | 값 |
|---|---|
| 아키텍처 | 1-D conditional U-Net: in 2ch(x_t, cond), base 64, mult (1,2,4), 레벨당 res block 2(GroupNorm8+SiLU+Conv k5), time = sinusoidal 128→MLP FiLM | 
| Diffusion | T=1000, cosine ᾱ, ε-prediction, MSE |
| CFG | p_uncond=0.1(조건 0으로 드롭), guidance w ∈ {1.0, 1.5, 2.0} 내부 검증으로 선택 |
| 샘플링 | DDIM 50 steps, η=0, video명 해시 시드(결정적) |
| 학습 | AdamW 2e-4, batch 256, 20k steps, EMA 0.999, RPPG_SEED=100 |
| 검증 지표 | 창 단위 hi-res HR MAE(8× zero-pad, 0.6–3.3) — refined vs GT, baseline은 cond 자체의 MAE |

## 5. 판정 기준 (사전 등록)

1. **주장**: MMPD held-out 540클립 Wilcoxon, CoTAF-TW-DIFF-L vs CoTAF-TW-L(10.51) p<0.05 개선 + halving 12 감소.
2. **비용 검사**: DLCN·MR-NIRP 3축·PURE·MCD 등 나머지 축 무손실(유의 악화 없음) — TS/TC를 기각시킨 기준 그대로.
3. **시드 강건성**: head(시드 100 backbone 쌍으로 1회 학습)를 s200/s300 backbone pickle에도 적용, 방향 일치 확인.
4. 실패 시 그대로 보고(§16 정직 보고 원칙). 특히 신호가 거의 없는 축(Walking·Skin 6)에서 prior가 "지어낸" 맥파로 MAE만 좋아 보이는지 r(상관)·조건별 분해로 교차 확인.

## 6. 리스크와 완화

- prior 과신(무신호 클립에서 hallucination) → guidance w를 내부 검증으로 낮게 선택, r 병기.
- GT 자체가 2f-강한 MMPD 피험자 → 학습 타깃 게이트로 차단(위 §3), 평가 GT는 정본 프로토콜이 처리.
- 추론 비용 → DDIM 50 steps, 창 배치 처리로 전 축 수 분 수준. 실시간이 필요하면 후속(distillation).

## 7. 산출물·실행

- 코드: `scripts/diffusion_refiner.py` (subcommand: pairs / train / apply), 체인: `scripts/run_chain31_diffusion.sh` (GPU 게이트, 백그라운드)
- 중간물: `physformer/refiner/pairs.npz`, `refiner/ckpt.pt`, `refiner/meta.json`(선택된 w·검증 수치)
- 결과: `results/agg_CoTAF_TW_DIFF_L{,_s200,_s300}{,_MMPD}.json`, cmp 로그 → §18 결과 절에 기록

## 8. 확장 — diffusion 변형 스윕 (체인 32, 사용자 지시 2026-08-31)

기본 head(V1: ε-U-Net DDPM+CFG+DDIM50)와 동일한 쌍·검증·판정 기준으로 4개 변형을 비교한다. 학습 쌍·pulse space·평가(post-hoc pickle) 전부 공유 — 순수 "denoiser 설계" 요인 분해.

| tag | 모델명 | 무엇을 검정하나 |
|---|---|---|
| REG | CoTAF-TW-DREG-L | 동일 U-Net **직접 회귀**(diffusion 제거) — 이득이 diffusion 과정에서 오는가, 단순 morphology 매핑에서 오는가 (핵심 ablation) |
| X0 | CoTAF-TW-DX0-L | **x0-parameterization** — 저차원 신호에서 ε-pred 대비 유불리 |
| DIT | CoTAF-TW-DDIT-L | **DiT-1D**(patchify 10, dim 256, depth 6, AdaLN-zero) — U-Net 대비 transformer denoiser |
| W600 | CoTAF-TW-DW600-L | **20초 장창(L=600)** — 창이 길수록 harmonic 모호성(halving) 해소에 유리한가 |

코드: `scripts/diffusion_refiner_v2.py`(기반 모듈 import, 미수정), 체인: `run_chain32_diffusion_sweep.sh`(체인 31 완료 대기 후 순차, GPU 게이트). 판정: §5와 동일 + V1과의 상호 비교(cmp_model REF=CoTAF_TW_DIFF_L). 승자(있다면)만 s200/s300 시드 확증으로 진행.

## 9. 체인 31 결과 (V1 기본 head, 2026-08-31 21:24)

학습: 쌍 1745/검증 154창, 내부 검증 w*=1.0 (valMAE 0.127 < cond baseline 0.146).

| 시드 | MMPD held-out (TW → +DIFF) | Wilcoxon p | halving | doubling |
|---|---|---|---|---|
| 100 | 10.51 → 10.14 | .28 | 12 → **4** | 7 |
| 200 | 10.06 → 9.76 | .90 | 3 | 5 |
| 300 | 10.37 → 9.81 | .19 | 6 | 7 |
| 평균 | **10.31±0.19 → 9.90±0.21** | — | — | — |

조건별(시드 100): Walking 19.63→**16.34**, Skin 6 17.26→**14.57**, LED-low 11.61→10.23, Nature 12.11→11.06. NIRP still 8.51→7.77, MCD stable 0.53→0.26.
**비용**: PURE 0.34→**3.75**, UBFC held-out 1.43→2.53, RTNI 0.66→1.32, MCD 운동후 3.08→3.60 — **깨끗한(고SNR) 축에서 prior가 오히려 해를 입힘**(§6에서 예상한 hallucination 방향).

**판정(§5 기준): V1 채택 불가** — MMPD 비유의(p=.28) + 깨끗한 축 손실. 단, (i) halving 12→4로 형태 사전이 의도한 메커니즘은 실제로 작동, (ii) 3시드 모두 MMPD 개선 방향 일치, (iii) 이득은 저SNR 조건(Walking·Skin 6·LED-low)에 집중 — "어려운 클립에서만 사전이 유효"라는 명확한 패턴. 체인 32(REG/X0/DIT/W600)와, 필요시 저SNR 선택 적용(confidence-gated refinement)이 후속 후보.

## 10. 체인 32 결과 (변형 4종, 2026-08-31 23:34)

| 변형 | MMPD (TW 10.51 →) | p | hlv | Walking | Skin 6 | PURE (0.34 →) | NIRP large (9.99 →) | 특이 |
|---|---|---|---|---|---|---|---|---|
| V1 DIFF (참고) | 10.14 | .28 | 4 | 16.34 | 14.57 | 3.75 | 10.32 | — |
| REG (회귀, diffusion 제거) | 10.47 | .81 | 4 | 16.57 | 14.77 | 3.78 | **13.60** | 결정성이 고SNR 손실을 막지 못함 |
| X0 | 10.22 | .57 | 6 | 16.54 | 14.21 | **4.79** | 12.03 | |
| DIT (w*=2.0) | 10.19 | .31 | 6 | 16.98 | 14.62 | 3.83 | 9.81 | MCD 운동후 3.08→5.62 악화 |
| W600 (20 s 창) | 10.43 | .95 | 5 | 17.07 | 15.23 | 3.06 | 13.09 | 쌍 751/67로 감소, valMAE 0.60 |

**판정: 4종 전부 §5 기준 미달(전원 MMPD 비유의 + PURE 3.1~4.8 손실).** V1이 여전히 최선.

**요인 분해가 말해주는 것:**
1. **PURE 손실은 diffusion 고유 문제가 아니다** — 결정적 REG도 동일하게 잃음(0.34→3.78). 원인은 "morphology 매핑" 자체: 학습 쌍의 cond가 대부분 이미 깨끗해(cond baseline HR MAE 0.146) 사실상 항등에 가까운 과제를 배웠고, 깨끗한 입력에서의 미세한 재구성 오차가 고SNR 축의 0-오차를 깨뜨림. 반면 저SNR 입력(Walking·Skin 6)에서는 형태 사전이 실제로 작동(전 변형 halving 12→4~6, Walking −2.6~−3.3).
2. 창 연장(W600)·transformer(DIT)·x0-param 모두 이 구조적 한계를 바꾸지 못함 — denoiser 설계가 아니라 **적용 정책**의 문제.
3. 따라서 다음 단계 후보는 **confidence-gated selective refinement**: 저SNR 클립(backbone 출력의 스펙트럼 SNR 기준)에만 refine을 적용하고 고SNR은 원본 유지 — 학습 불필요, 기존 refined pickle과 원본 pickle의 per-video 선택 결합으로 수 분 내 검정 가능. 게이트 임계값은 학습 분할(TRAINPOOL 덤프)에서 선택해야 함(평가 축에서 고르면 누수).

## 11. 체인 33 결과 — 학습 풀 보정 게이트 (2026-09-01 07:08)

calibrate(학습 풀 203, 누수 없음): 원본 MAE 0.894 vs 전부-refine 2.262 → **τ*=−10 dB(그리드 최소), refine 0/872 = "게이트 전부 끄기"가 최적**. CoTAF_TW_GATE_L ≡ TW(변화 0).
원인: 학습 풀에는 refine이 이득인 저SNR 영역이 사실상 없음(§10-1과 동일 근본 원인 — 학습 분포가 깨끗함). 학습 풀만으로는 임계를 보정할 수 없다는 정직한 null.

## 12. MMPD 학습 분할 보정 게이트 (GATEM, 2026-09-01)

MMPD 피험자 1~6(120클립)은 **지정 학습 분할**(CoTAM 계열이 사용)이므로 여기서 τ 보정 — 결과는 "MMPD train-split calibrated"로 별도 표기(라벨/분할 규칙 준수, 누수 없음). `selective_refine.py calibrate_m` → τ*=−4.75 dB, 분할 내 MAE 8.83→7.88(refine 38%).

| 시드 | MMPD held-out (TW→GATEM) | p | 비고 |
|---|---|---|---|
| 100 | 10.51 → 10.02 | .11 | refine 299/872, NIRP still 8.51→7.36 |
| 200 | 10.06 → 9.91 | .73 | |
| 300 | 10.37 → 9.97 | .14 | **NIRP still 7.86→12.07, small 7.82→10.36 후퇴** |
| 평균 | **10.31±0.19 → 9.97±0.06** | 풀링(3시드 평균/클립, 탐색적) **p=.057** | 개선 111 vs 악화 93 클립 |

고SNR 축은 완전 무손실(PURE·UBFC·RTNI/RTNC·DLCN·MCD stable 변화 0 — 게이트가 의도대로 통과). Walking 17.5~18.5, Skin 6 13.9~15.3.
**비용: MR-NIRP 시드 평균 후퇴**(still 7.82→8.84, small 8.68→9.42) — MMPD로 보정한 τ가 야간 NIR 도메인의 저SNR 클립에도 발화하는데, 거기서는 refined가 유해(§16-12 TS와 동일 방향: NIR 도메인은 다른 처방 필요).

**판정: 채택 불가(§5 기준 미달)** — 시드별 MMPD 비유의(최선 p=.11; 풀링 p=.057은 사전 등록 밖 탐색적 수치), NIRP 후퇴 시드 의존. 단 서술 가치는 있음: (i) 게이트는 고SNR 축 손실을 완전히 제거한 유일한 구성, (ii) "τ는 대상 도메인의 학습 분할에서만 보정 가능하고 다른 도메인에 이식하면 실패"는 도메인 조건부 배치 정책(보정 데이터 없는 도메인은 게이트 off)의 필요를 보여줌 — 논문 Discussion 소재.

## 13. 18단계 종합 판정 (2026-09-01)

diffusion refinement 계열 7구성(V1·REG·X0·DIT·W600·GATE·GATEM) 전부 사전 등록 기준(§5) 미달 → **공식 모델은 CoTAF-TW-L 유지**. 확립된 사실: ① 형태 사전은 저SNR에서 실제 작동(halving 12→4, Walking −3, 전 구성 재현) ② 무차별 적용은 고SNR을 해침(diffusion 고유 아님 — REG 재현) ③ SNR 게이트는 비용을 제거하지만 이득을 유의 수준까지 끌어올리지 못함(MMPD 잔여 오류의 상당 부분이 refine으로 고칠 수 없는 클립 — CV 상한 9.26과 정합) ④ 게이트 임계는 도메인 밖 이식 불가. 후속 후보(미착수): SSM 백본 비교 행(PhysMamba), refiner를 저SNR 쌍으로 재학습(MMPD 1~6 쌍 포함, CoTAM 계열 표기), consistency distillation.

## 14. 체인 34 — 저SNR 학습쌍 재학습(DLS) + 파라메트릭 harmonic template(PARM) (2026-09-01, 사용자 제안)

사용자 제안 2건의 구현. 판정 기준은 §5 동일, 산출 모델은 전부 "MMPD train-split calibrated (CoTAM 계열)" 표기.

| tag | 모델명 | 무엇을 검정하나 |
|---|---|---|
| DLS | CoTAF_TW_DLS_L | §10-1 근본 원인 직격: 학습 쌍에 MMPD 1~6(지정 학습분할) backbone출력-GT 쌍(노이즈 cond)을 추가해 V1 아키텍처 재학습 — "항등 매핑" 학습 분포를 실제 denoising 분포로 교정 |
| DLSG | CoTAF_TW_DLSG_L | DLS + 도메인 조건부 게이트: MMPD 1~6에서 τ 재보정, **MMPD 축에서만 발화**(그 외 축 원본) — §12의 NIRP 후퇴를 구조적으로 제거 |
| PARM | CoTAF_TW_PARM_L | "정답 심박을 함수형으로 파라미터 고정": 실 PPG 뱅크 131세그에서 harmonic amplitude ratio 중앙값(r2a=0.229, r3a=0.050) 추정 → 창별 f0만 자유 파라미터, harmonic-template matching(파워비 가중 r2p=0.052, r3p=0.002)으로 f0 선택, 파형은 고정 비율 harmonic 급수로 재구성(인간 맥파 형태 강제). 학습 불필요, 광대역(0.5–8Hz) 분석 |
| PARMG | CoTAF_TW_PARMG_L | PARM + 도메인 조건부 게이트(위와 동일) |

주의점(사전 기록): DLS의 τ 보정은 학습에 쓴 MMPD 1~6과 동일 클립이라 τ가 과발화 방향으로 편향될 수 있음(평가 누수는 아님 — held-out 7~33이 심판). PARM의 게이트 보정은 학습이 없어 이 편향 없음. 코드 `scripts/lowsnr_param.py`, 체인 `scripts/run_chain34_lowsnr.sh`, 로그 /tmp/chain34.log.

## 15. 체인 34 결과 (2026-09-01 16:04~) — 판정: 4구성 전부 채택 불가

**PARM/PARMG (파라메트릭 harmonic template, 학습 무)**
| 시드 | PARM (TW→) | p | PARMG (TW→) | p |
|---|---|---|---|---|
| 100 | 10.51→10.12 | .51 | 10.51→10.30 | .32 |
| 200 | 10.06→**10.46** | .17 | 10.06→**10.27** | .24 |
| 300 | 10.37→9.91 | .26 | 10.37→10.26 | .77 |

시드 간 방향 불일치(s200 악화) + 전부 비유의 → §5의 ①·③ 미달. 게이트 보정(τ*=−6.25dB, frac 0.10, 분할 내 8.83→7.99)은 유효했으나 held-out 이득이 없음. 해석: f0만 자유인 harmonic 재구성은 스펙트럼 argmax를 harmonic-sum argmax로 바꾸는 것과 사실상 동일 — 정보를 추가하지 못하고 창 경계 재구성 오차만 더함.

**DLS/DLSG (MMPD 1~6 노이즈 쌍 1192창 추가 재학습, 총 2874/217창, valMAE 3.37 vs cond baseline 4.05)**
| 시드 | DLS (TW→) | p | DLSG (TW→) | p |
|---|---|---|---|---|
| 100 | 10.51→10.62 | .37 | 10.51→10.69 | .45 |
| 200 | 10.06→10.37 | .11 | 10.06→10.26 | .33 |
| 300 | 10.37→10.83 | .11 | 10.37→10.78 | .32 |

**3시드 일관 악화.** 결정적 증거: 게이트 보정에서 학습에 쓴 MMPD 1~6 클립의 all-refine MAE가 8.83→**1.96**(τ*=−2.75dB, frac 0.68) — 학습 클립은 사실상 암기했으나 held-out 7~33에서는 악화 = **120클립 저SNR 쌍은 일반화가 아니라 암기로 귀결**(§14에 사전 기록한 과발화 편향이 실현됨, 다만 원인은 τ가 아니라 refiner 자체의 과적합). 무차별 DLS는 전 축 손상(C6 0→16.0, NIRP 13.8, RTNI 4.2 등 — V1보다 심함); 도메인 조건부 게이트(DLSG)는 비-MMPD 축 무손실을 확인(구조 자체는 의도대로 동작).

부수 관찰: DLS의 halving 12→5~6(전 구성 최저), Walking 19.63→17.33, Skin 6 17.26→16.03 — 형태 사전의 저SNR 이득 패턴은 재재현. 그러나 중간-SNR 클립(motion 1/3, skin 5) 손상이 이득을 상쇄.

### 18단계 최종 종합 (갱신)
diffusion/refinement 계열 **9구성**(V1·REG·X0·DIT·W600·GATE·GATEM·DLS·PARM(+게이트판)) 전부 §5 미달 → **공식 모델 CoTAF-TW-L 유지 확정**. 추가 확립 사실: ⑤ 저SNR 학습 쌍 부족은 데이터 추가(120클립)로 안 풀림 — 암기로 붕괴, 도메인 내 저SNR 다양성 자체가 병목. ⑥ 파라메트릭 형태 강제는 무정보(f0 추정 문제로 환원됨). 남은 미착수: PhysMamba 백본, consistency distillation(속도 목적), 대규모 저SNR 합성 쌍(노이즈 증강으로 쌍 수십 배 확장) — 단 ⑤가 재학습 계열의 기대치를 낮춤.

## 16. 체인 35 — 대규모 저SNR 합성 쌍(DSYN) + PhysMamba 백본 비교 (2026-09-01 착수, 사용자 지시)

**(B) DSYN — §15-⑤의 직접 검정**: "도메인 내 저SNR 다양성 부족 = 병목"이라면 합성으로 다양성을 채웠을 때 암기 붕괴가 사라져야 한다.
- 쌍: 체인 34 실 쌍 2874(TRAINPOOL 1899 + MMPD 1~6 쌍 1192, 실 cond 비중 ~7%) + 합성 ~42k — 깨끗한 창(실 GT 2874 + ppg_bank 654) × K=12 잡음 draw.
- 잡음 모델(실패 모드 모사, 단위분산 혼합): 광대역(0.5–8Hz) + distractor 발진(0.3–4Hz, AM, p=.5) + 2f0 doubling 교란(p=.4) + 0.5f0 subharmonic(p=.2) + 모션 버스트(p=.3). 혼합비는 MMPD 1~6 실측 spec-SNR 분포에서 표본(−3dB 보정; 합성 p10 −6.3 ≈ 실측 −6.2, 상위권은 2f 교란이 SNR 지표에 signal로 잡혀 더 넓음 — 의도된 hard case).
- V1 아키텍처 40k steps → CoTAF_TW_DSYN_L(무차별)·CoTAF_TW_DSYNG_L(MMPD 1~6 τ 재보정, 도메인 조건부). 판정 §5 동일. τ 보정 클립의 실 cond가 학습에 포함되나 비중 3% 미만(체인 34의 지배적 암기와 구분, 사전 기록).

**(A) PhysMamba 백본 비교** — SSM 백본에서 TW 코퍼스 효과의 이식성(9단계 백본 전이 행의 연장).
- 빌드: tools/mamba(bimamba 1.0.1)를 sm_120용으로 컴파일(setup.py에 compute_120 추가, conda env cuda128의 nvcc 12.8; 시스템 nvcc 12.0은 Blackwell 미지원). 추가 수리 2건: ① mamba 1.0.1이 요구하는 구식 transformers API → site-packages generation.py 임포트 가드, ② pip이 끌어온 causal_conv1d 1.7.0의 ABI 불일치 → v1.0.2 소스를 동일 아치 패치로 재빌드(스모크: 사전학습 로드·forward (2,128)·backward 통과). PhysMambaTrainer에 [Virtual-Face patch] MODEL.RESUME 미세조정 로드 추가(RhythmFormer 패치와 동일 패턴).
- 레시피(9단계 백본 전이와 동일 철학): UBFC-rPPG_PhysMamba_DiffNormalized.pth init, 10 epochs, LR 1e-4, batch 4, chunk 128(전용 pm_* 캐시, 두 모델 공유·종료 후 삭제), Neg_Pearson 파형 손실이라 cumsum 라벨 패치 불요.
- 비교: PM_CoTAF_L(기준 코퍼스 ALLF) vs PM_CoTAF_TW_L(TW 코퍼스 OFFICIAL), 1시드 우선 — 방향 확인 후 필요시 시드 확장. 체인 /tmp/chain35.log.

### §16-B 결과 (2026-09-01 16:44) — DSYN/DSYNG 채택 불가 (10번째 null)

학습: 45,210창(합성 42,336 + 실 2,874), 40k steps. **내부 검증(실 cond 217창)에서 이미 실패 신호**: valMAE 4.77 > cond baseline 4.05 — 합성 잡음으로 배운 복원이 실 잡음에 이득이 없음.

| 시드 | DSYN (TW→) | p | DSYNG (TW→) | p |
|---|---|---|---|---|
| 100 | 10.51→11.89 | **.0006 (유의 악화)** | 10.51→10.25 | .11 |
| 200 | 10.06→11.95 | **<.0001 (유의 악화)** | 10.06→10.13 | .73 |
| 300 | 10.37→11.80 | **.0005 (유의 악화)** | 10.37→10.76 | .22 |

게이트 재보정도 같은 진단: MMPD 1~6에서 all-refine 8.83→9.34(악화), τ*=−6.75dB로 **7%만 통과**(DSYNG refined 54/872) → DSYNG ≈ TW(비유의). 결론: **잡음 자체에 도메인 격차가 있다** — 실패 모드를 모사한 합성 잡음(대역·distractor·2f/0.5f·버스트)은 backbone 출력의 실 잡음 통계(시변 구조·신호-잡음 상관)를 재현하지 못하고, SNR 분포를 맞춰도 복원 규칙이 이식되지 않는다. 체인 34의 ⑤(실 쌍 부족→암기)와 결합하면: **저SNR 쌍은 "실 데이터는 너무 적고, 합성은 분포가 다르다" — waveform-refinement 축은 현 재료로 닫힘**(refinement 계열 10구성째 null, §5 기준 통과 0).

### §16-A 결과 (2026-09-01 18:57) — PhysMamba 백본: TW 효과 비이식, 공식 모델 변경 없음

학습 2회 정상(UBFC init, 10ep, chunk 128, 1시드): PM_CoTAF_L(기준 코퍼스) / PM_CoTAF_TW_L(TW 코퍼스).

| 비교 | MMPD held-out | p | 비고 |
|---|---|---|---|
| PM_CoTAF_L → PM_CoTAF_TW_L (백본 내 TW 효과) | 14.12 → 13.81 | .66 (비유의) | halving 41로 동일, Walking 26.5 동일 |
| CoTAF_TW_L → PM_CoTAF_TW_L (백본 간) | 10.51 → 13.81 | <.0001 (열세) | PhysFormer 우위 |

- **TW 효과는 PhysMamba에 이식되지 않음**: PhysFormer에서 −1.7~−1.9(유의)였던 개선이 PM에서는 −0.31(비유의). halving이 41로 그대로 — PM의 MMPD 오류는 파형 형태가 아니라 백본 자체의 harmonic 취약성이 지배(9단계 "백본 전이는 적용 가능하나 불변 아님" 재확인, SSM에서도 성립).
- **백본별 강약 분해**: PhysMamba가 강한 축 — MCD 운동후 0.62~0.88(PhysFormer 3.08 대비 최고), REALTEST/RTNI/RTNC 0.00, NIRP still 6.24(기준 코퍼스). PhysFormer가 강한 축 — MMPD 전반(10.51 vs 13.81), MCD stable(0.53 vs 2.46), DLCN(1.71 vs 2.59). TW가 PM에서 NIRP를 악화(6.24→9.25, 8.14→11.84)시키는 비용도 관찰.
- 판정: **공식 모델 CoTAF-TW-L(PhysFormer) 유지**. PhysMamba 행은 백본 의존성 표(T7류)에 서술 가치 — "waveform prior의 이득은 백본의 실패 모드가 파형 형태 결핍일 때만 발현"이라는 조건을 SSM 백본이 재확인.

## 17. 체인 35 종결 — 사용자 지시 2건(§16) 완료, 18단계 최종 확정 (2026-09-01)

refinement 10구성 + 백본 1행 모두 §5/판정 기준 미달 → **CoTAF-TW-L 최종 확정**. 남은 미착수는 consistency distillation(속도 목적, 정확도 이득 없음)뿐.
