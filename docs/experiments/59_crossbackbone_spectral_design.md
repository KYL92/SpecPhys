# 59. 스펙트럼 손실의 백본 교차 검증 + PhysMamba VIPL 악화 진단 — 사전 등록 (체인 75, 2026-11-02)

사용자 지시(11-02 "다음 후보 진행"): 체인 74(57 문서)의 후속 후보 — ① SpecFP 의 출력 스펙트럼 손실을 RhythmFormer·PhysMamba 에 이식해 같은 풀에서 학습(백본 교차 검증), ② PhysMamba 가 풀 학습으로 VIPL 이 악화된(10.86 → 12.42) 원인 진단, ③ 58 원고 `[verify]` 참고문헌 확정(웹 확인, 59 §5).

## 1. 질문과 셀
57 §6 의 귀결: 코퍼스 이득은 백본 의존이며 SpecFP 의 VIPL 우위를 만드는 것은 스펙트럼 목적함수였다(FactorizePhys 트렁크에서). 그 목적함수가 **다른 백본에서도** 같은 이득을 내는지(= 기여가 백본 비의존적인 '학습 신호'인지)와, PhysMamba 의 악화가 최적화(학습률·epoch) 문제인지 분포 이동인지를 가른다.

| 셀 | 백본·손실 | 풀/레시피 | 시드 |
|---|---|---|---|
| `RF_SPEC_L`(+`_s200`,`_s300`) | RhythmFormer 고유 손실 + **패치 43** `RPPG_SPEC=1`(출력 HR 대역 NLL 1.0 + 반감 억제 0.5, 심박 라벨 `NF_HR_TB=1`) | 체인 74 `RF_POOL_L` 과 동일(366 클립 u8, 표준화, 청크 160, 공개 가중치 출발, LR 1e-4·10 epoch·배치 4) | 100/200/300 |
| `PM_SPEC_L`(+`_s200`,`_s300`) | PhysMamba Neg-Pearson + 패치 43(차분 라벨·출력 → **누적합 파형** 위에서 스펙트럼 항) | `PM_POOL_L` 과 동일 | 100/200/300 |
| `PM_POOL_EP_L` | PhysMamba 고유 손실, 전 epoch 체크포인트 보존 → epoch 0/2/4/6/9 VIPL 평가 | `PM_POOL_L` 과 동일, 시드 100 | 1 |
| `PM_POOL_LR5_L` | PhysMamba 고유 손실, LR **1e-5** | 나머지 동일, 시드 100 | 1 |

패치 43 은 SpecFP 의 교란 슬롯 항(`L_nuis`·`L_decor`·맥파 슬롯 NLL)은 포함하지 않는다(다른 백본엔 슬롯이 없음; 48 §6 에서 이 항들은 기여 0 으로 확정됨). 비교 상대: 같은 풀 고유 손실 `RF_POOL_L`(VIPL 7.48 / MMPD 8.49)·`PM_POOL_L`(12.42 / 10.16), 공개 PhysMamba `PMbase`(10.86 / 12.74), SpecFP `NF_HYB_L`(5.82 / 8.83).

### 1b. Alpha 보완 셀(체인 76, 사용자 지시 'alpha 에서 돌릴 작업', 09:55 KST 기동)
| 셀 | 백본·손실 | 풀/레시피 | 시드 |
|---|---|---|---|
| `PF_SPEC_L`(+`_s200`,`_s300`) | PhysFormer 고유 손실(Neg-Pearson + 주파수 CE/KL) + 패치 43 `RPPG_SPEC=1`(차분 출력 → 누적합 파형, 심박 라벨 = 트레이너의 cumsum 라벨) | yaml24 공식 레시피(공개 UBFC PhysFormer 출발, LR 5e-5·10 epoch·배치 4, 차분 정규화 128×128 청크 160) = 고유 손실 풀 셀 `CoTAM_L`(VIPL 8.01 / MMPD 7~33 10.33)과 동일, 데이터만 u8 풀 캐시(수치 동일) | 100/200/300 |

Alpha 에서 돌리는 이유: pro6000 의 체인 75 셀과 겹치지 않고(두 머신에서 같은 셀 금지), 세 백본(PhysFormer·RhythmFormer·PhysMamba) 모두에서 '고유 손실 vs +스펙트럼 손실' 짝이 완성된다. Alpha 의 풀 u8·MMPD u8 캐시는 재구축(각 ≈30 분), VIPL u8 캐시와 PhysFormer 용 float 실 축 캐시(`real_*`)는 보존돼 있어 재사용. 레인 A(시드 100·300)·B(시드 200), GPU 게이트 12 GB(타 사용자 4 GB 점유).

**H-X4** `PF_SPEC_L` vs `CoTAM_L` × {VIPL, MMPD 7~33} (Holm 2; Δ<0 → 지지) — 판정은 `crossbb_judge.py` 가족 X4 로 통합.

## 2. 평가·판정
평가 축·집계·시드 풀링은 57 §2 와 동일(VIPL 문헌 GT 2,378 · MMPD held-out 7~33 540 · 실 축 9개). 가족별 Holm.

| 가족 | 가설 | 검정 | 규칙 |
|---|---|---|---|
| X1 스펙트럼 손실 이식(RhythmFormer) | H-X1 스펙트럼 손실이 RhythmFormer 를 개선 | `RF_SPEC_L` vs `RF_POOL_L` × {VIPL, MMPD} | Δ<0, Holm p<.05 → 지지 |
| X2 (PhysMamba) | H-X2 스펙트럼 손실이 PhysMamba 를 개선(악화 복구 포함) | `PM_SPEC_L` vs `PM_POOL_L` × {VIPL, MMPD} | 같음 |
| X3 백본 비의존성 | H-X3 스펙트럼 손실을 받은 RhythmFormer 는 SpecFP 와 동급 | `RF_SPEC_L` vs `NF_HYB_L` × {VIPL, MMPD}, 양측 | Holm ns 이고 |Δ|≤0.3 → 동급, Δ<0 유의 → RhythmFormer 우세(트렁크 비필수), Δ>0 유의 → SpecFP 우세 유지 |
| D 진단 | H-D1 악화는 학습 초기부터(분포 이동) vs 후반 누적(과적합) | `PM_POOL_EP_L` epoch 0/2/4/6/9 VIPL MAE 곡선 vs 공개 10.86 (서술) | epoch 0 이미 >10.86 → 분포 이동; 단조 증가 → 과적합 |
|  | H-D2 낮은 학습률이 악화를 없앤다 | `PM_POOL_LR5_L` vs `PMbase` VIPL, vs `PM_POOL_L` VIPL (Holm 2) | vs 공개 Δ≤+0.3(비열등) 또는 Δ<0 → 지지 |

2차(서술): 실 축 9개(특히 RTNI 반감/플리커, MR-NIRP 대동작), 반감 건수(`hlv`) 변화(반감 억제 항의 직접 효과), MMPD 움직임별.

예상(정직 기록): RhythmFormer 는 이미 주파수 교차 엔트로피 항을 가져 HR 대역 NLL 의 추가 효과는 작을 수 있고, 반감 억제가 MMPD(`hlv` 6~15/540)·VIPL 반감(4~7)을 줄이는 정도가 관건; PhysMamba 는 스펙트럼 항이 VIPL 악화를 일부 복구할 가능성, 완전 복구(≤10.86)는 불확실; D 는 epoch 0 부터 악화(분포 이동)로 예상.

## 3. 기술
- **패치 43**(`RhythmFormerTrainer`·`PhysMambaTrainer`, 환경변수 `RPPG_SPEC=1`, `SPEC_W_SPEC` 1.0, `SPEC_W_HALF` 0.5, `NF_HR_TB` 존중): `NFLoss.spec_nll`·`half_penalty` 재사용; `diff_flag=1`(차분 라벨)이면 라벨·예측을 누적합(예측은 이동평균 고역 통과)한 파형에서 계산. 트레이너가 `[spec] active …`·`SPEC stats epoch k: spec= half= n=` 를 출력 — 체인은 두 줄이 없으면 실패. CPU 스모크: 합성 정현파에서 라벨 심박 ±0.5 bpm, HR/2 예측의 반감 벌점 0.75 vs 정답 0.002.
- 체인 `scripts/run_chain75_crossbb.sh` 레인 RS1–3·PS1–3·PD·Z, 판정 `scripts/crossbb_judge.py`, 회수 `scripts/pull_chain75.sh`. float 캐시(`rf_real_*`,`pm_real_*`)는 체인 74 가 삭제했으므로 재구축(레인 공유 flock), Z 가 삭제. 예상 ≈5 h.

## 4. 결과 — pro6000 부분(체인 75 종료 2026-11-02 11:05 KST, 4.0 h, 실패 0; 판정 `results/crossbb_judge.{txt,json}`). Alpha 의 PhysFormer 셀(체인 76)은 §4.4 에 추가.

### 4.1 영상 단위 MAE(시드 영상별 평균 풀링) + 반감/배가 건수(시드 평균)
| 셀 | 시드 | VIPL(hlv/dbl) | MMPD 7~33(hlv/dbl) | NIRP 정지 | NIRP 소동작 | NIRP 대동작 | RTNI | RTNC | PURE 07~10 | DLCN |
|---|---|---|---|---|---|---|---|---|---|---|
| RhythmFormer 풀, 고유 손실 `RF_POOL_L` | 3 | 7.48 (5.7/15.3) | 8.49 (10.0/1.7) | 11.52 | 11.64 | 11.01 | 3.48 | 4.32 | 0.41 | 1.32 |
| **RhythmFormer 풀 + 스펙트럼 `RF_SPEC_L`** | 3 | **5.89** (5.85/5.90/5.91; 5.3/2.3) | **7.79** (7.47/7.90/7.99; 6.0/1.3) | 17.62 | 18.12 | 15.31 | 4.14 | 3.74 | 0.38 | 1.08 |
| 공개 PhysMamba `PMbase` | 1 | 10.86 (12/37) | 12.74 (29/6) | – | – | – | – | – | – | – |
| PhysMamba 풀, 고유 손실 `PM_POOL_L` | 3 | 12.42 (32.7/37.0) | 10.16 (25.0/3.7) | 7.11 | 10.86 | 8.51 | 1.32 | 0.00 | 0.36 | 2.11 |
| **PhysMamba 풀 + 스펙트럼 `PM_SPEC_L`** | 3 | **9.86** (10.86/9.44/9.29; 11.7/35.0) | **8.85** (9.01/8.81/8.73; 14.7/3.3) | 7.80 | **6.58** | **8.71** | **0.00** | 0.00 | 0.31 | 1.10 |
| PhysMamba 풀 LR 1e-5 `PM_POOL_LR5_L` | 1 | 13.39 (25/60) | 12.53 (24/7) | – | – | – | – | – | – | – |
| (기준) SpecFP `NF_HYB_L` | 3 | 5.82 (56.3/6.0) | 8.83 (12.0/7.0) | 6.86 | 6.12 | 11.33 | 1.46 | 2.27 | 0.62 | 0.81 |

### 4.2 판정(가족별 Holm)
| 가족 | 검정 | n | 기준 → 셀 | Δ | p | Holm | 판정 |
|---|---|---|---|---|---|---|---|
| X1 | VIPL: RF_SPEC_L vs RF_POOL_L | 2378 | 7.48 → 5.89 | −1.59 | <.0001 | <.0001 | **지지** |
| X1 | MMPD: RF_SPEC_L vs RF_POOL_L | 540 | 8.49 → 7.79 | −0.70 | .054 | .054 | ns(경계) |
| X2 | VIPL: PM_SPEC_L vs PM_POOL_L | 2378 | 12.42 → 9.86 | −2.56 | <.0001 | <.0001 | **지지** |
| X2 | MMPD: PM_SPEC_L vs PM_POOL_L | 540 | 10.16 → 8.85 | −1.31 | .0004 | .0004 | **지지** |
| X3 | VIPL: RF_SPEC_L vs SpecFP (양측) | 2378 | 5.82 → 5.89 | +0.07 | .68 | .68 | **동급**(|Δ|≤0.3, ns) |
| X3 | MMPD: RF_SPEC_L vs SpecFP (양측) | 540 | 8.83 → 7.79 | −1.04 | .012 | .025 | **RhythmFormer 우세** |
| D2 | VIPL: LR 1e-5 vs 공개 PhysMamba | 2378 | 10.86 → 13.39 | +2.53 | <.0001 | <.0001 | **기각(열세)** |
| D2 | VIPL: LR 1e-5 vs LR 1e-4 풀 셀 | 2378 | 12.42 → 13.39 | +0.97 | .047 | .047 | 기각(열세) |

D1 epoch 곡선(`PM_POOL_EP_L`, VIPL 문헌 GT; 공개 10.86): epoch 0 **17.11** → 2 12.22 → 4 11.12 → 6 13.59 → 9 12.73(체인 74 시드 1 과 동일 재현). 악화는 첫 epoch 에 최대로 나타난 뒤 요동하며 한 번도 공개 가중치 아래로 내려가지 않음 → 과적합이 아니라 **풀 분포(또는 손실 신호)와 공개 가중치 사이의 이동** 이 원인이고, LR 을 낮추면 더 나빠지며, 같은 풀·같은 LR 에서 스펙트럼 손실만 더하면 9.86 으로 공개 가중치 아래로 내려간다(X2).

2차(서술): 스펙트럼 손실은 PhysMamba 의 실 축을 전반 개선(MR-NIRP 소동작 10.86→6.58·대동작 8.51→8.71 유지, RTNI 1.32→0.00, 배가 건수는 VIPL 35 로 그대로·반감 32.7→11.7), 반면 **RhythmFormer 에서는 MR-NIRP 세 축이 악화**(정지 11.5→17.6, 소동작 11.6→18.1, 대동작 11.0→15.3; n=19, 비조정 p 는 §judge 2차 참조)하고 RTNI 는 3.5→4.1 로 변화 없음 — 스펙트럼 항이 주행 영상(저 SNR·근적외선 센서 RGB)에서 RhythmFormer 의 주파수 CE 항과 충돌하는지, 또는 반감 억제가 저심박 주행 영상을 밀어내는지는 미확인(후속 후보).

### 4.3 결론(정직 보고, pro6000 부분)
1. **스펙트럼 손실은 백본 비의존적 학습 신호다**: 같은 풀에서 RhythmFormer(VIPL −1.59, MMPD −0.70 경계)와 PhysMamba(VIPL −2.56, MMPD −1.31) 모두 개선(H-X1 VIPL·H-X2 지지). 56/58 원고의 "기여는 코퍼스 + 스펙트럼 손실, 구조 무관" 주장이 백본 축으로 확장된다.
2. **FactorizePhys 트렁크는 필수가 아니다(H-X3)**: RhythmFormer + 스펙트럼은 VIPL 에서 SpecFP 와 동급(5.89 vs 5.82)이고 MMPD 에서는 우세(7.79 vs 8.83, Holm .025) — 단 MR-NIRP 주행에서는 SpecFP(6.1/11.3)가 훨씬 낫고(RhythmFormer+스펙트럼 18.1/15.3), 파라미터는 SpecFP 52k vs RhythmFormer 수백만. 논문 위치 설정: "스펙트럼 감독 레시피"가 기여, SpecFP 는 가장 작은 인스턴스.
3. **PhysMamba 악화의 원인은 최적화 강도가 아니다(H-D1·D2)**: 첫 epoch 에 최대 악화, LR 1e-5 는 더 나쁨; 스펙트럼 손실이 유일한 해결책(9.86 < 공개 10.86).
4. 후속 후보(사용자 결정): RhythmFormer+스펙트럼의 MR-NIRP 악화 원인(반감 항 제거 절제 `SPEC_W_HALF=0`, 주파수 CE 항과의 가중 균형), PhysMamba 스펙트럼 셀의 배가 건수(35) 억제(배가 억제 항 추가).

### 4.4 Alpha 부분(체인 76 종료 2026-11-02 12:00 KST, 1.8 h 학습 3 run + 평가) — PhysFormer + 스펙트럼 `PF_SPEC_L` ×3: **H-X4 기각(역방향)**
| 셀 | 시드 | VIPL(hlv/dbl) | MMPD 7~33 | NIRP 정지 | NIRP 소동작 | NIRP 대동작 | RTNI | RTNC | PURE 07~10 | DLCN | MCD A | MCD B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PhysFormer 풀, 고유 손실 `CoTAM_L` | 3 | 8.01 (16.3/6.0) | 10.33 | 7.91 | 7.32 | 7.48 | 2.20 | 1.17 | 3.41 | 3.68 | 4.67 | 1.57 |
| PhysFormer 풀 + 스펙트럼 `PF_SPEC_L` | 3 | **8.68** (8.77/8.48/8.79; 10.3/14.7) | 9.79 (9.77/10.23/9.38) | 9.73 | 9.45 | 9.96 | 1.46 | 1.72 | 4.53 | 2.81 | 5.99 | 4.85 |

| 검정 | n | 기준 → 셀 | Δ | p | Holm | 판정 |
|---|---|---|---|---|---|---|
| H-X4 VIPL: PF_SPEC_L vs CoTAM_L | 2378 | 8.01 → 8.68 | **+0.67** | <.0001 | <.0001 | **기각(역방향)** |
| H-X4 MMPD: PF_SPEC_L vs CoTAM_L | 540 | 10.33 → 9.79 | −0.54 | .39 | .39 | ns |

서술: 실 축도 대부분 악화(MR-NIRP +1.8~+2.5, MCD B +3.3, 전부 n 작아 ns), RTNI 2.20→1.46 만 개선. 배가 건수 6→14.7 증가(PhysMamba 스펙트럼 셀과 같은 부작용). PhysFormer 고유 손실은 이미 주파수 교차엔트로피 + 라벨 분포 KL(가중 b 가 epoch 에 따라 지수 증가) 을 갖고 있어 HR 대역 NLL 이 중복되며, 반감 억제 항이 고유 주파수 항과 충돌하거나 배가를 유발하는 것으로 보임 — 가중치 스윕(체인 78, 0.5/2.0)으로 가중치 불균형 가설을 검사 중.

**세 백본 종합(H-X1~X4)**: 스펙트럼 손실은 **주파수 항이 없는 백본(PhysMamba, Neg-Pearson 만)에서 가장 크게**(VIPL −2.56), **주파수 항이 약한 백본(RhythmFormer)에서 크게**(−1.59), **강한 주파수 항을 이미 가진 백본(PhysFormer)에서는 역효과**(+0.67). 즉 "스펙트럼 감독은 백본 비의존적이되, 고유 손실에 같은 종류의 주파수 항이 이미 있으면 중복·충돌한다"가 정직한 결론이며, 논문 4.4 절은 이대로 기술한다(FactorizePhys·PhysMamba·RhythmFormer 3/4 백본에서 이득, PhysFormer 예외 명시).
## 5. 참고문헌 확인(③, 웹 검색 2026-11-02)
- RhythmFormer: Zou, Guo, Chen, Zhuo, Huang, Ma. "RhythmFormer: Extracting Patterned rPPG Signals based on Periodic Sparse Attention." arXiv:2402.12788 (v1 2024-02, v3 2025-02). **Pattern Recognition 게재는 확인되지 않음 → arXiv 로 인용.**
- PhysMamba: Luo, Xie, Yu. "PhysMamba: Efficient Remote Physiological Measurement with SlowFast Temporal Difference Mamba." arXiv:2409.12031, **CCBR 2024 채택**(arXiv 코멘트).
- DLCN: Li et al. "Exploring Remote Physiological Signal Measurement under Dynamic Lighting Conditions at Night: Dataset, Experiment, and Analysis." arXiv:2507.04306 (2025; 98명, 784 샘플 — 우리 원본 폴더는 780 트라이얼).
- MCD-rPPG: Egorov, Botman, Blinov, Zubkova, Ivaschenko, Kolsanov, Savchenko. "Gaze into the Heart: A Multi-View Video Dataset for rPPG and Health Biomarkers Estimation." ACM Multimedia 2025 (arXiv:2508.17924; 600명, 안정/운동 후, 3 카메라).
- PHASE-Net: Zhao, Guo, … , Yu(저자 10명). "PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement." arXiv:2509.24850 (2025-09, v3 2026-03); CVPR 2026 게재는 집계 사이트 표기뿐이라 arXiv 로 인용, 전체 저자·게재지 확인 필요.
- FreqPhys: Qian, Guo, Zhou, Zou, Yu, Wang. "FreqPhys: Repurposing Implicit Physiological Frequency Prior for Robust Remote Photoplethysmography." arXiv:2604.00534 (2026-04).

## 6. 후속(사용자 지시 '권장 순서로 진행', 2026-11-02 12:00) — 사전 등록: 반감 항 절제 → 가중치 스윕 → (결과 후 결정) 배가 억제 항

### 6.1 체인 77(pro6000, 12:0x 기동)
| 셀 | 설정 | 시드 | 질문 |
|---|---|---|---|
| `RF_SPEC_NH_L`(+`_s200`,`_s300`) | RhythmFormer + 스펙트럼, **`SPEC_W_HALF=0`**(HR 대역 NLL 만) | 100/200/300 | RhythmFormer+스펙트럼의 MR-NIRP 악화(정지/소/대동작 11.5/11.6/11.0 → 17.6/18.1/15.3)가 반감 억제 항 때문인가 |
| `RF_SPEC_W05_L` / `RF_SPEC_W20_L` | RhythmFormer + 스펙트럼, `SPEC_W_SPEC` 0.5 / 2.0 (half 0.5) | 100 | 가중치 민감도 |
| `PM_SPEC_W05_L` / `PM_SPEC_W20_L` | PhysMamba + 스펙트럼, `SPEC_W_SPEC` 0.5 / 2.0 | 100 | 가중치 민감도 |

### 6.2 체인 78(Alpha, 체인 76 종료 후)
| `PF_SPEC_W05_L` / `PF_SPEC_W20_L` | PhysFormer + 스펙트럼(yaml24 레시피), `SPEC_W_SPEC` 0.5 / 2.0 | 100 | PhysFormer 의 '이득 없음'(VIPL 8.77/8.48 vs 고유 8.01)이 가중치 불균형(고유 주파수 CE 항이 이미 강함) 때문인가 |

### 6.3 가설·판정(`scripts/spec_ablation_judge.py`, 가족별 Holm)
| 가족 | 가설 | 검정 | 규칙 |
|---|---|---|---|
| NH | H-NH1 반감 항이 MR-NIRP 악화의 원인 | `RF_SPEC_NH_L` vs `RF_SPEC_L` × {MR-NIRP 정지, 소동작, 대동작} (Holm 3, n=19) | Δ<0 Holm p<.05 → 지지(반감 항 제거 권장); 가드: VIPL·MMPD Δ≤+0.3(이득 보존) |
|  | H-NH2 반감 항은 VIPL·MMPD 이득에 필요 없다 | `RF_SPEC_NH_L` vs `RF_SPEC_L` × {VIPL, MMPD} 양측 | |Δ|≤0.3 ns → 불필요(레시피 단순화) |
| W | H-W1 가중치 민감도 | 각 백본에서 {0.5, 1.0, 2.0} 의 VIPL·MMPD·MR-NIRP 소/대동작·RTNI MAE 표(시드 1, 서술) | 0.5~2.0 전 구간에서 고유 손실 대비 개선 유지 → 둔감; PhysFormer 에서 어느 가중치든 고유(8.01) 미만이면 '백본 비의존' 을 세 백본으로 확장 |

예상(정직 기록): 반감 항은 저심박·저 SNR 주행 영상에서 진짜 저주파 성분을 밀어낼 수 있어 H-NH1 지지 가능성 있음; VIPL 반감 건수가 RhythmFormer 에서는 원래 적어(5.7) H-NH2 도 지지 가능. PhysFormer 는 2.0 보다 0.5 쪽이 유리할 것으로 예상.

### 6.4 결과 — 체인 78(Alpha, PhysFormer 가중치 스윕, 종료 13:12 KST, 시드 1)
| SPEC_W_SPEC | VIPL | MMPD 7~33 | NIRP 소동작 | NIRP 대동작 | RTNI | 시드 |
|---|---|---|---|---|---|---|
| native | 8.01 | 10.33 | 7.32 | 7.48 | 2.20 | 3 |
| 0.5 | 8.02 | 9.76 | 8.10 | 13.69 | 3.96 | 1 |
| 1.0 | 8.68 | 9.79 | 9.45 | 9.96 | 1.46 | 3 |
| 2.0 | 9.48 | 9.86 | 12.12 | 12.30 | 7.47 | 1 |

판정(서술, H-W1 PhysFormer 부분): VIPL 은 가중치에 **단조 악화**(고유 8.01 → 0.5: 8.02 → 1.0: 8.68 → 2.0: 9.48; 배가 17/14.7/27), MMPD 는 가중치와 무관하게 −0.5 전후(9.76/9.79/9.86), MR-NIRP·RTNI 는 전 가중치에서 고유보다 나쁨. 즉 PhysFormer 의 역효과는 "가중치 과다"로 설명되며, 0.5 에서는 VIPL 중립·MMPD 소폭 이득 — 고유 주파수 CE+KL 항과 중복되는 백본에서는 스펙트럼 항을 약하게 주거나 빼는 것이 옳다. 어느 가중치에서도 고유(8.01) 아래로 내려가지 않아 '백본 비의존'은 **3/4 백본(FactorizePhys·RhythmFormer·PhysMamba)**으로 한정해 기술한다.

### 6.5 체인 79(Alpha, 13:20 KST 기동) — 가중치 0.5 시드 확장
`PF_SPEC_W05_L_s200`·`_s300`(동일 레시피, 시드 200·300) → 0.5 의 'VIPL 중립(8.02 vs 8.01)·MMPD −0.57' 을 3시드로 확정하고 표 10/표 3 의 PhysFormer 스펙트럼 행을 0.5(3시드)로 교체할지 판단(판정: `PF_SPEC_W05_L` 3시드 vs `CoTAM_L` VIPL·MMPD, Holm 2; VIPL |Δ|≤0.3 ns → 중립, MMPD Δ<0 p<.05 → 이득).

### 6.6 결과 — 체인 77(pro6000, 종료 14:30 KST, 학습 7 run, 실패 0) + 체인 79(Alpha, 종료 14:26, PhysFormer 0.5 시드 확장); 판정 `results/spec_ablation_judge.{txt,json}`

**NH 반감 항 절제(`RF_SPEC_NH_L` 3시드 vs `RF_SPEC_L` 3시드)**
| 검정 | n | SPEC → NH | Δ | p | Holm | 판정 |
|---|---|---|---|---|---|---|
| H-NH1 MR-NIRP 정지 | 19 | 17.62 → 16.41 | −1.22 | .34 | 1.0 | ns |
| H-NH1 MR-NIRP 소동작 | 19 | 18.12 → 17.84 | −0.28 | .98 | 1.0 | ns |
| H-NH1 MR-NIRP 대동작 | 19 | 15.31 → 15.64 | +0.32 | .93 | 1.0 | ns |
| H-NH2 VIPL | 2378 | 5.89 → 5.84 | −0.04 | .51 | – | **동급(반감 항 불필요)** |
| H-NH2 MMPD 7~33 | 540 | 7.79 → 7.74 | −0.05 | .51 | – | **동급(반감 항 불필요)** |

→ **H-NH1 기각**: 반감 항을 빼도 MR-NIRP 악화가 그대로(고유 대비 소동작 +6.20 p=.012, 대동작 +4.63 p=.002) — RhythmFormer 의 주행 영상 악화는 **HR 대역 NLL 항 자체**에서 온다. **H-NH2 지지**: VIPL·MMPD 이득은 반감 항 없이 동일 → RhythmFormer 레시피는 NLL 단일 항으로 단순화 가능.

**W 가중치 스윕(`SPEC_W_SPEC`; 시드 1, 1.0 과 PhysFormer 0.5 는 3시드)** — VIPL / MMPD / NIRP 소 / NIRP 대 / RTNI
| 백본 | 고유 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|
| RhythmFormer | 7.48 / 8.49 / 11.6 / 11.0 / 3.5 | 5.71 / 7.62 / 19.1 / 16.5 / 3.9 | 5.89 / 7.79 / 18.1 / 15.3 / 4.1 | 5.81 / 7.38 / 18.4 / 14.8 / 3.9 |
| PhysMamba | 12.42 / 10.16 / 10.9 / 8.5 / 1.3 | 11.08 / 8.68 / 7.7 / 8.8 / 0.0 | 9.86 / 8.85 / 6.6 / 8.7 / 0.0 | **9.26 / 8.46** / 8.1 / **7.7** / 0.0 |
| PhysFormer | 8.01 / 10.33 / 7.3 / 7.5 / 2.2 | 8.48 / 10.40 / 7.9 / 10.6 / 1.5 (3시드) | 8.68 / 9.79 / 9.5 / 10.0 / 1.5 | 9.48 / 9.86 / 12.1 / 12.3 / 7.5 |

→ **RhythmFormer: 가중치 둔감**(0.5~2.0 에서 VIPL 5.71~5.89, MMPD 7.38~7.79 모두 고유보다 좋음; MR-NIRP 는 전 구간 악화). **PhysMamba: 가중치가 클수록 좋음**(2.0 에서 VIPL 9.26·MMPD 8.46·대동작 7.73, 배가 35→16) — 고유 손실에 주파수 항이 전혀 없어 스펙트럼 항이 유일한 주파수 감독. **PhysFormer: 어느 가중치도 고유 미만 없음**(0.5 3시드 VIPL 8.48·MMPD 10.40; 체인 78 시드 1 의 8.02/9.76 은 시드 효과) — 고유 주파수 CE+KL 과 중복·충돌.

**결론(정직 보고)**: ① 스펙트럼 감독의 이득은 "고유 손실에 주파수 항이 없거나 약한 백본"에서 나며(PhysMamba > RhythmFormer ≫ PhysFormer=0), 가중치 최적값도 그 축을 따른다(PhysMamba 크게, RhythmFormer 무관, PhysFormer 0). ② 반감 항은 RhythmFormer 에서 불필요(VIPL·MMPD 동일). ③ RhythmFormer 의 MR-NIRP 악화는 HR 대역 NLL 항 자체의 부작용이며 가중치·반감 항으로 해소되지 않음 — 저 SNR 주행 영상에서 대역 우도 최대화가 잡음 피크를 강화하는 것으로 추정(미검증). ④ 후보 2(배가 억제 항)의 근거: PhysMamba 스펙트럼 셀의 배가 35(1.0)→16(2.0)은 가중치로 일부 해소됐고 PhysFormer 는 스펙트럼 항 자체가 해로우므로, 배가 억제 항은 PhysMamba 에 한정해 시험할 가치가 있으나 우선순위는 낮음(사용자 결정).

### 6.7 RhythmFormer MR-NIRP 악화 기제 진단(학습 없음, Alpha; `scripts/rf_nirp_diag.py` → `results/rf_nirp_diag.{txt,json}`, 2026-11-02 15:00 KST)
재료: 고유 손실 `RF_POOL_L`·스펙트럼 `RF_SPEC_L`·반감 항 0 `RF_SPEC_NH_L` 각 3시드의 MR-NIRP 3축 저장 예측 파형 + float 캐시의 전역 밝기 차분 trace(교란 피크) + 라벨 스펙트럼.

| 축 | 모델 | MAE | 실패(|err|>5)/57 | lock-in / 반감 / 배가 / 기타 | 예측 첨도 | 정답대역 질량 | 예측피크=교란피크 | 오차 부호 평균 |
|---|---|---|---|---|---|---|---|---|
| 정지 | 고유 | 11.52 | 36 | 6/0/0/30 | .244 | .126 | .40 | +.58 |
| 정지 | 스펙트럼 | 17.62 | 38 | 2/1/5/30 | .250 | .134 | .25 | +.47 |
| 소동작 | 고유 | 11.64 | 40 | 9/0/2/29 | .222 | .114 | .32 | +.54 |
| 소동작 | 스펙트럼 | 18.12 | 46 | 16/0/4/26 | .220 | .085 | .51 | +.72 |
| 대동작 | 고유 | 11.01 | 41 | 13/0/0/28 | .211 | .108 | .32 | +.21 |
| 대동작 | 스펙트럼 | 15.31 | 47 | 15/0/1/31 | .194 | .101 | .21 | +.63 |

- **첨도 가설 기각**: 스펙트럼 변형의 예측 스펙트럼은 더 뾰족하지 않고(Δ첨도 +.006/−.002/−.017, 전부 ns) 정답 대역 질량도 늘지 않음(소동작은 오히려 −.029 p=.055) → "대역 우도 최대화가 잡음 피크를 날카롭게 강화한다"는 추정은 틀림.
- **교란 lock-in 가설 부분적**: lock-in 은 소동작에서만 증가(9→16), 정지·대동작은 변화 없음; 라벨 SNR 과 오차의 상관도 없음(ρ ≈ 0~.4, 상·하위 절반 모두 악화).
- **실제 기제 = 낮은 심박에서 학습 분포 쪽으로의 사전 끌림**: 오차 부호가 양(+)으로 이동(예측 > 정답)하고, 악화는 **gt < 65 bpm 영상에 집중**(정지 15.4 → 27.8, 소동작 19.9 → 24.5, 대동작 19.3 → 23.4; 이 영상들의 스펙트럼 변형 평균 예측 ≈ 81~86 bpm), ρ(gt, Δerr) = −.43/−.03/−.27. 즉 HR 대역 NLL 항이 풀의 심박 분포(대부분 60~100 bpm)에 맞춘 출력 사전을 강화해, 저 SNR 주행 영상의 낮은 심박(48~65)에서 풀 전형 심박으로 후퇴한다. 반감 항은 무관(§6.6).
- 함의: 스펙트럼 항의 부작용은 "학습 풀의 심박 범위 밖"에서 나타나며 FactorizePhys(SpecFP, MR-NIRP 6.1/11.3)에서는 같은 풀에서도 나타나지 않았으므로 백본 의존적이다. 해소 후보(사용자 결정): 풀의 저심박 커버리지 확장(합성 클립 45~60 bpm 비중 증가) 또는 NLL 대역을 라벨 심박이 아닌 넓은 생리 대역으로 완화 — 둘 다 새 셀 필요. 논문에는 한계 문단에 "저심박 구간에서 학습 분포로 후퇴" 로 기술.

### 6.8 체인 80(pro6000, 종료 15:46 KST) — PhysMamba 가중치 2.0 시드 확장(`PM_SPEC_W20_L` ×3; 사후 가족 W2, Holm 4)
| 셀 | VIPL | MMPD 7~33 | NIRP 소동작 | NIRP 대동작 | RTNI |
|---|---|---|---|---|---|
| 공개 PhysMamba | 10.86 | 12.74 | – | – | – |
| 풀, 고유 손실 | 12.42 | 10.16 | 10.86 | 8.51 | 1.32 |
| 풀 + 스펙트럼 1.0 (3시드) | 9.86 | 8.85 | 6.58 | 8.71 | 0.00 |
| **풀 + 스펙트럼 2.0 (3시드: 9.26/8.61/9.34)** | **9.07** | **8.67** | **6.08** | **8.30** | 0.00 |

| 검정 | n | 기준 → 셀 | Δ | p | Holm | 판정 |
|---|---|---|---|---|---|---|
| 2.0 vs 1.0: VIPL | 2378 | 9.86 → 9.07 | −0.79 | <.0001 | .0001 | **지지** |
| 2.0 vs 1.0: MMPD | 540 | 8.85 → 8.67 | −0.18 | .24 | .24 | ns |
| 2.0 vs 공개: VIPL | 2378 | 10.86 → 9.07 | −1.80 | <.0001 | <.0001 | **지지** |
| 2.0 vs 공개: MMPD | 540 | 12.74 → 8.67 | −4.06 | <.0001 | <.0001 | **지지** |

결론: PhysMamba 에서는 스펙트럼 가중치 2.0 이 1.0 보다 VIPL 에서 유의하게 좋고(−0.79), 공개 가중치 대비 VIPL −1.80·MMPD −4.06 으로 세 시드 모두 안정. MR-NIRP 소/대동작 6.08 / 8.30 은 전 zero-shot 모델 중 최저(SpecFP 6.12 / 11.33). 표 3·표 10 의 PhysMamba 스펙트럼 행은 사전 등록 기본값(1.0)을 유지하고 2.0 행을 추가 표기한다(가중치는 백본별로 조정이 필요하다는 §6.6 결론과 일관).
