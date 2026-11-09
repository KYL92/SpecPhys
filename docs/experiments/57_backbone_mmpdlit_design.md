# 57. 백본 일치 비교(RhythmFormer·PhysMamba 동일 풀 학습) + MMPD 문헌 분할(7:1:2) 실험 — 사전 등록 (체인 74, 2026-11-02)

사용자 지시(11-02): "② pro6000 에서 RhythmFormer·PhysMamba 동일 풀 학습과 MMPD 7:1:2 분할 실험(약 하루)". 56 문서(SpecFP 결과 원고)의 두 공백을 메운다 —
(a) 코퍼스 이득이 FactorizePhys 백본에 한정되는지(같은 풀로 다른 백본을 학습한 행이 RhythmFormer 에는 없고, PhysMamba 는 청크 128·MMPD 만 있음),
(b) FactorizePhys·RhythmFormer 논문이 쓰는 MMPD 인트라(문헌 7:1:2) 분할에서 SpecFP 의 in-domain 수치.
학습은 모두 pro6000 컨테이너(`vface_train`, `-u user`)에서, 집계·판정·문서는 Alpha 에서 한다. 정본 readout(툴박스 프로토콜·영상 단위)만 쓴다.

## 1. 셀 정의

| 셀 | 백본 | 학습 데이터 | 출발 가중치 | 손실 | 입력/라벨·청크 | LR·epoch·배치 | 시드 |
|---|---|---|---|---|---|---|---|
| `RF_POOL_L`(+`_s200`,`_s300`) | RhythmFormer | 하이브리드 풀 366 클립(`COTRAIN_ALLF_HYB4`, u8 캐시 `cotrain_allf_hyb4_u8`) | 공개 UBFC-rPPG RhythmFormer | 백본 고유(시간+주파수 하이브리드) | Standardized/Standardized, 128×128, 160, NDCHW | 1e-4 · 10 · 4 | 100/200/300 |
| `PM_POOL_L`(+`_s200`,`_s300`) | PhysMamba | 같은 풀(u8) | 공개 UBFC-rPPG PhysMamba | 백본 고유(Neg-Pearson) | DiffNormalized/DiffNormalized, 128×128, **160**(패치 42), NCDHW | 1e-4 · 10 · 4 | 100/200/300 |
| `PMbase` | PhysMamba | — (공개 가중치 그대로) | 공개 UBFC-rPPG PhysMamba | — | 청크 160 | — | — |
| `ML_NF_L`(+`_s200`,`_s300`) | SpecFP(FactorizePhys + 스펙트럼 손실) | MMPD 피험자 1~23(문헌 train) **+ 풀** (`mmpd_lit_train_cotrain_allf_hyb4_u8.csv`, 8,717 청크) | `NF_HYB_L` 같은 시드 체크포인트(2단계) | SpecFP 4항 + `NF_HR_TB=1` | Raw 72×72 / Standardized, 160 | 5e-5 · 10 · 4 | 100/200/300 |
| `ML_NFO_L`(+`_s200`,`_s300`) | SpecFP | MMPD 1~23 만(`mmpd_lit_train.csv`, 5,052 청크) | 공개 UBFC FactorizePhys FSAM_Res | SpecFP 4항 + `NF_HR_TB=1` | 같음 | 1e-4 · 10 · 4 | 100/200/300 |

- 풀 셀의 비교 상대(기존): SpecFP `NF_HYB_L` ×3, PhysFormer 고유 손실 `CoTAM_L` ×3(VIPL 8.70 / MMPD 7~33 9.98), FactorizePhys 분해 손실 `FP_OFF_HYB` ×3, 공개 가중치 RhythmFormer(`RFbase` VIPL 16.21, MMPD 7~33 11.90)·PhysMamba(`PMbase`, 이번에 평가).
- MMPD 문헌 분할: 툴박스 MMPD 인트라 설정의 영상 색인 0.7/0.8/1.0 분할을 피험자 단위로 옮긴 1~23 / 24~26(검증, 미사용 — `USE_LAST_EPOCH`) / 27~33(테스트 140 영상)이며 34 문서 Phase 2 의 `ML_BASE_L`(PhysFormer, MMPD 만, **5.68**)·`ML_HYB_L`(PhysFormer + 코퍼스 + 분해, 8.00)과 같은 목록이다. 문헌 값(RhythmFormer 논문 표: PhysFormer 11.99 · PhysFormer++ 4.88 · RhythmFormer 3.07)은 readout·분할 세부가 달라 참고로만 병기한다.
- 기존 `PM_CoTAM_L` ×3(같은 풀, 청크 128, float 캐시, MMPD 7~33 10.27/9.48/9.92)·`PM_HYB4_L` ×3(분해 손실, MMPD 7~33 9.89)은 **청크 길이 일관성 점검**(서술)에만 쓴다.

## 2. 평가 축과 집계
- 풀 셀·`PMbase`: VIPL-HR 전체 2,378 영상(zero-shot, u8 캐시, 문헌 GT `gt_lit`; `agg_vipl.py`), MMPD held-out 7~33 540 영상(u8 `mmpd_zs_7_33.csv`; 학습 run 의 테스트 = `agg_mmpd_cv2.py MODEL`, `PMbase` 는 `--outdir`), 실 축 9개 PURE·MR-NIRP 정지/소동작/대동작·DLCN·MCD A/B·RTNI·RTNC(float 캐시 `rf_real_*`/`pm_real_*`, 레인 공유 flock, 종료 후 삭제; `agg_model2.py`). 합성 축 C3/C6/E3·REALTEST·UBFC 는 생략(논문 표 미사용).
- 문헌 분할 셀: 테스트 27~33(`agg_mmpd_cv2.py --lit`, 피험자 ≥27 assert). **in-domain 으로 별도 표기, zero-shot 표와 섞지 않는다.**
- 시드 3개는 영상별 |오차| 를 시드 평균한 뒤 영상 단위 Wilcoxon 부호 순위(기존 판정 스크립트와 동일); 공개 가중치(시드 1)는 그 한 값과 짝짓는다. 가족별 Holm.

## 3. 가설(사전 등록)
| 가족 | 가설 | 검정(짝) | 판정 규칙 |
|---|---|---|---|
| B1 코퍼스 이득의 백본 의존성 | H-B1 SpecFP 는 같은 풀로 학습한 다른 백본보다 낮다 | `NF_HYB_L` vs `RF_POOL_L`, vs `PM_POOL_L` × {VIPL 전체, MMPD 7~33} (4 검정) | Δ=SpecFP−백본 < 0 이고 Holm p<.05 → 지지. 2차(서술): MR-NIRP 소/대동작·RTNI·PURE held-out·DLCN·MCD |
| B2 코퍼스 이득은 백본 일반 | H-B2 풀 학습은 각 백본의 공개 가중치보다 낮다 | `RF_POOL_L` vs `RFbase`/공개 RhythmFormer, `PM_POOL_L` vs `PMbase` × {VIPL, MMPD 7~33} (4 검정) | Δ<0, Holm p<.05 → 지지 |
| B3 MMPD 문헌 분할(in-domain) | H-B3a 풀 동반 2단계 vs MMPD 단독 SpecFP | `ML_NF_L` vs `ML_NFO_L` (27~33, n=140) | 양측; Δ<0 → 코퍼스가 in-domain 에도 도움, Δ>0 → PhysFormer(`ML_HYB_L` +2.32)처럼 역방향 |
|  | H-B3b SpecFP 2단계 vs PhysFormer MMPD 단독 | `ML_NF_L` vs `ML_BASE_L`(5.68, 시드 1) | Δ≤+0.3 → 비열등, Δ<0 Holm p<.05 → 우세 |
|  | H-B3c in-domain 이득 | `ML_NF_L` vs zero-shot `NF_HYB_L` 같은 140 영상(3시드 7.73; 시드 1 은 7.03) | Δ<0 Holm p<.05 → 지지 |

예상(정직 기록): B1 지지(두 백본 모두; RhythmFormer 공개 16.21·`RFA` 209 클립 코퍼스 MMPD 14.07 이 선행), B2 지지, B3a 는 PhysFormer 선례로 역방향 가능, B3b 는 체인 73 MMPD 5-fold(7.01 vs PhysFormer 5.58)로 미루어 기각 가능성 큼, B3c 지지. 결과가 어느 쪽이든 56 원고 표에 그대로 싣는다.

## 4. 기술 준비(11-02, 학습 전 완료)
- **패치 41**(BaseLoader): 모델 청크가 캐시 청크(160+1)보다 짧으면 앞부분만 쓰고 라벨도 같이 자름 — 보호용, 이번 체인에선 모든 백본이 160 이라 발동하지 않음.
- **패치 42**(PhysMambaTrainer): `PhysMamba(frames=청크 길이)`. 모델 끝의 `AdaptiveAvgPool3d((frames,1,1))`·`view(-1, length)` 가 128 고정이라 160 입력이 불가했음; 시간 축은 stride 4/2 → ×2 ×2 로 입력 길이로 복원되므로 frames=T 면 풀링이 항등. 스모크(`scripts/pm_smoke_t160.py`): 공개·`PM_HYB4_L` 체크포인트 결손 키 0, T=128·160 전방/역전파 정상(배치 2 피크 3.5 GB).
- **mamba 커널(pro6000)**: `mamba_ssm` 1.0.1 은 툴박스 `tools/mamba`(sm_120 패치)로 소스 빌드됐으나 `causal_conv1d` 1.0.2 가 sm_90 까지의 사전 빌드 휠이라 "no kernel image" → Alpha 의 동일 토치(2.11.0+cu128)·py3.10 빌드 `.so`(compute_120) 를 컨테이너 site-packages 에 복사(원본 `.so.sm90` 보관). `transformers` 5.16 이 `GreedySearchDecoderOnlyOutput` 를 없애 `mamba_ssm/utils/generation.py` 에 try/except 별칭 셰임.
- **yaml 생성기** `scripts/bb_yaml.py`(`tpl` 백본 틀 / `ev_axis` 실 축 ev yaml, 캐시 `rf_*`/`pm_*`), `vipl_yaml.py train` 이 `--template` 일 때 템플릿의 `DATA_FORMAT`(RhythmFormer NDCHW) 유지. u8 표준화 사이드카 `scripts/u8_stats_any.py` 로 풀 366·MMPD 660 영상 생성.
- 캐시: 풀 u8 3,665 청크·MMPD u8 7,260 청크 재구축(목록 5종 경로 결손 0), VIPL u8 96 GB 유지. pro6000 가용 293 GB; float 캐시 ≈ 150 GB 예상 → Z 레인이 삭제.

## 5. 운영
- `scripts/run_chain74_backbones.sh` 레인 R1 R2 R3(RhythmFormer 시드별) · P1 P2 P3(PhysMamba 시드별, P1 끝에 `PMbase`) · M(`ML_NF_L` ×3 직렬) · MO(`ML_NFO_L` ×3 직렬) · Z(판정 `backbone_judge.py` + 캐시 삭제); 기동 11-02 (아래 이력), GPU 게이트 24 GB(학습)/6 GB(평가), 레인 시작 시차 0~20 분, `RPPG_NUM_WORKERS=4 RPPG_MP_QUOTA=4`, BLAS 2 스레드. 예상 4~6 h.
- 실패 조건: yaml 검증(`NAME`·목록·RESUME·NDCHW) 불일치, 체크포인트 부재, `[finetune] loaded pretrained weights` 부재, SpecFP 셀의 `NF stats`/`hr_tb=1` 부재, 집계 실패 → `/tmp/chain74_<LANE>.fail`.
- 산출물: `results/agg_{RF_POOL_L*,PM_POOL_L*,PMbase}{,_VIPL,_MMPD}.json`, `agg_{ML_NF_L*,ML_NFO_L*}_MMPD.json`, `results/chain74_ft_*.log`, `results/backbone_judge.{txt,json}`; 논문 표는 `paper_tables_nf.py` 에 '백본 일치' 표와 'MMPD 문헌 분할' 표를 추가해 56/58 원고에 반영.

## 6. 결과 (체인 74 종료 2026-11-02 03:47 KST, pro6000; 판정 `scripts/backbone_judge.py` → `results/backbone_judge.{txt,json}`, 표 `paper_tables_nf.md` Table 10·11)

**실행**: 레인 9개 동시(최대 8 학습 병렬, GPU 60 GB, 디스크 여유 284 GB 유지), 00:22→03:47(3.4 h), 실패 레인 0. 동시 부하로 RhythmFormer 는 epoch 당 ≈20 분(1.3 s/it)이었으나 예상 안에 끝남. float 캐시 `rf_real_*`·`pm_real_*` 는 Z 레인이 삭제.

### 6.1 영상 단위 MAE(시드 영상별 평균 풀링; VIPL = 문헌 GT 2,378 / MMPD held-out 7~33 540 / 2차 축)

| 셀 | 시드 | VIPL | MMPD 7~33 | NIRP 정지 | NIRP 소동작 | NIRP 대동작 | RTNI | RTNC | PURE 07~10 | DLCN | MCD A | MCD B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 공개 RhythmFormer | 1 | 16.21 | 11.90 | 18.09 | 19.43 | 20.40 | 30.13 | 28.00 | 0.51 | 10.94 | 32.65 | 32.04 |
| `RF_POOL_L` RhythmFormer 풀 | 3 | **7.48** (7.50/7.75/7.17) | **8.49** (8.23/8.45/8.80) | 11.52 | 11.64 | 11.01 | 3.48 | 4.32 | 0.41 | 1.32 | 3.69 | 1.26 |
| 공개 PhysMamba `PMbase` | 1 | 10.86 | 12.74 | – | – | – | – | – | – | – | – | – |
| `PM_POOL_L` PhysMamba 풀 | 3 | **12.42** (12.73/12.31/12.23) | **10.16** (9.77/10.57/10.14) | 7.11 | 10.86 | **8.51** | 1.32 | 0.00 | 0.36 | 2.11 | 1.93 | 1.33 |
| (기존) PhysFormer 풀 `CoTAM_L` | 3 | 8.01 | 10.33 | 7.91 | 7.32 | 7.48 | 2.20 | 1.17 | 3.41 | 3.68 | 4.67 | 1.57 |
| (기존) FactorizePhys 풀+분해 `FP_OFF_HYB` | 3 | 8.44 | 10.66 | 6.54 | 10.44 | 14.90 | 1.32 | 3.08 | 0.51 | 0.89 | 0.79 | 1.22 |
| (기존) SpecFP `NF_HYB_L` | 3 | 5.82 | 8.83 | 6.86 | 6.12 | 11.33 | 1.46 | 2.27 | 0.62 | 0.81 | 0.70 | 1.85 |

### 6.2 판정(가족별 Holm)

| 가족 | 검정 | n | 기준 → 셀 | Δ | p | Holm | 판정 |
|---|---|---|---|---|---|---|---|
| B1 | VIPL: SpecFP vs RhythmFormer 풀 | 2378 | 7.48 → 5.82 | −1.66 | <.0001 | <.0001 | **지지** |
| B1 | MMPD 7~33: SpecFP vs RhythmFormer 풀 | 540 | 8.49 → 8.83 | +0.34 | .155 | .155 | **ns(RhythmFormer 풀 근소 우세)** |
| B1 | VIPL: SpecFP vs PhysMamba 풀 | 2378 | 12.42 → 5.82 | −6.60 | <.0001 | <.0001 | **지지** |
| B1 | MMPD 7~33: SpecFP vs PhysMamba 풀 | 540 | 10.16 → 8.83 | −1.33 | .0013 | .0025 | **지지** |
| B2 | VIPL: RhythmFormer 풀 vs 공개 | 2378 | 16.21 → 7.48 | −8.73 | <.0001 | <.0001 | **지지** |
| B2 | MMPD: RhythmFormer 풀 vs 공개 | 540 | 11.90 → 8.49 | −3.41 | <.0001 | <.0001 | **지지** |
| B2 | VIPL: PhysMamba 풀 vs 공개 | 2378 | 10.86 → 12.42 | **+1.56** | <.0001 | <.0001 | **기각(역방향)** |
| B2 | MMPD: PhysMamba 풀 vs 공개 | 540 | 12.74 → 10.16 | −2.58 | <.0001 | <.0001 | **지지** |
| B3a | `ML_NF_L`(2단계, MMPD+풀) vs `ML_NFO_L`(MMPD 단독) | 140 | 5.59 → 6.31 | +0.72 | .090 | .180 | ns(방향은 코퍼스 역효과) |
| B3b | `ML_NF_L` vs `ML_BASE_L`(PhysFormer MMPD 단독 5.68) | 140 | 5.68 → 6.31 | +0.63 | .646 | .646 | **기각(비열등 한계 +0.3 초과)** |
| B3c | `ML_NF_L` vs zero-shot `NF_HYB_L` 같은 140 영상 | 140 | 7.73 → 6.31 | −1.42 | .020 | .059 | 기각(Holm 후; 비조정은 유의) |

2차·서술(비조정 p): SpecFP vs RhythmFormer 풀 — MR-NIRP 소동작 6.12 vs 11.64(p=.03)·정지 6.86 vs 11.52(p=.025) 우세, PURE 0.62 vs 0.41·MCD B 열세(ns); SpecFP vs PhysMamba 풀 — DLCN 0.81 vs 2.11(p=.005) 우세, **MR-NIRP 대동작 11.33 vs 8.51(p=.09) 열세**, RTNC 2.27 vs 0.00(ns). 청크 길이 점검: `PM_POOL_L`(160) 10.16 vs `PM_CoTAM_L`(128) 9.89, Δ+0.27 p=.32 → 청크 길이 영향 없음. MMPD 문헌 분할 서술: `ML_NFO_L`(SpecFP MMPD 단독) 5.59 ± 0.94 vs PhysFormer MMPD 단독 5.68 (Δ−0.09 p=.47) · vs zero-shot 7.73 (−2.14 p=.002); PhysFormer 의 코퍼스 동반 `ML_HYB_L` 8.00 (+2.32 p=.001); 같은 140 영상 zero-shot: RhythmFormer 풀 7.62 ≈ SpecFP 7.73 (ns), PhysMamba 풀 10.45 (+2.71 p=.02). 움직임별(문헌 분할 테스트): SpecFP MMPD 단독 정지 1.79·회전 5.66·대화 1.09·**보행 17.61**, 2단계 3.54/4.62/0.98/18.88, PhysFormer MMPD 단독 2.17/7.44/1.44/15.19 — 보행은 모든 in-domain 셀에서 15~19.

### 6.3 결론(정직 보고)
1. **코퍼스 이득은 백본 일반이 아니다(H-B2 부분 지지)**: 같은 풀이 RhythmFormer 를 VIPL 16.21→7.48·MMPD 11.90→8.49 로 끌어올리고 모든 백본을 플리커 면역(RTNI 30→1.3~3.5)으로 만들지만, PhysMamba 는 VIPL 이 **악화**(10.86→12.42)되고 MMPD 만 개선 — FactorizePhys 분해 손실 셀(6.88→8.44)과 같은 패턴. 이미 강한 공개 가중치(PhysMamba·FactorizePhys)는 풀 미세조정만으로는 VIPL 에서 손해.
2. **SpecFP 의 우위는 VIPL 에서 확고(두 백본 모두 Holm <.0001), MMPD 는 RhythmFormer 풀 셀과 동급(8.83 vs 8.49, ns)**: RhythmFormer 풀 셀은 MMPD held-out·PURE 최저이나 MR-NIRP(주행) 전 축 11~12 로 약함; PhysMamba 풀 셀은 MR-NIRP 대동작 8.51 로 zero-shot 최저. 전 축 최저 또는 근접인 단일 설정은 SpecFP 뿐(52k 파라미터).
3. **MMPD 문헌 분할(in-domain)**: SpecFP 를 MMPD 만으로 학습하면 5.59 로 PhysFormer MMPD 단독 5.68 과 동급; 코퍼스 체크포인트에서 풀을 동반한 2단계는 6.31 로 열세(ns) — PhysFormer 선례(8.00)와 같은 방향. **in-domain 에서는 코퍼스 동반이 도움이 되지 않음**(H-B3a ns·H-B3b 기각·H-B3c Holm 후 기각). 문헌 값(RhythmFormer 3.07 등)은 readout·분할이 달라 직접 비교 불가.
4. 논문 반영: 58 본문 4.4 절·표 3, 4.7 절·표 6, 한계 문단; 보충 S1-T10/T11·S5(검정 30건 중 11건 지지). 후속 후보(사용자 결정): RhythmFormer 풀 셀에 스펙트럼 손실 이식(백본 교차 검증), PhysMamba 의 VIPL 악화 원인(학습률/epoch 과적합 여부) 진단.
