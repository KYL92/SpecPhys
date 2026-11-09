# 39. 체인 54 사전 등록 — FactorizePhys + 공식 코퍼스 (+ 하이브리드 분해) zero-shot (2026-10-21 04:58, virtual-face-e8)

## 1. 왜 (목표 /goal 의 남은 빈칸)
38 문서 근거 행렬에서 "기존 방법 대비 향상" 이 성립하지 않는 칸은 모두 **FactorizePhys 공개 가중치**(UBFC 42명, zero-shot VIPL 6.88·MMPD 9.79) 대비 zero-shot 칸이다(MMPD 동등, VIPL 열세). 체인 53 은 사용자의 원래 가설 "FactorizePhys + 우리 코퍼스 + 분해 손실 = 성능 최대치" 를 **UBFC 34 출발점**에서만 검정했고(기각), 코퍼스 전체(PURE·MCD 실 클립 포함)에 FactorizePhys 를 얹은 셀은 아직 없다. 체인 51·46b·53 은 공식 코퍼스 우위의 원인을 PURE·MCD 실 클립으로 귀속했으므로, 같은 재료를 FactorizePhys 에 주면 공개 체크포인트를 넘는지가 남은 검정이다.

## 2. 셀 (시드 100·200·300, 공개 FactorizePhys 체크포인트에서 미세조정, lr 1e-4·10 ep·batch 4, Raw 72×72·Standardized 라벨 — 체인 53 FactorizePhys 레시피 동일)
| 셀 | 학습 풀 | 손실 | 대응 PhysFormer 셀 |
|---|---|---|---|
| `FP_OFF_TW` | `COTRAIN_ALLF_TW` 203클립(공식 코퍼스: UBFC 34 + PURE + MCD + 가상얼굴 42 + 실 노이즈·압축 사본) | 기본 | 채택 모델 `CoTAF_TW_L`(8.88 / 10.84) |
| `FP_OFF_HYB` | `COTRAIN_ALLF_HYB4` 366클립(+ 이중 렌더 잡음 클립 + MMPD 1~6) | 4항 하이브리드(패치 34) | 하이브리드 `CoTAM_HYB4_L`(8.08 / 9.69) |

평가: 13축 + MMPD held-out 7~33(540편) + VIPL 2,378편(장치 기준값), 모두 zero-shot. 캐시 `fp_cotrain_allf_tw`·`fp_cotrain_allf_hyb4`(학습 후 삭제), 평가 축 캐시는 기존 `fp_real_*`·`fp_mmpd_u_B*`(묶음마다 삭제).

## 3. 가설·판정 규칙 (영상 단위 쌍대 Wilcoxon, 시드 3개 영상별 평균 풀링, 1차 가족 Holm 4건)
| 가설 | 1차 비교 | 지지 기준 |
|---|---|---|
| H-F1 코퍼스가 FactorizePhys 의 VIPL zero-shot 을 개선 | `FP_OFF_TW` vs 공개 FactorizePhys(`FPbase`) VIPL 문헌 GT | Δ<0, Holm p<.05 |
| H-F2 코퍼스가 FactorizePhys 의 MMPD zero-shot 을 개선 | `FP_OFF_TW` vs 공개 FactorizePhys MMPD held-out 전체 | Δ<0, Holm p<.05 |
| H-F3 분해 손실이 FactorizePhys 위에서 추가 개선 | `FP_OFF_HYB` vs 공개 FactorizePhys VIPL / MMPD(2건) | Δ<0, Holm p<.05 |
2차: MMPD 움직임 합산, VIPL 큰 움직임(v2)·휴대폰 움직임(v9), 13축(vs 공개 FactorizePhys 영상 단위), 백본 간(`FP_OFF_*` vs 같은 풀의 PhysFormer 5시드 풀링). 지지되면 38 §D 의 FactorizePhys 열을 "zero-shot 개선(코퍼스 귀속)" 으로 갱신하고 원고 §4.9 에 한 단락 추가; 기각되면 그대로 부정 결과로 기록(원고 주장 불변). 붕괴 점검: r<.3 이면 학습 사고 확인 후 재실행.

## 4. 비용·운영
학습 6 run(캐시 생성 ≈2.5 h 포함 ≈4 h) + 평가(13축 6 × ≈10 분, MMPD 6묶음 ≈2.5 h, VIPL 6 × ≈8 분) ≈ 8 h. 체인 52 Phase 3(virtual-face-04 세션)과 GPU 게이트 12 GB 로 병행, `RPPG_MP_QUOTA=3 RPPG_NUM_WORKERS=4`. 스크립트 `scripts/run_chain54_fp_official.sh`(로그 /tmp/chain54.log, 마커 /tmp/chain54.done), 판정 `scripts/fp_official_judge.py` → `results/fp_official_judge.json`.

## 5. 실행 기록
- 10-21 04:58 기동: `RPPG_MP_QUOTA=3 RPPG_NUM_WORKERS=4 bash scripts/run_chain54_fp_official.sh`(체인 52 Phase 3 와 병행).
- 10-21 07:00 게이트 12 → 7 GB 로 재기동(FactorizePhys 학습 ≈6 GB; 52p3 PhysFormer 학습 2개와 3자 병행, 가용 RAM 29 GB·전처리 없음).
- 10-21 08:29 평가 게이트 6 → 3 GB 로 재기동(GPU 가 52p3 학습 2개 + 타 사용자로 26.9 GB 점유되어 평가가 게이트에 막힘; FactorizePhys 평가 ≈2~3 GB).
- 10-21 09:12 **운영 변경**: MMPD 평가의 FactorizePhys float 전처리(워커당 4~5 GB, 묶음당 ≈85 GB 디스크)가 타 사용자 학습·체인 52 Phase 3 와 겹쳐 가용 RAM 9 GB → 체인 54 중단 후, MMPD 평가를 **uint8 캐시 경로**로 교체: `mmpd_all_u8`(660편, 체인 52 Phase 3 가 재생성) + 목록 `DataFileLists/mmpd_zs_7_33.csv`(held-out 7~33, 540편 5,940청크) 를 VIPL 로더로 읽어 Raw 72×72 로 변환(패치 32), 전처리 없음. 집계 `agg_mmpd_cv2.py --outdir runs/ev_{M}_MMPDZS M` → `agg_{M}_MMPD.json`(행 스키마 동일). 평가 게이트 3 GB. 13축·VIPL 평가는 그대로. 메모리 가드(10 GB 미만 일시 정지) 유지.
- 10-21 09:30 **검증 실패 → 경로 복귀**: uint8 캐시 MMPD 경로의 타당성 검사로 공개 FactorizePhys 체크포인트를 같은 경로로 평가하니 held-out 7~33 **13.16(움직임 16.32)** 로 float 정본 경로의 9.79(움직임 10.73)보다 +3.38(p=.011) 나쁨 → `mmpd_all_u8` → Raw 72×72 변환은 FactorizePhys(원시 RGB 입력)에 **무효**(PhysFormer DiffNormalized 재구성과 달리 수치 동치가 아님; 원인 후보: 스트리밍 캐시의 crop/스케일 처리). 체인 54 의 u8 경로 MMPD 결과 6건(13.0~14.1)은 `results/legacy_readout/*_u8path_invalid.json` 으로 격리하고, **체인 54b**(`run_chain54b_mmpd_float.sh`: float 정본 경로 MMPD_B1~6, 전처리 워커 1개, 묶음 전 가용 RAM ≥16 GB 대기, 체인 54 종료 후 실행)로 재평가. **주의**: VIPL 의 FactorizePhys zero-shot(공개 6.88, 체인 53·54 셀)은 모두 u8 경로(vipl_u8, `vipl_preprocess.py` 생성)라 같은 의심을 받으나 float 참조가 없음 — 별도 검사(VIPL 부분집합 float 전처리 vs u8) 필요 항목으로 기록.
- 10-21 09:50 **원인 분해(virtual-face-04 세션 진단)**: uint8 경로 격차 3.38 중 ≈2.7 은 **라벨 읽기** — `mmpd_all_u8` 의 라벨 청크는 PhysFormer 용 DiffNormalized 형인데 FactorizePhys 평가 pickle 의 `label_type` 'Standardized' 로 누적합 없이 GT 를 계산해 62/540 영상의 GT 가 어긋남(u8 예측을 float 경로 GT 로 읽으면 10.44, float 9.79 대비 +0.65); 크롭은 동일(픽셀 차 0), 128→72 이중 리사이즈 잔차 ≈0.8 그레이(상관 .9996) → 입력 측 잔차 ≈0.65. 함의: FactorizePhys 의 VIPL **문헌 GT**(장치 HR 사이드카) 수치는 유효하되 공통 ≈0.6 비관 편향 가능, VIPL **정본 GT** 수치와 u8 MMPD 수치는 무효(격리 유지). 체인 54b(float 정본 경로)가 올바른 조치.
- 10-21 09:57 **체인 54 종료(학습 6 run + 13축·VIPL 평가; MMPD 는 체인 54b 로)**: VIPL zero-shot 문헌 GT(3시드 영상별 평균 풀링, u8 경로 공통 — 공개 FactorizePhys 6.88 도 같은 경로): `FP_OFF_TW` **9.43**, `FP_OFF_HYB` **8.44** → **H-F1·H-F3(VIPL) 기각·역방향**(공개 FactorizePhys 대비 +2.55 / +1.56, Holm p<.0001; 큰 움직임 v2 +1.22 ns / +1.86 p=.028; 휴대폰 움직임 v9 +1.87 p=.044 / +5.28 p=.001). POS 7.77 보다도 열세(+1.66 p<.0001 / +0.67 p=.050), 같은 풀의 PhysFormer(8.88 / 8.08)보다도 열세(+0.55 p=.012 / +0.36 p<.0001), 소프 7.06 보다 열세. 즉 **공개 FactorizePhys 체크포인트를 우리 코퍼스(공식 203 / 하이브리드 366)로 미세조정하면 VIPL zero-shot 이 나빠지며**, 분해 손실 풀이 기본 풀보다는 낫다(8.44 < 9.43). 체인 53 U0(UBFC 34 만) 7.72 보다도 나쁘므로, 미세조정 자체(우리 레시피 lr 1e-4·10 ep·Haar 크롭·표준화 라벨)가 공개 체크포인트의 VIPL 일반화를 깎고 코퍼스 양이 늘수록 더 깎는 방향. 판정 전문 `results/fp_official_judge_vipl_interim.txt`. MMPD 판정은 체인 54b(float 경로) 후.

## 6. 최종 판정 (10-21 10:45; VIPL = u8 경로 문헌 GT 3시드 풀링, MMPD = u8 경로 예측 + float 정본 GT 재읽기, 기준도 같은 경로의 공개 체크포인트 `FPBASE_U8CHK`)

체인 54b(float 정본 경로 MMPD 재평가)는 워커 1개 전처리가 291 초/편(타 사용자 학습·체인 52 Phase 3 와의 CPU/IO 경합)으로 6묶음 ≈54 h 가 걸려 중단. 대신 virtual-face-04 세션의 진단(격차의 ≈2.7 이 라벨 읽기, 입력 잔차 ≈0.65)에 따라 u8 경로 예측을 float 경로 GT 로 다시 읽고(`agg_FP_OFF_*_MMPD.json`, 행에 `gt_src: float_path`), **같은 경로로 평가한 공개 체크포인트 `FPBASE_U8CHK`(10.44; float 정본 9.79 → 경로 비관 편향 ≈0.65)** 를 기준으로 경로 내 쌍대 비교했다. float 정본 재평가는 머신이 한가할 때(52 Phase 3 종료 후) 워커 3개로 다시 수행할 항목으로 남긴다.

| 가설 | 비교 | 결과(3시드 풀링) | 판정 |
|---|---|---|---|
| H-F1 코퍼스 → VIPL | `FP_OFF_TW` 9.43 vs 공개 6.88 | **+2.55(Holm p<.0001) 악화**; v9 +1.87(p=.044) | **기각(역방향)** |
| H-F2 코퍼스 → MMPD | `FP_OFF_TW` 10.56 vs 공개(같은 경로) 10.44 | +0.12(ns, Holm .33); 움직임 −0.51 ns | **기각(무효)** |
| H-F3 분해 → VIPL / MMPD | `FP_OFF_HYB` 8.44 vs 6.88 / 10.52 vs 10.44 | **+1.56(Holm p<.0001) 악화** / +0.08 ns | **기각(VIPL 역방향, MMPD 무효)** |

2차: 두 셀 모두 POS 보다 MMPD 우세(−1.86 p=.0009 / −1.90 p=.0001)이나 VIPL 은 POS 보다 열세(+1.66 / +0.67); 같은 풀의 PhysFormer 대비 `FP_OFF_TW` 는 MMPD 움직임 −1.09(p=.004) 우세·VIPL +0.55 열세, `FP_OFF_HYB` 는 VIPL +0.36·MMPD +0.83 열세; 소프(7.06 / 9.28)보다 모두 열세. 분해 풀은 기본 풀보다 VIPL 에서 낫다(8.44 < 9.43, 같은 방향으로 PhysFormer 8.08 < 8.88).

**결론**: 사용자의 원래 가설 "FactorizePhys + 우리 코퍼스 + 분해 손실 = 성능 최대치" 는 공식 코퍼스 전체에서도 기각된다. 공개 FactorizePhys 체크포인트는 우리 코퍼스로 미세조정하면 VIPL zero-shot 이 나빠지고 MMPD 는 변하지 않는다. 38 §D 의 FactorizePhys 열("zero-shot 동등~열세")은 이 실험으로도 바뀌지 않으며, 공식 코퍼스의 zero-shot 우위는 PhysFormer 백본에 한정된 현상(공개 PhysFormer 가중치가 약하기 때문)으로 서술해야 한다. 주의: FactorizePhys 미세조정 레시피(lr 1e-4·10 ep·Haar 크롭·표준화 라벨; Phase 0 는 UBFC 검증 손실로만 선택)가 공개 체크포인트의 일반화를 깎는 요인일 수 있으나, 같은 레시피의 PhysFormer 는 공개 가중치보다 크게 좋아지므로 백본 간 비대칭은 실재한다.
- 10-21 10:35 체인 54b 재기동(전처리 워커 2개, 묶음마다 가용 RAM ≥16 GB 대기): FactorizePhys 셀의 MMPD 정본 float 경로 수치를 확정하기 위한 야간 실행(경합 시 10~15 h). 완료 시 39 §6 표를 float 수치로 갱신.
- 10-21 11:39 virtual-face-04 세션 요청으로 체인 54b 를 체인 52 Phase 3 종료(마커 2개, 10-23 오전 예상) 이후로 보류(첫 묶음 전처리 74/120 폐기, 부하·RAM 게이트 유지).

## 7. 확정 판정 — MMPD 정본 float 경로 재평가 완료 (10-22 08:47, 체인 54b; §6 의 경로 내 비교를 대체)

| 가설 | 비교(3시드 영상별 평균 풀링) | 결과 | 판정 |
|---|---|---|---|
| H-F1 코퍼스 → VIPL | `FP_OFF_TW` 9.43 vs 공개 FactorizePhys 6.88 | **+2.55, Holm p<.0001 악화** | 기각(역방향) |
| H-F2 코퍼스 → MMPD | `FP_OFF_TW` **10.60** vs 공개 FactorizePhys **9.79**(float 정본) | **+0.81, Holm p=.023 악화**; 움직임 11.51 vs 10.73(+0.78 ns) | 기각(역방향) |
| H-F3 분해 → VIPL / MMPD | `FP_OFF_HYB` 8.44 vs 6.88 / **10.66** vs 9.79 | **+1.56 Holm p<.0001 악화** / +0.88 ns(움직임 +0.98 ns) | 기각(VIPL 역방향·MMPD 무효) |

- 시드별 MMPD held-out: TW 10.55 / 10.42 / 10.83, HYB 10.85 / 10.87 / 10.27. 두 셀 모두 POS(12.42)보다는 우세(−1.82 p=.002 / −1.76 p=.0001), 소프(9.28)보다 열세(+1.22 p=.005 / +1.24 ns); 같은 풀의 PhysFormer 대비 `FP_OFF_TW` 는 전체 동등(−0.24 ns)·움직임 −0.90(p=.016) 우세, `FP_OFF_HYB` 는 +0.97 ns.
- §6 의 u8 경로 내 비교(10.56 / 10.52 vs 10.44, ns)는 정본 경로에서 TW 셀이 유의 악화로 바뀜 → u8 경로 수치는 참고로만 남기고 이 표를 최종으로 한다.
- **최종 결론**: 공개 FactorizePhys 체크포인트에 우리 코퍼스(±분해 손실)를 미세조정하면 VIPL·MMPD zero-shot 모두 좋아지지 않으며(VIPL 은 두 셀 모두 유의 악화, MMPD 는 코퍼스 셀 유의 악화·분해 셀 동등), 사용자 가설 "FactorizePhys + 우리 코퍼스 + 분해 손실 = 최대 성능" 은 확정 기각. 38 §E 갱신. 판정 전문 `results/fp_official_judge_final.{txt,json}`, 체인 로그 `results/chain54b_mmpd_float.log`.
