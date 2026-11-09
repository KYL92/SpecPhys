# 27단계 — 학습된 라우터 + 시드 앙상블/가중치 소프 (체인 45, 2026-10-08 사전 등록)

표적: 방법 편 원고(`28_manuscript_method.md` §5)의 두 한계 — "the decomposition does not yet route itself"
(자기 라우팅 부재)와 "per-seed significance 3/5"(시드 분산). **재학습 없음** — 기존 5시드 가중치·평가 pickle만 사용.

## 사전 등록 설계 (실행 전 고정)

### 공통 전제
- 공식 모델 = 파형 뱅크 co-training 모델 CoTAF_TW_L(이하 "기준 TW"), 움직임 축 보조 = 하이브리드 분해 모델
  CoTAM_HYB4_L(이하 "하이브리드 HYB4"). 각 5시드(s100/s200/s300/s400/s500). 고전 = POS.
- 평가 집합 = MMPD held-out 피험자 7~33, 540클립. per-clip 오차는 정본 toolbox 프로토콜 agg JSON에서만.
- "5시드 풀링 오차" = 시드별 per-clip 절대오차의 평균(`pool5_judge.py` 정의). 기준 수치: TW 10.84, HYB4 9.69,
  GT-모션 라우팅(정지→TW, 움직임→HYB4) 9.74, 움직임 12.42→10.60.
- 모든 쌍대 비교는 per-clip Wilcoxon 부호순위(양측).

### 트랙 A — 학습된 라우터 (CPU, 모델 학습 없음)
가설 A: GT 모션 라벨 없이, 관측 가능한 입력 특징 + 모델 출력 내재 특징만으로 클립별 추정기를 고르는 라우터가
GT-모션 라우팅(9.74)에 근접하거나 상회한다.
- 후보 추정기(3-way): 기준 TW, 하이브리드 HYB4, POS. PhysMamba 하이브리드(PM_HYB4_L, 3시드)는 보조 분석만.
- 라우터 입력 특징(**MMPD 메타 라벨 light/motion/exercise/skin 사용 금지**):
  ① 관측 특징(원본 .mat 320×240@30fps, Haar 얼굴 추적 5프레임 간격): bbox 중심 변위 평균·최대, 중심 위치 표준편차,
     bbox 크기 변화율(log 면적 std), 검출률, 평균 얼굴 크기, 프레임 차분 에너지(전체/얼굴 상자, 회색조),
     얼굴 상자 평균 휘도, 휘도 시간 표준편차, 휘도 저주파 드리프트(1초 이동평균 std), 휘도 맥박대역 전력비(0.6~3.3Hz).
  ② 출력 내재 특징: 각 후보 출력 파형(정본 전처리: cumsum→detrend→0.6~3.3Hz)의 자기 피크 스펙트럼 SNR
     (toolbox `_calculate_SNR` 정의, 기준 주파수 = 자기 피크), 후보 간 HR 불일치 |HR_TW−HR_HYB4|·|HR_TW−HR_POS|·
     |HR_HYB4−HR_POS|, 시드 간 HR 불일치(TW·HYB4 각 5시드 HR std — 앙상블 불일치, 배치 시 관측 가능), HR 수준(TW).
     모델 출력 특징은 시드별로 계산해 5시드 평균(SNR, HR). POS 파형은 `eval_classical_mmpd.py`와 동일 프로토콜
     (YCrCb 피부 마스크 평균 → POS_WANG)로 재계산.
  ③ 하이브리드 노이즈 채널 출력: 평가 pickle에 `predictions/labels`만 존재(노이즈 채널·conf 덤프 없음) →
     **생략**(재평가 금지 규칙).
- 라벨: **주 결과 = 클립별 oracle 최적 후보(5시드 풀링 오차 argmin, 3-class) 분류**. 보조 = 후보별 오차 회귀 후 argmin.
- 분류기: 주 = 로지스틱 회귀(표준화, class_weight=balanced, C=1.0); 보조 = 얕은 HistGradientBoosting(depth 3,
  100 iter). 특징 중요도(순열 중요도) 보고.
- 분할: held-out 7~33 안 **피험자 분리 5-fold CV**(GroupKFold, 피험자 단위). 보조: MMPD 1~6(학습 분할, HYB4가
  학습한 데이터 → 출력 특징 낙관 편향)만으로 학습 → 7~33 평가.
- 판정 지표: 라우터 합성 MAE(선택 후보의 5시드 풀링 오차), 움직임 분할(정지 MO1/움직임 MO2~4/Walking MO4),
  Wilcoxon vs TW·vs HYB4·vs GT-모션 라우팅, oracle 대비 regret, 선택 정확도(oracle 일치율)·선택 비중.
- **사전 등록 성공 기준**: 합성 MAE가 HYB4 단독(9.69) 대비 유의 개선(p<.05) **이고** GT-모션 라우팅(9.74) 이하 →
  "자기 라우팅 성공"; HYB4 대비 비유의·GT-모션과 동등(비유의) → "모션 라벨 없이 동등(부분 성공)"; 그 외 부정 결과.
- 주의(사전 기록): oracle 라벨은 풀링 오차로 정의되므로 라우터 평가도 풀링 오차로 — 단일 시드 배치 성능은 보조로
  병기. 항상-HYB4(9.69)가 GT-모션 라우팅(9.74)보다 이미 낮으므로 "HYB4 단독 유의 개선"이 실질 관문.

### 트랙 B — 시드 앙상블과 가중치 소프 (평가만, GPU 경량)
가설 B: 5시드 HYB4의 시드 분산은 예측 앙상블 또는 균등 가중치 평균(uniform soup)으로 축소되며, 단일 시드 평균보다
유의하게 낮은 MAE를 준다.
- B1 예측 앙상블(GPU 불필요): 기존 pickle에서 5시드 예측 파형을 청크 정렬·연결 후 **시드별 z-표준화 → 평균**,
  정본 프로토콜로 HR 추출 → per-clip 오차. HYB4·TW 각각, MMPD held-out + pickle이 있는 13축 전부.
  "시드별 오차의 평균(풀링)"과 "평균 예측의 오차(앙상블)"는 다른 양 — 둘 다 표에 병기.
- B2 가중치 소프: HYB4 5시드 Epoch9 state_dict 균등 평균(키 일치·dtype 확인, BatchNorm running 버퍼 포함 평균,
  `num_batches_tracked`는 정수 평균) → `runs/finetune_CoTAM_HYB4_SOUP_L/PreTrainedModels/
  SYNTH_CoTAM_HYB4_SOUP_L_physformer_Epoch9.pth`. yaml은 `gen_yaml`로 HYB4와 동일 구성 복제(평가 전용),
  `RPPG_NOISE_HEAD=1`(노이즈 헤드 키 포함 strict 로드). **평가만** MMPD B1~6 + 13축, GPU 게이트(6000MB),
  배치마다 drop_cache. 집계 `agg_model2.py` + `agg_mmpd.py`.
- 판정: 앙상블/소프 vs 5시드 풀링 평균 vs 최고 단일 시드 vs 시드별, 움직임 분할 + Wilcoxon(per-clip, 풀링 오차 대응).
  13축 부작용(DLCN·NIRP·PURE 등) 병기. 소프 발산 시 그대로 보고.
- 결합(라우터가 소프/TW/POS 중 선택)은 트랙 A·B 모두 성공 시에만 보조.

### 실행
- 체인 `scripts/run_chain45_router_soup.sh`(chain_lib.sh, `/tmp/chain45.log`, 마커 `/tmp/chain45.done`),
  `scripts/make_soup.py`, `scripts/seed_ensemble.py`, `scripts/mmpd_obs_features.py`(관측 특징, CPU 2워커),
  `scripts/learn_router.py`. 결과 `_workspace/data/results/{router_cv.json, seed_ensemble.json, mmpd_obs_features.json,
  agg_CoTAM_HYB4_SOUP_L*.json}`.
- 재학습 없음. 타인 프로세스 종료 없음. 전처리 캐시 배치 완료 즉시 삭제.

---
(아래는 실행 후 기록)

## 구현 기록 (2026-10-08 21:00 체인 45 기동)
- 소프: 5시드 state_dict 503키(공통 499 + conf_head 4키). **발견**: s400/s500은 체인 41 이후 코드로 학습되어 `conf_head`
  키(하이브리드 HYB4에서 손실 미연결·forward 미사용·무작위 초기화)를 추가로 가짐(s100~s300엔 없음). 공통 키 5시드 평균,
  conf_head는 보유 2시드 평균(현 코드의 strict 로드용, 예측 경로 무관). 시드 간 파라미터 최대 격차 7.69(단일 원소).
- 평가 yaml `yaml_CoTAM_HYB4_SOUP_L/`(gen_yaml로 HYB4 ev yaml 19종 복제, finetune yaml 삭제), env `RPPG_NOISE_HEAD=1`.
- 관측 특징: 660클립 CPU 2워커(클립당 ≈14 s), 토대 `scripts/mmpd_obs_features.py`.
- 노이즈 채널 출력 특징: pickle에 `predictions/labels/label_type/fs`만 존재 → 생략(사전 등록대로).

## 결과 B1 — 시드 예측 앙상블 (`scripts/seed_ensemble.py`, results/seed_ensemble.json)
"풀링" = 시드별 오차의 평균(논문 수치), "앙상블" = 5시드 z-표준화 파형 평균의 오차. per-clip Wilcoxon(풀링 vs 앙상블).

| MMPD held-out(540) | 풀링 | 최고 단일 시드 | 앙상블 | p(vs 풀링) |
|---|---|---|---|---|
| 하이브리드 HYB4 전체 | 9.69 | — | **9.10** | **.0003** |
| HYB4 정지(MO1) | 8.33 | — | **7.33** | **<.0001** |
| HYB4 움직임(MO2~4) | 10.60 | — | 10.28 | .14 |
| HYB4 Walking | 18.12 | — | 18.39 | .95 |
| 기준 TW 전체 | 10.84 | — | **10.17** | **.0002** |
| TW 정지 | 8.47 | — | 7.81 | .0039 |
| TW 움직임 | 12.42 | — | 11.74 | .013 |
| TW Walking | 20.78 | — | 19.89 | .25 |
| (MMPD 전체 660) HYB4 | 8.45 | 8.04(s100) | 7.94 | .0003 (vs 최고단일 p=.78) |
| (MMPD 전체 660) TW | 10.56 | 9.97(s200) | 9.67 | <.0001 (vs 최고단일 p=.55) |

13축(HYB4, 풀링→앙상블): MCD_B 0.96→0.48(p=.08) · MCD_A 4.19→4.83(ns) · PURE 1.09→0.53(ns; s100 2.66 이상치 상쇄) ·
DLCN 1.84→**0.97(p=.036)** · NIRP still 8.44→8.14(ns) · NIRP small 10.71→**7.63(p=.0014)** · NIRP large 12.09→10.41(ns) ·
REALTEST/RTNI/RTNC 소표본 ns · **C3 23.65→33.52 / C6 7.43→16.05 악화**(n=4: 시드 3개가 배가 모드인 축에서는 파형 평균이
배가 다수결로 수렴 — 앙상블이 다수결 실패를 그대로 상속) · E3 13.14→0.06.
13축(TW): NIRP still 7.47→10.22(p=.055 악화 방향)·small 8.60→10.32(ns)·DLCN 1.92→2.72(ns) — TW는 NIRP에서 앙상블 역효과,
C3 3.57→0.06(s200 배가 상쇄).
해석(잠정): 앙상블 이득은 **정지 조건에 집중**(시드 간 상보적 소오차 평균화), 움직임·Walking은 비유의 — 시드 분산 축소는
"per-seed 3/5" 한계의 해결책이 아니라 정지 축의 추가 이득. 앙상블은 추론 비용 5배(배치 관점 단점) — 소프(B2)가 같은 효과를
1배 비용으로 주는지가 관건.

## 진행 상태 (2026-10-08 21:50, 세션 중단 시점 — 체인은 자율 진행 중)
- 완료: 소프 생성, B1 앙상블(위 표), 소프 13축 평가(C3/C6/E3/DLCN/MCD/NIRP/PURE/REALTEST/RTN 전부 OK), MMPD B1 OK.
- 진행 중: 소프 MMPD B2~B6 평가 → `agg_model2.py`/`agg_mmpd.py` 집계(`/tmp/agg_CoTAM_HYB4_SOUP_L.log`).
- 메모리 사고: 시스템 가용 메모리 4GB까지 하락(툴박스 MMPD 전처리 워커 ~5GB×4 + 타 사용자) → 관측 특징 추출(181/660 저장)을
  **자발적으로 중단**. `scripts/chain45_resume_features.sh`(nohup)가 소프 집계 완료를 기다려 3워커로 재개 → `/tmp/chain45_feat.done`
  → 체인이 `learn_router.py`를 실행하고 `/tmp/chain45.done`을 남김(예상 23:30~00:30).
- 남은 작업(체인 완료 후): ① `/tmp/chain45.log`·`/tmp/chain45_router.log`·`results/router_cv.json`·`agg_CoTAM_HYB4_SOUP_L*.json` 확인
  ② 본 문서에 B2 소프 결과표·트랙 A 결과표·판정·논문 반영 지침 추가 ③ `.claude/CLAUDE.md` 완료 행 추가.

## 결과 B2 — 가중치 소프 (`scripts/make_soup.py` → `CoTAM_HYB4_SOUP_L`, 평가 체인 45, 판정 `scripts/soup_judge.py` → results/soup_judge.json, 2026-10-08 22:10)

소프 평가 실행에서 8클립(held-out 3)이 Haar "No Face Detected"로 탈락(이 실행에서만, 시드 평가에는 존재) → 비교는 **공통 537클립**에서.
"풀링" = 시드별 오차 평균(논문 수치), "앙상블" = B1의 5시드 파형 평균, "소프" = 가중치 균등 평균 단일 모델(추론 비용 1배).

| MMPD held-out(공통 537) | 풀링 | 최고 단일(s100) | 앙상블(5배 비용) | **소프(1배 비용)** | p 소프 vs 풀링 | vs 최고단일 | vs 앙상블 |
|---|---|---|---|---|---|---|---|
| 전체 | 9.69 | 9.15 | 9.15 | **9.28** | **.014** | .80 | .57 |
| 정지(MO1) | 8.34 | 7.97 | 7.40 | **7.91** | **.019** | .39 | .22 |
| 움직임(MO2~4) | 10.58 | 9.94 | 10.31 | 10.18 | .16 | .50 | .92 |
| Walking(MO4) | 18.12 | 16.43 | 18.39 | 19.50 | .20 | **.006(악화)** | .25 |
| 조건부 배치(정지→기준 TW) | 9.70 | — | — | 9.46 | .16 | — | — |

시드별 전체: s100 9.15 · s200 9.74 · s300 10.34 · s400 9.81 · s500 9.41 → 소프 9.28은 상위 2위 수준(최고 단일과 비유의).

13축(풀링 → 소프, vid 대응 Wilcoxon): NIRP still 8.44→**5.78(p=.013)** · NIRP small 10.71→**7.86(p=.055)** · NIRP large 12.09→10.87(ns) ·
DLCN 1.84→1.10(ns) · PURE 1.09→0.55(ns) · MCD_A 4.19→3.96(ns) · MCD_B 0.96→2.33(ns) · REALTEST 2.33→1.10(ns) ·
**RTNI 2.07→7.58 · RTNC 2.66→6.59(n=8, ns — 수치상 악화, 실노이즈 소표본)** · C3 23.65→33.57(n=4, 배가 2 — 앙상블과 같은 배가 상속) ·
C6 7.43→4.57 · E3 13.14→16.00(n=4). 배가·반감: 실데이터 13축 전부 0(NIRP small 배가 1).

### 트랙 B 판정 (사전 등록 가설 B 대조)
- **지지(부분)**: 소프는 5시드 풀링 대비 전체 유의 개선(9.69→9.28, p=.014)이며, 5배 비용의 예측 앙상블(9.15)과 통계적으로 동등 —
  "앙상블 효과를 1배 추론 비용으로" 확보. 시드 분산도 축소(단일 모델이 시드 분포 상위에 위치). MR-NIRP 정지·소동작 축에서 유의/경계 개선.
- **미지지**: 이득은 **정지 조건에 집중**(앙상블과 동일 패턴). 움직임 분할은 비유의, Walking은 최고 단일 시드(s100) 대비 유의 악화(p=.006) —
  시드 평균이 Walking의 시드 의존 이득을 희석한다. 따라서 **"per-seed 유의 3/5" 한계(움직임 축)의 해법이 아니다**.
- 부작용 경계: 실노이즈 축(RTNI/RTNC, n=8)에서 수치상 큰 악화(ns) — 소프 채택 시 이 축 재확인 필요. 합성 C3 배가 상속.
- 채택 지침: 소프를 **정지·저동작 배치용 단일 모델 대안**으로 보조 기록(공식 모델·하이브리드 분해 모델의 지위는 불변).
  논문(방법 편 §4 또는 부록)에는 "5시드 가중치 평균은 정지 조건 MAE를 유의하게 낮추고(p=.019) 앙상블과 동등하나, 움직임 분할의
  시드 의존성은 해소하지 못한다"로 한 문단. 소프 가중치: `runs/finetune_CoTAM_HYB4_SOUP_L/PreTrainedModels/`.

## 결과 A — 학습된 라우터 (`scripts/learn_router.py` → results/router_cv.json, 로그 /tmp/chain45_router.log, 2026-10-08 22:51)

관측 특징 660클립 추출 완료(`mmpd_obs_features.json`), 출력 특징은 pickle에서 정본 프로토콜로 재계산(agg HR과 평균 차 0.000 BPM, POS 재계산도 정본과 0.000 BPM 정합).
held-out oracle 라벨 비중 기준 TW / 하이브리드 HYB4 / POS = 0.34 / 0.38 / 0.29 — 세 추정기가 고르게 "어딘가에서 최선".

| held-out 7~33 (540, 5시드 풀링 오차) | MAE | 정지 | 움직임 | Walking | p vs 항상-HYB4 | p vs GT-모션 라우팅 | regret(oracle 대비) | 선택 정확도 |
|---|---|---|---|---|---|---|---|---|
| 항상-TW(기준 모델) | 10.84 | 8.47 | 12.42 | 20.78 | <.0001 | <.0001 | 4.89 | .34 |
| 항상-HYB4(하이브리드 분해 모델) | **9.69** | 8.33 | 10.60 | 18.12 | — | .16 | 3.74 | .38 |
| GT-모션 라우팅(정지→TW, 움직임→HYB4) | 9.74 | 8.47 | 10.60 | 18.12 | .16 | — | 3.79 | .40 |
| oracle 2-way(TW/HYB4) | 7.77 | 6.02 | 8.93 | 16.37 | <.0001 | <.0001 | 1.82 | .71 |
| oracle 3-way(+POS) | 5.95 | 4.89 | 6.66 | 13.24 | <.0001 | <.0001 | 0 | 1.00 |
| **[주, 사전 등록] 로지스틱 3-way, 관측+출력 특징** | **10.19** | 9.01 | 10.98 | 18.87 | **.087(악화 방향)** | .52 | 4.24 | .49 |
| 사후 최선 변형: GBM 회귀→argmin 3-way | 9.25 | 7.70 | 10.28 | 18.53 | .11 | .018 | 3.30 | .45 |
| 관측 모션 검출기 라우팅(얼굴 추적 → 움직임 여부, AUC .92) | 9.97 | 8.53 | 10.93 | 18.12 | .0098(악화) | .012(악화) | 4.02 | .41 |
| 관측 특징만(obs) 3-way 로지스틱 | 10.91 | 9.41 | 11.91 | 20.44 | .0002(악화) | .0006(악화) | 4.96 | .37 |
| 출력 특징만(out) 3-way 로지스틱 | 9.98 | 8.66 | 10.86 | 19.39 | .34 | .89 | 4.03 | .50 |

24개 변형(분류/회귀 × 로지스틱/GBM × 관측/출력/전체 × 2-way/3-way) 중 **항상-HYB4를 유의하게 이기는 변형은 0개**(최선 9.25, p=.11 — 사후 선택이므로 주장 불가).
보조 — MMPD 1~6(학습 분할)로 학습해 7~33 평가: 최선 9.41(p vs HYB4 .14), 역시 비유의. 순열 중요도(주 라우터): 얼굴 변위 평균 +0.38 ≫ TW 시드 간 HR 불일치 +0.22 > bbox 크기 변동 +0.19 > 프레임 차분 +0.16 — 라우터는 사실상 "움직임 검출기"로 수렴.

### 트랙 A 판정 (사전 등록 기준 대조 — 정직 보고)
- 사전 등록 성공 기준(HYB4 단독 대비 유의 개선 **그리고** GT-모션 라우팅 이하)은 **미달**. 주 라우터는 GT-모션 라우팅과 통계적으로 동등(p=.52)하지만 항상-HYB4보다 수치상 나쁘고(10.19 vs 9.69, p=.087), 설계서가 미리 적어 둔 실질 관문("HYB4 단독 유의 개선")을 넘지 못했다 → **부정 결과: 자기 라우팅 실패**.
- 원인 진단: ① oracle 상한(5.95)과의 격차는 크지만, 관측 특징(움직임·조명)과 출력 내재 특징(SNR·HR 불일치·시드 불일치) 어느 쪽도 "어느 모델이 이 클립에서 틀리는가"를 담지 못한다 — 선택 정확도 최고 .56(3-way 우연 .38). ② 움직임 검출은 잘 된다(AUC .92)지만 **GT-모션 라우팅 자체가 항상-HYB4보다 낫지 않다**(9.74 vs 9.69) — MMPD에서는 라우팅의 가치가 "정지/움직임" 축에 있지 않고 oracle 라벨이 가리키는 클립별 실패 패턴에 있으며, 그 패턴은 입력·출력 통계에 드러나지 않는다(24단계 트랙 B의 결론 재확인). ③ 출력 특징만 쓴 변형이 관측 특징만 쓴 변형보다 일관되게 낫다 — 그러나 둘을 합쳐도 HYB4를 넘지 못함.
- 체인 41(학습된 신뢰도 헤드)·체인 45(사후 라우터)로 **자기 라우팅 축 2회 부정** → 이 축은 종결. 남는 배치 정책은 "항상-하이브리드(MMPD류 모바일 도메인)" + "도메인 단위 축 정책(NIR 대동작→POS)"뿐이며 이는 이미 논문에 있다.

## 종합 판정 (27단계, 체인 45)
| 가설 | 결과 | 판정 |
|---|---|---|
| A 자기 라우팅(GT 모션 라벨 없이 oracle 이득 실현) | 주 라우터 10.19 vs 항상-HYB4 9.69(p=.087 악화 방향), 24변형 전부 비유의, oracle 5.95 미접근 | **부정** — 자기 라우팅 축 종결(2회 부정) |
| B1 시드 예측 앙상블 | 9.69→9.10(p=.0003), 정지 집중(p<.0001), 움직임·Walking ns, 비용 5배 | 부분 지지 |
| B2 가중치 소프(1배 비용) | 9.69→9.28(p=.014), 앙상블과 동등(p=.57), 정지 집중(p=.019), 움직임 ns, Walking 최고 단일 대비 악화(p=.006), NIRP 정지 유의 개선 | 부분 지지 — 정지·저동작 배치용 단일 모델 대안으로 보조 채택 |

공식 모델(CoTAF_TW_L)·움직임 보조 모델(CoTAM_HYB4_L)의 지위 불변. 재학습 없음.

### 논문 반영 지침 (`28_manuscript_method.md` §5 한계 문장 수정안)
- 현재: "The learned confidence head is a negative result; the decomposition does not yet route itself."
- 수정안: "The decomposition does not route itself: a learned confidence head (ratio-supervised) and a post-hoc router trained on oracle labels from
  observable motion/illumination statistics and output-intrinsic features (spectral SNR, cross-estimator and cross-seed HR disagreement) both fail to
  beat always-hybrid on held-out subjects (10.19 vs 9.69, *p* = .09; best of 24 variants 9.25, *n.s.*), despite a large oracle gap (5.95). Which
  estimator fails on a given clip is not legible in its inputs or outputs — routing by observable condition (motion, domain) is the only policy we
  can defend."
- 현재: "Per-seed significance is three of five ..."
- 추가 문장: "Averaging the five seeds — as predictions or as weights (uniform soup, no retraining) — lowers held-out MAE (9.69→9.15 / 9.28, *p* ≤ .014)
  but only in the stationary stratum; the motion-stratum seed dependence is not removed, so the pooled statistics remain the honest summary."
- 표: 부록에 소프 1행(13축 + MMPD 분할) 추가 가능(`results/agg_CoTAM_HYB4_SOUP_L*.json`, `soup_judge.json`).
