# 6b단계 준비 보고서: CoTNH 귀인 분리 ablation (CoTNC / CoTNW / CoTNH2)

작성: physformer-training-engineer (2026-08-25). **GPU 실행 없음** — 데이터셋 조립·합성 생성(CPU)·yaml·실행 스크립트 준비까지. 학습/평가는 `scripts/run_phase6b.sh`로 GPU 여유 시 실행.

## 0. 목적

17 보고서 §요약 4: CoTNH(COTRAIN_H, 108클립)는 CoTN(COTRAIN_N, 88클립) 대비 **(a) neurokit 합성 20클립 → 고조파 그리드 TRAIN20H 교체**와 **(b) 압축 증강 20클립 추가**가 동시에 바뀌어 MCD 개선(42.63→5.19)을 파형 다양성에 귀인할 수 없다. 12 보고서 §4: TRAIN20H의 a2/φ2 밀도가 실분포와 반대(a2<0.3에 55%, φ2 1.4–2.8 rad에 60%). 이를 분리·개선하기 위한 3개 모델을 준비한다.

| 모델 | 학습셋 | 클립 | 파형(201~220) | 압축 증강(501~510, 601~610) | 검증 질문 |
|---|---|---|---|---|---|
| CoTN (기존) | COTRAIN_N | 88 | neurokit (TRAIN20) | 없음 | 기준 |
| **CoTNC** | COTRAIN_NC | 108 | neurokit (TRAIN20) | 있음 (601~610 = **TRAIN20** 홀수 압축) | 압축 축 단독 효과 |
| **CoTNW** | COTRAIN_NW | 88 | 해석 그리드 (TRAIN20H) | 없음 | 파형 다양성 단독 효과 |
| CoTNH (기존) | COTRAIN_H | 108 | 해석 그리드 (TRAIN20H) | 있음 (601~610 = TRAIN20H 홀수 압축) | 둘 다 |
| **CoTNH2** | COTRAIN_H2 | 108 | 실분포 정합 (TRAIN20H2) | 있음 (601~610 = **TRAIN20H2** 홀수 압축) | 파형 밀도 교정 효과 |

## 1. COTRAIN_N vs COTRAIN_H 링크 대상 비교 (실측 `readlink`)

| 링크 범위 | COTRAIN_N (88) | COTRAIN_H (108) | 차이 |
|---|---|---|---|
| subject1~41 (34개; 2,6,7,19,21,28,29 제외) | `UBFC_dataset/DATASET_2/subjectN` | 동일 | 없음 |
| subject201~220 (20개) | `TRAIN20/subject1~20` (neurokit, seed 101~120, BPM 55–115) | `TRAIN20H/subject1~20` (해석 파형 a2 그리드, seed 501~520, BPM 52–118) | **교체** (파형·시드·BPM 모두 다름) |
| subject301~341 (34개; 302,306,307,319,321,328,329 제외) | `COTRAIN_NOISY_REAL/subjectN` | 동일 | 없음 |
| subject501~510 (10개) | 없음 | `CMP_REAL/subject501~510` = UBFC subject 1,3,4,5,8,9,10,11,12,13 mpeg4 q6 사본 | **추가** |
| subject601~610 (10개) | 없음 | `CMP_SYN/subject601~610` = **TRAIN20H subject 1,3,5,…,19** mpeg4 q6 사본 | **추가** (GT 해시로 출처 확인) |

**발견 1**: 기존 `CMP_SYN`(601~610)은 TRAIN20H 홀수 subject의 압축본이다. 따라서 "CoTN + 기존 CMP_SYN"으로 CoTNC를 만들면 고조파 그리드 파형 10클립이 섞여 파형 축이 오염된다. 지시("파형은 neurokit 그대로")에 맞추기 위해 **`CMP_SYN_N`**(TRAIN20 subject 1,3,…,19 → 601~610, 동일 `compress_aug.py --q 6`)을 새로 만들어 COTRAIN_NC에 연결했다. 같은 논리로 COTRAIN_H2의 601~610도 TRAIN20H 압축본이 아니라 **`CMP_SYN_H2`**(TRAIN20H2 홀수 압축본)로 연결했다 — 그렇지 않으면 H2 학습셋에 그리드 파형 10클립이 잔존해 "밀도 교정" 비교가 흐려진다. (엄격히 "201~220만 교체"를 원하면 `COTRAIN_H2/subject60x` 링크를 `CMP_SYN/subject60x`로 되돌리면 된다.)

## 2. 조립된 데이터셋 (전부 심볼릭링크, 원본 복사 없음)

| 데이터셋 | 링크 수 | 1~41 | 201~220 → | 301~341 | 501~510 → | 601~610 → | 무결성(vid+GT) |
|---|---|---|---|---|---|---|---|
| `COTRAIN_NC` | 108 | UBFC 34 | `TRAIN20` | NOISY_REAL 34 | `CMP_REAL` | `CMP_SYN_N` (신규) | 108/108 |
| `COTRAIN_NW` | 88 | UBFC 34 | `TRAIN20H` | NOISY_REAL 34 | – | – | 88/88 |
| `COTRAIN_H2` | 108 | UBFC 34 | `TRAIN20H2` (신규) | NOISY_REAL 34 | `CMP_REAL` | `CMP_SYN_H2` (신규) | 108/108 |

신규 실체 디렉토리: `TRAIN20H2` (1.5 GB, FFV1), `CMP_SYN_N`, `CMP_SYN_H2` (각 10클립 mpeg4 q6, ~0.7 MB/clip). 기존 디렉토리는 수정하지 않았다.

누설 점검: 평가셋(MCD_A/B 정체성 1~20, REALTEST/RTNC/RTNI = UBFC 42~49, TEST_C3/C6/E3 별도 정체성·시드)과 학습 정체성/시드 겹침 없음(기존 CoTN/CoTNH와 동일 분리). VALID는 기존 yaml과 같이 TRAIN20의 마지막 10%(subject19–20)를 쓰며 `USE_LAST_EPOCH: True`라 모델 선택에 관여하지 않는다 — 단 COTRAIN_NC/NW 계열은 TRAIN20 subject19–20이 학습에 포함되므로 valid 손실은 참고치일 뿐이다(CoTN도 같은 구성).

## 3. `configs/train_h2_proposal.json` 검토 및 TRAIN20H2 생성

검토(train20h.json 대비): `out`=TRAIN20H2, fps 30 / size 256 동일, 20 subject의 **id·frames_dir(sfhq_01/02/03/05/06)·BPM(52–118)·seed(501–520)·spatial/coverage·noise 딕셔너리 전부 동일** — 파형 파라미터(a2, φ2, a3, φ3)만 다르다. a2 최대 0.85 = `make_waveform` 캡 준수(초과 없음). a2 < 0.3 클립 0개(TRAIN20H는 11개). φ2 범위 −0.40~+1.15 rad(TRAIN20H 0–2.8 균등). a3 0.17–0.50 연속(TRAIN20H 0/0.2). φ3 신규(TRAIN20H 없음). 즉 TRAIN20H와 **파형 외 조건이 완전히 통제된 대조**이며, 노이즈 궤적 시드가 같아 노이즈 외우기 측면에서도 동등하다.

생성: `build_synth_dataset.py --config configs/train_h2_proposal.json` (CPU, 약 4분, 로그 `runs/build_TRAIN20H2/stdout.log`). 20/20 성공, 실패 없음. 각 1008프레임 256², FFV1 확인(ffprobe).

### TRAIN20H2 ground_truth 검증 (rfft n×8 피크 BPM; Hilbert 위상 최소제곱 등가 a2/φ2/a3 — 12 보고서와 같은 정의)

| id | 목표BPM | achieved | 요청 a2 | 적합 a2 | 요청 φ2 | 적합 φ2 | 요청 a3 | 적합 a3 | H2/H1 | 피크=f0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 52 | 52.01 | 0.342 | 0.336 | −0.24 | −0.25 | 0.315 | 0.303 | 0.119 | Y |
| 2 | 64 | 64.06 | 0.376 | 0.368 | +0.66 | +0.65 | 0.351 | 0.349 | 0.143 | Y |
| 3 | 76 | 75.89 | 0.399 | 0.397 | −0.04 | −0.05 | 0.320 | 0.313 | 0.161 | Y |
| 4 | 88 | 87.95 | 0.417 | 0.415 | −0.01 | −0.01 | 0.358 | 0.352 | 0.178 | Y |
| 5 | 100 | 100.00 | 0.430 | 0.427 | +0.06 | +0.06 | 0.285 | 0.279 | 0.187 | Y |
| 6 | 112 | 112.05 | 0.448 | 0.445 | −0.17 | −0.17 | 0.256 | 0.251 | 0.202 | Y |
| 7 | 58 | 58.04 | 0.470 | 0.464 | +0.03 | +0.04 | 0.168 | 0.165 | 0.224 | Y |
| 8 | 70 | 70.09 | 0.491 | 0.484 | +0.36 | +0.35 | 0.272 | 0.262 | 0.247 | Y |
| 9 | 82 | 81.92 | 0.513 | 0.505 | +0.35 | +0.34 | 0.459 | 0.456 | 0.266 | Y |
| 10 | 94 | 93.97 | 0.540 | 0.530 | +0.64 | +0.64 | 0.181 | 0.175 | 0.294 | Y |
| 11 | 106 | 106.03 | 0.570 | 0.564 | −0.40 | −0.39 | 0.416 | 0.408 | 0.329 | Y |
| 12 | 118 | 118.08 | 0.602 | 0.592 | +0.56 | +0.55 | 0.212 | 0.205 | 0.368 | Y |
| 13 | 55 | 54.91 | 0.641 | 0.634 | −0.13 | −0.14 | 0.251 | 0.241 | 0.412 | Y |
| 14 | 67 | 66.96 | 0.686 | 0.669 | +0.45 | +0.44 | 0.311 | 0.304 | 0.478 | Y |
| 15 | 79 | 79.02 | 0.746 | 0.735 | −0.26 | −0.25 | 0.424 | 0.407 | 0.566 | Y |
| 16 | 91 | 91.07 | 0.842 | 0.821 | +0.38 | +0.37 | 0.491 | 0.482 | 0.717 | Y |
| 17 | 103 | 102.90 | 0.600 | 0.585 | +1.15 | +1.12 | 0.500 | 0.488 | 0.363 | Y |
| 18 | 115 | 114.96 | 0.700 | 0.686 | +0.43 | +0.43 | 0.232 | 0.220 | 0.496 | Y |
| 19 | 61 | 60.94 | 0.800 | 0.778 | −0.10 | −0.12 | 0.291 | 0.263 | 0.651 | Y |
| 20 | 73 | 72.99 | 0.850 | 0.830 | +0.20 | +0.19 | 0.410 | 0.395 | 0.730 | Y |

요약: achieved BPM 오차 ≤ 0.11 BPM(20/20), 스펙트럼 피크 = 기본파 20/20(a2 = 0.85 포함), 등가 a2 오차 ≤ 0.022(고 a2에서 대역 필터 누설로 약간 과소), φ2 오차 ≤ 0.03 rad, a3 오차 ≤ 0.03. 등가 a2 분포 0.34–0.83, 중앙값 0.55 — 실 a2 p5–p95(0.34–0.84)·중앙값 0.50과 정합. 이 검증은 GT 파형에 대한 것이며 픽셀 수준 무결성(POS lock-in)은 12 보고서 방법을 TRAIN20H2에 재적용해야 한다(미실행).

## 4. yaml 목록 (`_workspace/data/physformer/yaml10/`, 27개)

yaml7의 `finetune_CoTNH.yaml` / `ev_CoTNH_*.yaml`을 복제하고 경로·이름만 치환(diff로 확인: 학습 yaml은 LOG.PATH, MODEL_DIR, MODEL_FILE_NAME, TRAIN.DATA_PATH, TRAIN.EXP_DATA_NAME, OUTPUT_SAVE_DIR 6줄; 평가 yaml은 LOG.PATH, OUTPUT_SAVE_DIR, MODEL_PATH 3줄). 하이퍼파라미터(RESUME 사전학습, batch 4, 10 epoch, LR 5e-5, DiffNormalized, HC crop 128², chunk 160)는 CoTNH와 동일.

| 모델 | 학습 yaml | DATA_PATH | EXP_DATA_NAME (캐시) | 체크포인트 | 평가 yaml (8) |
|---|---|---|---|---|---|
| CoTNC | `finetune_CoTNC.yaml` | COTRAIN_NC | `cotrain_nc` | `runs/finetune_CoTNC/PreTrainedModels/SYNTH_CoTNC_physformer_Epoch9.pth` | `ev_CoTNC_{MCD_A,MCD_B,REALTEST,RTNC,RTNI,C3,C6,E3}.yaml` |
| CoTNW | `finetune_CoTNW.yaml` | COTRAIN_NW | `cotrain_nw` | `…/SYNTH_CoTNW_physformer_Epoch9.pth` | `ev_CoTNW_*.yaml` |
| CoTNH2 | `finetune_CoTNH2.yaml` | COTRAIN_H2 | `cotrain_h2` | `…/SYNTH_CoTNH2_physformer_Epoch9.pth` | `ev_CoTNH2_*.yaml` |

캐시 이름은 기존(`cotrain`, `cotrain_n`, `cotrain_h`, `cotrain_hm`)과 충돌하지 않는다. 평가셋 캐시(`real_MCD_A` 등)는 기존 것을 재사용(`DO_PREPROCESS: False`).

## 5. 실행

    bash /SPECPHYS_ROOT/scripts/run_phase6b.sh

- 작업 디렉토리 `repos/rPPG-Toolbox`에서 `$PY main.py --config_file <yaml>` (16 보고서 재현 절 및 `scripts/run_chain_gated.sh`와 동일 방식, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
- 순서: finetune_CoTNC → finetune_CoTNW → finetune_CoTNH2 → 24개 평가. 단계별 로그 `runs/<name>/stdout.log`, 체인 로그 `runs/phase6b_chain.log`.
- GPU 가드: 매 단계 전 `nvidia-smi` 여유 VRAM 확인, 부족하면 60초 폴링 대기. 평가 게이트 `EVAL_GATE_MIB=8192`(요청 사양). 학습은 과거 실측 batch 4 ≈ 19 GB이므로 `TRAIN_GATE_MIB=19000` 기본(8 GB로 낮추려면 `TRAIN_GATE_MIB=8192 bash …`; OOM 시 1회 재대기·재시도). 
- 재실행 안전: Epoch9 체크포인트가 있으면 학습 생략, pickle 출력이 있으면 평가 생략. `MODELS="CoTNW" SETS="MCD_A MCD_B"` 등으로 부분 실행 가능.
- 현재 GPU: 27.8/32.6 GB 사용 중(타 세션) → 스크립트는 대기 상태로 시작된다. **이번 세션에서는 실행하지 않았다.**

## 6. 예상 소요 (CoTNH 실측 기준: 전처리 108클립 ≈ 3분, 10 epoch ≈ 12분, 평가 8세트 ≈ 1.5분)

| 단계 | 소요 | 비고 |
|---|---|---|
| finetune_CoTNC / CoTNH2 (108클립) | 각 ≈ 15분 | 캐시 신규 ≈ 32 GB/모델 |
| finetune_CoTNW (88클립) | ≈ 13분 | 캐시 ≈ 27 GB |
| 평가 24개 | ≈ 5분 | 캐시 재사용 |
| **합계** | **≈ 50분** (GPU 대기 제외) | 디스크 여유 773 GB → 신규 캐시 ≈ 91 GB 문제없음 |

## 7. 발견 및 주의

1. (§1 발견 1) 기존 `CMP_SYN` 601~610은 TRAIN20H 압축본 → CoTNC용 `CMP_SYN_N`, CoTNH2용 `CMP_SYN_H2`를 신규 생성. 이로써 CoTNC의 파형은 전부 neurokit, CoTNH2의 파형은 전부 TRAIN20H2가 된다.
2. 결과 해석 설계: CoTN→CoTNC = 압축 효과, CoTN→CoTNW = 파형 다양성 효과, CoTNC+CoTNW vs CoTNH = 상호작용, CoTNH→CoTNH2 = 밀도 교정 효과. 단일 시드·소표본(17 보고서 §요약 5)이라 탐색적 결론으로 한정. 결과 분석은 `scripts/analyze_phase6.py`의 모델 목록에 CoTNC/CoTNW/CoTNH2를 추가하면 동일 프로토콜로 계산 가능.
3. TRAIN20H2는 a2 > 0.85(실분포 상위 ~5%)를 캡 때문에 담지 못한다(12 보고서 §6 제안대로 캡 개방은 inject_pulse.py 수정이 필요 — 미수정).
4. 다른 세션 소유 파일(`16_*.md`, `04_*.md`, `inject_pulse.py`, 기존 데이터셋/런 디렉토리)은 건드리지 않았다.

## 생성 파일

- 데이터: `_workspace/data/physformer/{TRAIN20H2, CMP_SYN_N, CMP_SYN_H2, COTRAIN_NC, COTRAIN_NW, COTRAIN_H2}`, 빌드 로그 `runs/build_{TRAIN20H2,CMP_SYN_N,CMP_SYN_H2}/stdout.log`
- yaml: `_workspace/data/physformer/yaml10/*.yaml` (27)
- 스크립트: `scripts/run_phase6b.sh`
- 본 보고서: `_workspace/18_phase6b_prep.md`
