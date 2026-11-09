# 가상얼굴 합성 보고서 — 실사(photoreal) 소스 확보 및 영상화

실행: 2026-08-21 · face-synth 단계 (요청: "실제 얼굴처럼 보이는 가상얼굴")

## 배경 — 첫 데모의 문제

첫 데모에 쓴 `s0.jpg`는 LivePortrait 동봉 예제 중 **유화 초상화**였다. LivePortrait은 얼굴 생성기가 아니라 **애니메이터**이므로 사실성은 소스에서 나온다. → 실사 가상얼굴 소스를 별도 확보하여 해결.

## 확보한 실사 가상얼굴 (모두 실존하지 않는 정체성)

| 계열 | 소스 | 확보 방법 | 수량 | 특성 |
|------|------|----------|------|------|
| **GAN (SFHQ)** | `pravsels/SFHQ` (HF 미러, MIT) | `hf download` 샤드 1개(750MB) | 8장 (1024²) | StyleGAN2 큐레이션, FFHQ급 실사, 다양한 연령/인종 |
| **디퓨전 (SDXL-Turbo)** | `stabilityai/sdxl-turbo` | RTX 5090 로컬 생성, 시드 42 고정 | 3장 (768²) | 프롬프트로 속성 제어(연령·성별·조명), 장당 ~1초 |

- 갤러리: `_workspace/data/faces/photoreal/gallery.png` (SFHQ 6 + SDXL 3)
- 파일: `_workspace/data/faces/photoreal/sfhq_01..08.jpg`, `diff_{female_01,male_01,female_02}.png`
- 시도 후 배제: thispersondoesnotexist.com (봇 차단), Kaggle 원본(계정 필요 — HF 미러로 대체)

## 실사 얼굴 영상화 (LivePortrait)

| 출력 | 소스 | 구동 | 스펙 |
|------|------|------|------|
| `animations/sfhq_04--d6.mp4` | SFHQ 실사 남성 | d6.mp4 | 1008프레임 @30fps (33.6s) |
| `animations/diff_female_01--d6.mp4` | SDXL 실사 여성 | d6.mp4 | 1008프레임 @30fps (33.6s) |

프레임 스트립(`*_strip.png`)으로 시간 일관성 확인 — 표정·시선이 자연스럽게 변화, 정체성 유지.

## rPPG 실험용 권고

- **소스**: SFHQ를 기본으로(라이선스 최청정, 대량 정체성). 속성 통제 조건이 필요하면 SDXL-Turbo 보충.
- **구동 영상**: d6는 표정 변화(웃음/말하기)가 커서 데모용으로는 좋지만, **주입 실험의 clean plate로는 저모션 구동**(눈 깜빡임 위주)이 유리 — 움직임 잡음이 POS/CHROM의 잔여 오차를 키운다.
- 재현: 소스/구동/시드가 모두 고정 파일이므로 `inference.py -s <소스> -d <구동>`으로 동일 재생성.

## 재현 절차

```bash
# SFHQ 실사 얼굴 (HF 미러)
conda run -n vface-gen hf download pravsels/SFHQ shard_10_of_86.zip --repo-type dataset --local-dir <dir>
# SDXL-Turbo 생성 (시드 고정)
# diffusers AutoPipelineForText2Image, "stabilityai/sdxl-turbo", 4-step, seed 42
# 영상화
conda run -n vface-gen python repos/LivePortrait/inference.py \
  -s _workspace/data/faces/photoreal/sfhq_04.jpg \
  -d repos/LivePortrait/assets/examples/driving/d6.mp4 \
  -o _workspace/data/faces/animations
```
