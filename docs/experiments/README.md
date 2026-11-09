# Experiment records / 실험 기록

These are the working records of the study, written in Korean during the experiments: design and pre-registration of each study, the results, and the change log. Internal cell names (for example `NF_NSP2_L`) are the identifiers used in `scripts/`. Manuscript drafts are not included.

연구 진행 중에 쓴 실험 기록입니다. 문서마다 설계·사전 등록·결과가 들어 있고, 셀 이름(예: `NF_NSP2_L`)은 `scripts/` 의 식별자와 같습니다. 원고 초안은 넣지 않았습니다.

- `00_changelog_archive.md`, `00_changelog_recent.md` — 실험·코드 변경 이력 (change log)
- `16_physformer_experiment.md` — PhysFormer 단계 실험 수치의 정본 (early PhysFormer stages)
- `47`–`53` — NF-Phys (two-slot FactorizePhys) 설계와 자기지도 분리 신호 (two-slot design, label-free separation)
- `59`–`62` — 스펙트럼 손실의 백본 이식, 심박 범위, 피부 분할·잡음 참조, 잡음 부분공간 투영 = 최종 모델 (spectral transfer, heart-rate coverage, skin parsing, noise-subspace projection = final model)
- `64`–`69` — 잡음 라벨·합성 클립의 기여, 학습된 분리, 입력 배경 비교, 맥파 마스크 수정, 정체성–심박 분리, 기여 근거 (contribution studies)

이름 대응 (names): SpecPhys = `NF_NSP2_L`, SpecPhys-B = `NF_HYB_L`, SpecPhys-L = `NF_SKREF_REP_L`, SpecPhys-H = `NF_SKREF_X15_L`, SpecPhys-R = `NF_ENS_AB_L`.
