# 21단계 — 뱅크 확장 LODO-CV (체인 38, 2026-09-02 사전 등록, 사용자 지시)

질문: TW2(외부 PPG 뱅크)의 실패가 "모달리티 격차" 때문이라면, **얼굴 rPPG 데이터셋들의 접촉 GT**로 뱅크를
확장하면 이득이 있는가? 평가는 데이터셋 단위 Leave-One-Dataset-Out — 대상 데이터셋의 GT는 해당 폴드 뱅크에서
완전 제외(누수 0), 대상 축 성능만 그 폴드의 지표.

## 구성
- 대상 5 데이터셋: DLCN_face·MMPD·MR-NIRP·MCD-rPPG·PURE.
- 소스 세그먼트(v1 게이트 동일): ubfc 62(상수) + pure 69 + mmpd 120 + mcd 48 + dlcn 40 + nirp 57.
  MMPD/PURE/MCD는 지정 학습분할만; DLCN·NIRP는 전량이 평가셋이므로 "타 폴드의 소스"로만 쓰임(해당 폴드에선 제외).
- 폴드 뱅크: bank_lodo_<F> = 전체 − <F> (276~356세그). MMPD 소스가 운동 조건 실 고심박(p90 157 BPM) 공급.
- 주입: TW config 그대로(idx 정책 동일, nearest 미사용 — TW2 용의자 ② 배제), 42클립 재주입 → 실 161 + 42 = 203.
- 모델: CoTAF_LO<F>_L, 시드 100, 10ep. 평가: 폴드 대상 축만(MMPD는 held-out 7~33).
- 비교: CoTAF_L(해석적) / CoTAF_TW_L(v1 뱅크 = ubfc+pure) / LODO — per-clip Wilcoxon.
  각주: PURE 폴드에서 TW는 뱅크에 PURE-GT 포함(비-LODO), LODO만 무누수. UBFC는 대상 아님(전 폴드 공통).

## 해석 지침
- LODO > TW: 뱅크 확장(정합 모달리티)이 유효 — TW2 실패가 소스 격차 때문임이 확정.
- LODO ≈ TW: 뱅크 규모는 포화(131이면 충분), 파형 사전의 이득은 "실 형태" 자체에서 옴.
- LODO < TW: 타 데이터셋 GT 혼입도 희석 — v1 조합(ubfc+pure)이 특이적으로 좋은 이유 추가 분석 필요.
코드: scripts/{build_ppg_bank_lodo,lodo_summary}.py, run_chain38_lodo.sh (/tmp/chain38.log)
