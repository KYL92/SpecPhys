#!/usr/bin/env python3
"""체인 39(트랙 B): 조건부 라우팅의 상한·정책 분석 — 기존 per-video agg만 사용(무학습).

추정기: TW=CoTAF_TW_L(PhysFormer), PM=PM_CoTAF_TW_L(PhysMamba), POS(고전, MMPD만 보유).
- oracle: 클립별 최소 오차(달성 불가 상한 — 라우팅의 가치 측정)
- axis-policy: 축(도메인) 단위로 최적 추정기 고정(실배치 가능한 가장 단순한 정책)
"""
import json
import numpy as np

R = "/SPECPHYS_ROOT/_workspace/data/results"

def mmpd(m):
    # POS의 vid 표기가 달라 조건 튜플로 조인 (cmp_model.py와 동일 규약)
    return {(r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]): r["err"]
            for r in json.load(open(f"{R}/agg_{m}_MMPD.json")) if r["subject"] >= 7}

def ax(m, a):
    d = json.load(open(f"{R}/agg_{m}.json"))
    return {r["vid"]: r["err"] for r in d.get(a, [])}

print("=== MMPD held-out (540) ===")
tw, pm, pos = mmpd("CoTAF_TW_L"), mmpd("PM_CoTAF_TW_L"), mmpd("POS")
ks = sorted(set(tw) & set(pm) & set(pos))
T = np.array([tw[k] for k in ks]); P = np.array([pm[k] for k in ks]); O = np.array([pos[k] for k in ks])
stack = np.stack([T, P, O])
print(f"n={len(ks)} TW {T.mean():.2f} | PM {P.mean():.2f} | POS {O.mean():.2f} | "
      f"oracle(클립별 최적) {stack.min(0).mean():.2f} | 선택비중 TW/PM/POS = "
      f"{(stack.argmin(0)==0).mean():.2f}/{(stack.argmin(0)==1).mean():.2f}/{(stack.argmin(0)==2).mean():.2f}")

print("=== MR-NIRP (축별) ===")
rows = []
for a in ("NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE"):
    t, p = ax("CoTAF_TW_L", a), ax("PM_CoTAF_TW_L", a)
    ks = sorted(set(t) & set(p))
    T = np.array([t[k] for k in ks]); P = np.array([p[k] for k in ks])
    best = "TW" if T.mean() <= P.mean() else "PM"
    orc = np.minimum(T, P).mean()
    rows.append((a, T.mean(), P.mean(), best, orc))
    print(f"{a:11s} n={len(ks)} TW {T.mean():.2f} | PM {P.mean():.2f} | 축정책={best} | oracle {orc:.2f}")
print("문헌 참조: NIRP 대동작은 POS가 학습모델 전부를 상회(16문서 §13) — 축정책의 LARGE 슬롯은 POS 권장.")

print("=== 축 단위 정책(배치 가능) 요약 ===")
print("MMPD→TW, NIRP still/small→축별 승자, NIRP large→POS, MCD 운동후→PM(0.62~0.88), REALTEST/RTN→PM(0.00)")
print("ROUTING_DONE")
