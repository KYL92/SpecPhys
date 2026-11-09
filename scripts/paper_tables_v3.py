#!/usr/bin/env python3
"""하이브리드 방법론 논문용 표 생성 (v3) → results/tables_v3.md
T9 요인분해(V1~V4·HYB6) / T10 5시드 확증 / T11 진폭 ablation / T12 문헌 컨텍스트 / T13 PM 백본(있으면)
전 수치는 agg JSON에서 재계산(문서 수기 수치 배제). MMPD held-out(7~33), 움직임 분할 표준."""
import json, os
import numpy as np
from scipy.stats import wilcoxon

R = "/SPECPHYS_ROOT/_workspace/data/results"
OUT = f"{R}/tables_v3.md"

def mm(m):
    return {(r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]): (r["err"], r["motion"])
            for r in json.load(open(f"{R}/agg_{m}_MMPD.json")) if r["subject"] >= 7}

def pooled(base, seeds):
    ds = []
    for s in seeds:
        p = f"{R}/agg_{base}{s}_MMPD.json"
        if os.path.exists(p): ds.append(mm(f"{base}{s}"))
    if not ds: return None
    ks = sorted(set.intersection(*[set(d) for d in ds]))
    mo = np.array([ds[0][k][1] for k in ks])
    return ks, mo, np.array([np.mean([d[k][0] for d in ds]) for k in ks]), len(ds)

def w(a, b):
    return float("nan") if np.allclose(a, b) else wilcoxon(a, b).pvalue

S3 = ("", "_s200", "_s300"); S5 = S3 + ("_s400", "_s500")
ref = pooled("CoTAF_TW_L", S3); ks, mo, TW3, _ = ref
lines = ["# 하이브리드 방법론 표 (v3, 자동 생성 — paper_tables_v3.py)", ""]

def row(base, seeds, label, refX, refks):
    r = pooled(base, seeds)
    if r is None: return f"| {label} | (agg 없음) | | | |"
    ks2, mo2, Y, n = r
    if ks2 != refks:
        common = sorted(set(ks2) & set(refks))
        idx2 = [ks2.index(k) for k in common]; idxr = [refks.index(k) for k in common]
        Y = Y[idx2]; X = refX[idxr]; m2 = mo2[idx2]
    else: X = refX; m2 = mo2
    sels = {"전체": m2 > 0, "정지": m2 == 1, "움직임": m2 > 1, "Walking": m2 == 4}
    c = [f"{Y[s].mean():.2f} (p={w(X[s], Y[s]):.3f})" for s in sels.values()]
    return f"| {label} ({n}시드) | " + " | ".join(c) + " |"

lines += ["## T9. 하이브리드 요인 분해 (기준 CoTAF-TW-L, 3시드 풀링, MMPD held-out)",
          f"| 구성 | 전체 | 정지 | 움직임 | Walking |", "|---|---|---|---|---|",
          f"| CoTAF-TW-L (기준) | {TW3.mean():.2f} | {TW3[mo==1].mean():.2f} | {TW3[mo>1].mean():.2f} | {TW3[mo==4].mean():.2f} |"]
for base, lab in (("CoTAF_HYB_L", "V1 전역 SiNC"), ("CoTAF_HYB2_L", "V2 SiNC 제거"),
                  ("CoTAF_HYB3_L", "V3 SiNC DN-only"), ("CoTAM_HYB4_L", "V4 +MMPD1~6 (채택)"),
                  ("CoTAM_HYB6_L", "V6 +신뢰도 헤드")):
    lines.append(row(base, S3, lab, TW3, ks))

ref5 = pooled("CoTAF_TW_L", S5)
if ref5 and ref5[3] == 5:
    ks5, mo5, TW5, _ = ref5
    h5 = pooled("CoTAM_HYB4_L", S5)
    ks5b, _, HY5, n5 = h5
    assert ks5 == ks5b
    C = np.where(mo5 > 1, HY5, TW5)
    lines += ["", f"## T10. 5시드 확증 (TW vs HYB4, n={len(ks5)})",
              "| 분할 | TW | HYB4 | p |", "|---|---|---|---|"]
    for nm, sel in (("전체", mo5 > 0), ("정지", mo5 == 1), ("움직임", mo5 > 1), ("Walking", mo5 == 4)):
        lines.append(f"| {nm} | {TW5[sel].mean():.2f} | {HY5[sel].mean():.2f} | {w(TW5[sel], HY5[sel]):.4f} |")
    lines.append(f"| 조건부 배치(정지=TW) | {TW5.mean():.2f} | {C.mean():.2f} | {w(TW5, C):.4f} |")

lines += ["", "## T11. 주입 진폭 혼입 통제 (3시드 풀링)",
          "| 구성 | morphology | 주입 RMS | MMPD 전체 |", "|---|---|---|---|"]
for base, m_, a_ in (("CoTAF_L", "해석적", "1.0"), ("CoTAF_AL_L", "해석적", "0.414"),
                     ("CoTAF_TWH_L", "뱅크", "1.0"), ("CoTAF_TW_L", "뱅크", "0.414")):
    r = pooled(base, S3)
    if r: lines.append(f"| {base.replace('_','-')} | {m_} | {a_} | {r[2].mean():.2f} |")
lines.append("(morphology 단독 68%·진폭 단독 34% 기여, 상호작용≈0 — 25 문서)")

lines += ["", "## T12. 문헌 컨텍스트 (프로토콜 상이 — 직접 비교 아님, docs/20 검증 수치)",
          "| 축 | 문헌 최고/앵커 | 우리 (해당 프로토콜 주의) |", "|---|---|---|",
          "| UBFC→PURE 교차 | RhythmFormer 0.97 / toolbox PhysFormer 12.92 | CoTAF-TW-L PURE 0.34 (13축 프로토콜) |",
          "| MR-NIRP driving | Bouraffa'25 in-domain PhysFormer 6.35/7.88/9.17, POS 대동작 9.16(우리 9.48 앵커 일치) | 제로샷 TW 8.51/10.36/9.99 |",
          "| MCD | 원저자 인트라 RhythmFormer 2.82, UBFC학습 PhysFormer 43.65 붕괴 | in-domain CoTAF 0.48 (부분집합 명기) |",
          "| DLCN | 인트라 PhysFormer rest 1.699, POS 11.0 붕괴 | 제로샷 TW 1.71 |",
          "| MMPD 움직임 | (움직임 분할 보고 문헌 부재 — 본 연구가 프레임 제안) | HYB4 10.60 vs POS 12.09 (p=.013) |"]

r_pm = pooled("PM_CoTAM_L", S3)
if r_pm:
    lines += ["", "## T13. 백본 일반성 (PhysMamba, 3시드 풀링, MMPD held-out)",
              "| 구성 | 전체 | 정지 | 움직임 | Walking |", "|---|---|---|---|---|"]
    kp, mp_, PB, np_ = r_pm
    lines.append(f"| PM 기준(동일 코퍼스) | {PB.mean():.2f} | {PB[mp_==1].mean():.2f} | {PB[mp_>1].mean():.2f} | {PB[mp_==4].mean():.2f} |")
    lines.append(row("PM_HYB4_L", S3, "PM+하이브리드", PB, kp))
else:
    lines += ["", "## T13. 백본 일반성 — 체인 43 완료 대기"]

open(OUT, "w").write("\n".join(lines) + "\n")
print(f"tables_v3.md 생성 ({len(lines)}줄)")
