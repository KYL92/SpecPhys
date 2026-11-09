#!/usr/bin/env python3
"""[66 문서 §7, 탐색 — 사전 등록 밖] 깨끗한 배경 분리 시험(맥파 마스크 = YCrCb ∧ 얼굴 부위 분할 피부) 재평가 비교. 학습 없음, 기존 체크포인트.
셀: 기준 NF_NSP2DR_L, 게이트 보정 학습·평가 NF_BGREF_L, 평가 때만 NF_NSP2DR_BGT_L, 학습된 빼기 정답 있음 NF_LSUB_L·없음 NF_LSUB0_L(65), 현재 최종 NF_NSP2_L(옛 풀).
지표: 심박 판독 ±6 bpm(청크 비율), 지연 보정 r(모델·시드마다 깨끗한 OFFBAND6 에서 고른 고정 지연). 쌍대 검정은 클립 단위 5시드 평균, 정확 부호 뒤집기(양측, 보정 없음). 옛(배경 누출) 판과 같은 클립끼리 비교도 출력.
  python scripts/sepx_clean_cmp.py → results/sepx_clean_cmp.txt"""
import itertools, json, os
import numpy as np
R = "/SPECPHYS_ROOT/_workspace/data/results"; L = []
def out(s): print(s); L.append(s)
CELLS = {"기준": ("NF_NSP2DR_L", range(2600, 3001, 100)), "게이트 학습·평가": ("NF_BGREF_L", range(2600, 3001, 100)), "게이트 평가만": ("NF_NSP2DR_BGT_L", range(2600, 3001, 100)),
         "빼기 + 정답": ("NF_LSUB_L", range(2600, 3001, 100)), "빼기만": ("NF_LSUB0_L", range(2600, 3001, 100)), "현재 최종(옛 풀)": ("NF_NSP2_L", range(2100, 2501, 100))}
lagc = json.load(open(f"{R}/sepx_lag_c.json")); lago = json.load(open(f"{R}/sepx_lag.json"))
def grp(m): return m["set"].replace("STRESS_", "") + (":" + m["kind"] if m["set"] == "STRESS_SEPX24" else "")
def per_clip(cell, seeds, key, clean=True):
    acc, meta = {}, {}
    for s in seeds:
        if key == "hr_ok":
            p = f"{R}/sepx{'c' if clean else ''}_{cell}_s{s}.json"
            if not os.path.exists(p): continue
            for sub, v in json.load(open(p))["clips"].items(): acc.setdefault(sub, []).append(v["hr_ok"]); meta[sub] = v
        else:
            v = (lagc if clean else lago).get(f"{cell}_s{s}")
            if v is None: continue
            for sub, c in v["clips"].items(): acc.setdefault(sub, []).append(c["rL"]); meta[sub] = c
    return {k: float(np.mean(x)) for k, x in acc.items()}, meta
def exact(dd):
    dd = np.asarray(dd, float); obs = abs(dd.mean()); hit = tot = 0
    for sg in itertools.product((-1.0, 1.0), repeat=len(dd)): tot += 1; hit += abs((np.array(sg) * dd).mean()) >= obs - 1e-12
    return hit / tot
SETS = ["SEPX24:G", "INBAND6", "NEAR6", "TEST_C3", "OFFBAND6", "PMOT6", "SEPX24:L"]
for key, nm in (("hr_ok", "심박 판독 ±6 bpm"), ("rL", "지연 보정 r")):
    out(f"=== 깨끗한 배경: {nm}, 셀별 5시드·클립 평균")
    out("  " + " " * 18 + "  ".join(f"{s:>9s}" for s in SETS))
    for lab, (cell, seeds) in CELLS.items():
        v, meta = per_clip(cell, seeds, key)
        if not v: out(f"  {lab:18s} (자료 없음)"); continue
        out(f"  {lab:18s}" + "  ".join(f"{np.mean([x for s_, x in v.items() if grp(meta[s_]) == st]) if any(grp(meta[s_]) == st for s_ in v) else float('nan'):9.2f}" for st in SETS))
    out(f"--- {nm}: 기준 대비 쌍대(클립 단위, 정확 부호 뒤집기, 보정 없음)")
    B, metaB = per_clip("NF_NSP2DR_L", CELLS["기준"][1], key)
    for lab in ("게이트 학습·평가", "게이트 평가만", "빼기 + 정답", "빼기만"):
        A, _ = per_clip(*CELLS[lab], key)
        cells = []
        for st in ("SEPX24:G", "INBAND6", "OFFBAND6", "PMOT6", "SEPX24:L"):
            subs = sorted(s for s in A if s in B and grp(metaB[s]) == st); dd = [A[s] - B[s] for s in subs]
            cells.append(f"{st} Δ{np.mean(dd):+.2f} (n={len(dd)}, p={exact(dd):.3f})")
        out(f"  {lab:14s} " + " | ".join(cells))
out("=== 배경 누출 판 vs 깨끗한 판(같은 클립, 기준 셀 심박 판독)")
for lab in ("기준", "게이트 평가만", "현재 최종(옛 풀)"):
    A, meta = per_clip(*CELLS[lab], "hr_ok", clean=True); O, _ = per_clip(*CELLS[lab], "hr_ok", clean=False)
    out(f"  {lab:14s} " + "  ".join(f"{st} {np.mean([O[s] for s in O if grp(meta.get(s, {'set':'', 'kind':''})) == st]) if any(grp(meta.get(s, {'set':'', 'kind':''})) == st for s in O) else float('nan'):.2f}→{np.mean([A[s] for s in A if grp(meta[s]) == st]):.2f}" for st in SETS))
open(f"{R}/sepx_clean_cmp.txt", "w").write("\n".join(L) + "\n")
