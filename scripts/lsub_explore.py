#!/usr/bin/env python3
"""[65 문서 §6, 탐색 — 사전 등록 밖] 체인 124 서술 비교: 정답 없는 빼기(NF_LSUB0_L) vs 기준(NF_NSP2DR_L) — 모듈만의 효과, 그리고 새 풀 기준 vs 현재 최종 모델(NF_NSP2_L, 옛 풀, 시드 2100~2500, 시드가 달라 짝 없음 → 영상별 평균만).
  python scripts/lsub_explore.py → results/lsub_explore.txt"""
import numpy as np
src = open("/SPECPHYS_ROOT/scripts/lsub_judge.py").read(); src = src[:src.index('OUT["M"] = family(')]
g = {}; exec(compile(src, "lj", "exec"), g)
pfam, bundle, cmp, flip_p, R = g["pfam"], g["bundle"], g["cmp"], g["flip_p"], g["R"]; PRIM, GUARD, DESC = g["PRIM"], g["GUARD"], g["DESC"]
L = []
def out(s): print(s); L.append(s)
out("=== 정답 없는 빼기 NF_LSUB0_L vs 기준 NF_NSP2DR_L, 시드 2600~3000 (Δ = 앞 − 뒤, 보정 없음)")
P = pfam("NF_LSUB0_L", "NF_NSP2DR_L")
for k in PRIM + GUARD + DESC:
    r = P.get(k)
    if r: out(f"  {k:18s} n={r['n']:5d} 기준 {r['base']:6.2f} / 빼기만 {r['cell']:6.2f} Δ{r['d']:+6.2f} perm={r['perm']:.4f} W={r['p']:.4f}")
out("=== 서술: 셀별 영상 평균 오차 (현재 최종 모델은 옛 풀·다른 시드)")
cells = (("현재 최종 NF_NSP2_L", "NF_NSP2_L", [2100, 2200, 2300, 2400, 2500]), ("기준(새 풀)", "NF_NSP2DR_L", g["S"]), ("빼기 + 정답", "NF_LSUB_L", g["S"]), ("빼기만", "NF_LSUB0_L", g["S"]))
B = {nm: bundle(c, s) for nm, c, s in cells}
for k in PRIM + GUARD:
    out(f"  {k:18s} " + "  ".join(f"{nm} {np.mean(list(B[nm][k].values())):6.2f}" for nm, _, _ in cells))
open(f"{R}/lsub_explore.txt", "w").write("\n".join(L) + "\n")
