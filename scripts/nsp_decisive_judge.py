#!/usr/bin/env python3
"""[62 문서 §12, 사전 등록] 결정 실험: 새 시드 2100~2500, MMPD 공통 정답, 영상 단위 부호 뒤집기 순열 검정(MAE 차이, 양측).
가족 A(Holm 2, 같은 학습의 원본 FactorizePhys 어텐션 대비 모델 구조의 이점): A1 풀 고심박, A2 풀 저심박. 가드: VIPL 전체·MMPD 전체 Δ ≤ +0.3.
가족 B(Holm 2, 현재 최종 설정 SpecPhys-R 대비 교체): B1 MMPD 전체, B2 풀 저심박. 가드: VIPL 전체·풀 고심박 Δ ≤ +0.3.
  python scripts/nsp_decisive_judge.py → results/nsp_decisive.{txt,json}"""
import json, sys, numpy as np
src = open("/SPECPHYS_ROOT/scripts/nsg_judge.py").read(); src = src[:src.index('if "--screen2" in sys.argv:')]
g = {}; exec(compile(src, "nj", "exec"), g); bundle, cmp, flip_p, holm, R = g["bundle"], g["cmp"], g["flip_p"], g["holm"], g["R"]
S = [2100, 2200, 2300, 2400, 2500]; LINES = []; OUT = {}
def out(s): print(s); LINES.append(s)
T = bundle("NF_NSP2_L", S)
DESC = ("VIPL <60", "VIPL >=120", "VIPL v2", "VIPL v7", "VIPL v9", "MMPD <110", "MMPD >=110", "MMPD walking", "MMPD rotation", "NIRP still", "NIRP small", "NIRP large", "DLCN", "PURE 07-10", "RTNI", "RTNC", "MCD_A", "MCD_B", "UBFC 42-49")
def fam(title, ref, prim, guards):
    B = bundle(ref, S); out(f"=== {title}: NF_NSP2_L vs {ref}, 시드 {S}"); ps = [flip_p(T[k], B[k]) for k in prim]; hs = holm(ps); res = {}; sup = []; bad = []
    for k, p, h in zip(prim, ps, hs):
        r = cmp(T[k], B[k]); r.update(perm=p, holm=h); res[k] = r; ok = r["d"] < 0 and h < .05; sup += [k] if ok else []
        out(f"  {k:20s} n={r['n']:5d} {r['base']:6.2f} -> {r['cell']:6.2f} Δ{r['d']:+6.2f} Wilcoxon={r['p']:.4f} perm={p:.4f} Holm={h:.4f} → {'지지' if ok else '미지지'}")
    for k in guards:
        r = cmp(T[k], B[k]); res[k] = r; ok = r["d"] <= 0.3; bad += [] if ok else [k]; out(f"  가드 {k:15s} n={r['n']:5d} {r['base']:6.2f} -> {r['cell']:6.2f} Δ{r['d']:+6.2f} (≤+0.3) → {'유지' if ok else '위반'}")
    for k in DESC:
        r = cmp(T[k], B[k]); res[k] = r; out(f"  (서술) {k:13s} n={r['n']:5d} {r['base']:6.2f} -> {r['cell']:6.2f} Δ{r['d']:+6.2f} W={r['p']:.4f}")
    v = bool(sup) and not bad; out(f"  판정: 지지 {sup} · 가드 위반 {bad} → {'확인' if v else '미확인'}"); return dict(results=res, supported=sup, guard_violations=bad, confirmed=v)
OUT["A_vs_original_attention"] = fam("가족 A 모델 구조의 이점", "NF_ORIGA_L", ("pooled high", "pooled low"), ("P2 VIPL-HR all", "P3 MMPD 7-33 all"))
OUT["B_vs_current_final"] = fam("가족 B 현재 최종 설정 교체", "NF_ENS_AB_L", ("P3 MMPD 7-33 all", "pooled low"), ("P2 VIPL-HR all", "pooled high"))
open(f"{R}/nsp_decisive.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/nsp_decisive.json", "w"), indent=1, ensure_ascii=False, default=float)
