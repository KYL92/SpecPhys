#!/usr/bin/env python3
"""[62 문서 §10, 사전 등록] 잡음 부분공간 투영 최종 설정 후보 vs 현재 최종 설정 SpecPhys-R(NF_ENS_AB_L, 같은 시드 600~1000).
1차(후보마다 Holm 4, 영상 단위 부호 뒤집기 순열 검정, MAE 차이): VIPL 전체, MMPD 전체, 풀 저심박, 풀 고심박. 권장 규칙: 가장 단순한 후보부터(파서 없는 단일 < 단일 < 라우팅)
네 지표 모두 Δ ≤ +0.3 이고 하나 이상 Holm p<.05 로 낮으면 권장. 서술: Wilcoxon, 하위 대역, 표 6 의 기존 4모델 대비는 final_config_tables 로 별도 생성.
  python scripts/nsp_final_judge.py → results/nsp_final_judge.{txt,json}"""
import json, sys, numpy as np
src = open("/SPECPHYS_ROOT/scripts/nsg_judge.py").read(); src = src[:src.index('if "--screen2" in sys.argv:')]
g = {}; exec(compile(src, "nj", "exec"), g); bundle, cmp, flip_p, holm, f_of, R = g["bundle"], g["cmp"], g["flip_p"], g["holm"], g["f_of"], g["R"]
S = [600, 700, 800, 900, 1000]; LINES = []; OUT = {}
def out(s): print(s); LINES.append(s)
PRIM = ("P2 VIPL-HR all", "P3 MMPD 7-33 all", "pooled low", "pooled high")
DESC = ("VIPL <60", "VIPL v2", "VIPL v7", "VIPL v9", "VIPL >=120", "MMPD <110", "MMPD >=110", "MMPD walking", "MMPD rotation", "NIRP still", "NIRP small", "NIRP large", "DLCN", "PURE 07-10", "RTNI", "RTNC", "MCD_A", "MCD_B", "UBFC 42-49")
BASE = "NF_ENS_AB_L"; TB = bundle(BASE, S)
CANDS = (("파서 없는 단일 모델", "NF_NSP2P_L"), ("단일 모델", "NF_NSP2_L"), ("라우팅(단일 + 1.5배속 구성원)", "NF_ENS_N_L"))
out(f"=== 62 §10 최종 설정 후보 vs SpecPhys-R({BASE}), 시드 {S}; MMPD 공통 정답; 1차 = 부호 뒤집기 순열 검정 Holm 4"); rec = None
for nm, cell in CANDS:
    if not all(f_of(cell, s, suf) for s in S for suf in ("_VIPL", "_MMPD", "")): out(f"--- {nm} {cell}: 미완료"); continue
    T = bundle(cell, S); ps = [flip_p(T[k], TB[k]) for k in PRIM]; hs = holm(ps); res = {}; ok_all = True; better = []
    out(f"--- {nm} {cell}")
    for k, p, h in zip(PRIM, ps, hs):
        r = cmp(T[k], TB[k]); r.update(perm=p, holm=h); res[k] = r; ok_all &= r["d"] <= 0.3; better += [k] if (r["d"] < 0 and h < .05) else []
        out(f"  {k:20s} n={r['n']:5d} {r['base']:6.2f} -> {r['cell']:6.2f} Δ{r['d']:+6.2f} W={r['p']:.4f} perm={p:.4f} Holm={h:.4f}")
    for k in DESC:
        r = cmp(T[k], TB[k]); res[k] = r; out(f"  (서술) {k:15s} n={r['n']:5d} {r['base']:6.2f} -> {r['cell']:6.2f} Δ{r['d']:+6.2f} W={r['p']:.4f}")
    verdict = ok_all and bool(better); out(f"  네 지표 Δ≤+0.3: {ok_all} · Holm 유의 개선: {better} → 권장 조건 {'충족' if verdict else '미충족'}"); OUT[cell] = dict(results=res, ok_all=ok_all, better=better, meets=verdict)
    if verdict and rec is None: rec = cell
out(f"권장 규칙 결과(가장 단순한 충족 후보): {rec}"); OUT["recommended"] = rec
open(f"{R}/nsp_final_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/nsp_final_judge.json", "w"), indent=1, ensure_ascii=False, default=float)
