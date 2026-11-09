#!/usr/bin/env python3
"""[64 문서, 사전 등록] 최종 모델에서 잡음 라벨(NF_NSP2NL_L)과 합성 클립(NF_NSP2NS_L)의 기여. 시드 2100~2500, MMPD 공통 정답, 영상 단위 부호 뒤집기 순열 검정(양측), 가족마다 Holm 5. Δ = 최종 모델 − 뺀 셀.
  python scripts/nsp_contrib_judge.py → results/nsp_contrib.{txt,json}"""
import json, numpy as np
src = open("/SPECPHYS_ROOT/scripts/nsg_judge.py").read(); src = src[:src.index('if "--screen2" in sys.argv:')]
g = {}; exec(compile(src, "nj", "exec"), g); bundle, cmp, flip_p, holm, R = g["bundle"], g["cmp"], g["flip_p"], g["holm"], g["R"]
S = [2100, 2200, 2300, 2400, 2500]; LINES = []; OUT = {}
def out(s): print(s); LINES.append(s)
T = bundle("NF_NSP2_L", S)
PRIM = ("P2 VIPL-HR all", "P3 MMPD 7-33 all", "pooled low", "pooled high", "NIRP small")
DESC = ("VIPL <60", "VIPL >=120", "VIPL v2", "VIPL v7", "VIPL v9", "MMPD <110", "MMPD >=110", "MMPD walking", "MMPD rotation", "NIRP still", "NIRP large", "DLCN", "PURE 07-10", "RTNI", "RTNC", "MCD_A", "MCD_B", "UBFC 42-49")
def fam(title, ref):
    B = bundle(ref, S); out(f"=== {title}: 최종 NF_NSP2_L vs {ref}, 시드 {S} (Δ = 최종 − 뺀 셀)"); ps = [flip_p(T[k], B[k]) for k in PRIM]; hs = holm(ps); res = {}; good = []; bad = []
    for k, p, h in zip(PRIM, ps, hs):
        r = cmp(T[k], B[k]); r.update(perm=p, holm=h); res[k] = r; good += [k] if (r["d"] < 0 and h < .05) else []; bad += [k] if (r["d"] > 0 and h < .05) else []
        out(f"  {k:20s} n={r['n']:5d} 뺀 셀 {r['base']:6.2f} / 최종 {r['cell']:6.2f} Δ{r['d']:+6.2f} Wilcoxon={r['p']:.4f} perm={p:.4f} Holm={h:.4f}")
    for k in DESC:
        r = cmp(T[k], B[k]); r["perm"] = flip_p(T[k], B[k]); res[k] = r; out(f"  (서술) {k:13s} n={r['n']:5d} 뺀 셀 {r['base']:6.2f} / 최종 {r['cell']:6.2f} Δ{r['d']:+6.2f} W={r['p']:.4f} perm={r['perm']:.4f}")
    v = "기여 확인" if good and not bad else "역효과" if bad and not good else "혼재" if good and bad else "측정되지 않음"; out(f"  판정: {v} (최종이 유의하게 낮음 {good}, 유의하게 높음 {bad})"); return dict(results=res, better=good, worse=bad, verdict=v)
OUT["noise_label"] = fam("가족 N 잡음 라벨", "NF_NSP2NL_L"); OUT["synthetic_clips"] = fam("가족 S 합성 클립", "NF_NSP2NS_L")
open(f"{R}/nsp_contrib.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/nsp_contrib.json", "w"), indent=1, ensure_ascii=False, default=float)
