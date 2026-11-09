#!/usr/bin/env python3
"""[62 문서 §14, 사전 등록] 2×2 아블레이션(시드 2100~2500; 어텐션 배치 × 투영) + 최종 설정의 MMPD in-domain 축(시드 600~800, 피험자 7~33, 공통 정답).
아블레이션 가족(Holm 2, 부호 뒤집기 순열 검정): 원본 어텐션에서 투영의 효과 C1 풀 고심박, C2 풀 저심박. 서술: 두 슬롯의 효과(투영 있을 때·없을 때). in-domain 1차(Holm 2): I1 MMPD ≥110, I2 MMPD 전체 Δ<0 (zero-shot 같은 시드 대비).
  python scripts/nsp_abl_judge.py → results/nsp_abl.{txt,json}"""
import json, os, sys, numpy as np
src = open("/SPECPHYS_ROOT/scripts/nsg_judge.py").read(); src = src[:src.index('if "--screen2" in sys.argv:')]
g = {}; exec(compile(src, "nj", "exec"), g); bundle, cmp, flip_p, holm, R, key, _MGC = g["bundle"], g["cmp"], g["flip_p"], g["holm"], g["R"], g["key"], g["_MGC"]
LINES = []; OUT = {}
def out(s): print(s); LINES.append(s)
S = [2100, 2200, 2300, 2400, 2500]; CELLS = {"원본 어텐션": "NF_ORIGA_L", "원본 어텐션 + 투영": "NF_NSP2O_L", "두 슬롯": "NF_SKREF_REP_L", "두 슬롯 + 투영(최종)": "NF_NSP2_L"}
KS = ("P2 VIPL-HR all", "P3 MMPD 7-33 all", "pooled low", "pooled high", "VIPL <60", "VIPL >=120", "MMPD <110", "MMPD >=110", "MMPD walking", "NIRP still", "NIRP small", "NIRP large", "RTNI")
B = {n: bundle(c, S) for n, c in CELLS.items()}
out(f"=== 62 §14 2×2 아블레이션, 시드 {S}, MMPD 공통 정답"); out("| 지표 | " + " | ".join(CELLS) + " |")
lv = {}
for k in KS:
    vals = [float(np.mean(list(B[n][k].values()))) for n in CELLS]; lv[k] = dict(zip(CELLS, vals)); out(f"| {k} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
OUT["levels"] = lv
def eff(title, a, b, keys=("pooled high", "pooled low", "P2 VIPL-HR all", "P3 MMPD 7-33 all", "MMPD >=110"), fam=None):
    out(f"--- {title}: {b} → {a}"); res = {}
    for k in keys:
        r = cmp(B[a][k], B[b][k]); r["perm"] = flip_p(B[a][k], B[b][k]); res[k] = r
    if fam:
        hs = holm([res[k]["perm"] for k in fam])
        for k, h in zip(fam, hs): res[k]["holm"] = h
    for k, r in res.items(): out(f"  {k:20s} n={r['n']:5d} {r['base']:6.2f} -> {r['cell']:6.2f} Δ{r['d']:+6.2f} W={r['p']:.4f} perm={r['perm']:.4f}" + (f" Holm={r['holm']:.4f} → {'지지' if r['d'] < 0 and r['holm'] < .05 else '미지지'}" if "holm" in r else ""))
    return res
OUT["C_projection_on_original"] = eff("C 원본 어텐션에서 투영의 효과(1차 Holm 2)", "원본 어텐션 + 투영", "원본 어텐션", fam=("pooled high", "pooled low"))
OUT["projection_on_two_slot"] = eff("두 슬롯에서 투영의 효과(서술)", "두 슬롯 + 투영(최종)", "두 슬롯")
OUT["two_slot_with_projection"] = eff("투영이 있을 때 두 슬롯의 효과(서술)", "두 슬롯 + 투영(최종)", "원본 어텐션 + 투영")
OUT["two_slot_without_projection"] = eff("투영이 없을 때 두 슬롯의 효과(서술)", "두 슬롯", "원본 어텐션")
# [62 §15, 결과 후 추가, 서술] 상호작용 대비 (두 슬롯+투영 − 원본+투영) − (두 슬롯 − 원본), 영상 단위 순열 검정; 사전 등록 아님
out("--- 상호작용 대비(서술, 결과 후 추가): 투영이 있을 때의 두 슬롯 효과 − 없을 때의 두 슬롯 효과"); OUT["interaction"] = {}
for k in ("pooled high", "pooled low", "P2 VIPL-HR all", "P3 MMPD 7-33 all", "MMPD >=110", "VIPL >=120", "VIPL <60"):
    ks = sorted(set(B["두 슬롯 + 투영(최종)"][k]) & set(B["원본 어텐션 + 투영"][k]) & set(B["두 슬롯"][k]) & set(B["원본 어텐션"][k]), key=str)
    x = {i: B["두 슬롯 + 투영(최종)"][k][i] - B["원본 어텐션 + 투영"][k][i] for i in ks}; y = {i: B["두 슬롯"][k][i] - B["원본 어텐션"][k][i] for i in ks}
    r = cmp(x, y); r["perm"] = flip_p(x, y); OUT["interaction"][k] = r; out(f"  {k:20s} n={r['n']:5d} 없을 때 {r['base']:+6.2f} / 있을 때 {r['cell']:+6.2f} 상호작용 {r['d']:+6.2f} W={r['p']:.4f} perm={r['perm']:.4f}")
OUT["orig_proj_vs_final_extra"] = eff("원본 어텐션 + 투영 대비 최종(서술, 다른 축)", "두 슬롯 + 투영(최종)", "원본 어텐션 + 투영", keys=("NIRP still", "NIRP small", "NIRP large", "RTNI", "VIPL <60", "VIPL >=120", "MMPD <110", "MMPD walking"))
OUT["orig_proj_vs_orig_extra"] = eff("원본 어텐션 대비 원본 어텐션 + 투영(서술, 다른 축)", "원본 어텐션 + 투영", "원본 어텐션", keys=("NIRP still", "NIRP small", "NIRP large", "RTNI", "VIPL <60", "VIPL >=120", "MMPD <110", "MMPD walking"))
# in-domain
Sd = [600, 700, 800]
def load(tag, sds):
    acc = {}
    for s in sds:
        f = f"{R}/agg_{tag}_s{s}_MMPD.json"
        if not os.path.exists(f): return None
        for r in _MGC.load(f):
            if 7 <= r["subject"] <= 33: acc.setdefault(key(r), []).append(r)
    return {k: dict(e=float(np.mean([x["err"] for x in v])), gt=float(np.median([x["gt"] for x in v])), mo=v[0]["motion"], n=len(v)) for k, v in acc.items()}
I = load("MCV5_SKN_L", Sd); Z = load("NF_NSP2_L", Sd); RI = load("MCV5_SKC_L", Sd); Pv = load("MCV5_NF_L", []) if False else None
if I:
    sel = {"MMPD >=110": lambda x: x["gt"] >= 110, "MMPD all": lambda x: True, "MMPD <110": lambda x: x["gt"] < 110, "MMPD <60": lambda x: x["gt"] < 60, "walking": lambda x: x["mo"] == 4}
    pick = lambda D, f: {k: v["e"] for k, v in D.items() if f(v)}
    out(f"=== 62 §14 MMPD in-domain, 시드 {Sd}, 피험자 7~33, 공통 정답(zero-shot 축과 별도)"); res = {}
    for k, f in sel.items(): res[k] = cmp(pick(I, f), pick(Z, f)); res[k]["perm"] = flip_p(pick(I, f), pick(Z, f))
    hs = holm([res["MMPD >=110"]["p"], res["MMPD all"]["p"]]); res["MMPD >=110"]["holm"], res["MMPD all"]["holm"] = hs
    for k, r in res.items(): out(f"  zero-shot → in-domain {k:12s} n={r['n']:4d} {r['base']:6.2f} -> {r['cell']:6.2f} Δ{r['d']:+6.2f} W={r['p']:.4f} perm={r['perm']:.4f}" + (f" Holm={r['holm']:.4f}" if "holm" in r else ""))
    OUT["indomain_vs_zero"] = res; res2 = {}
    if RI:
        for k, f in sel.items(): res2[k] = cmp(pick(I, f), pick(RI, f)); out(f"  (서술) in-domain SpecPhys-R → in-domain 최종 {k:12s} {res2[k]['base']:6.2f} -> {res2[k]['cell']:6.2f} Δ{res2[k]['d']:+6.2f} W={res2[k]['p']:.4f}")
        OUT["indomain_vs_routed"] = res2
    for s in Sd:
        d = json.load(open(f"{R}/agg_MCV5_SKN_L_s{s}_MMPD.json")); out(f"  seed {s}: all 660 MAE {np.mean([r['err'] for r in d]):.2f}")
    OUT["all660"] = float(np.mean([np.mean([r["err"] for r in json.load(open(f"{R}/agg_MCV5_SKN_L_s{s}_MMPD.json"))]) for s in Sd]))
open(f"{R}/nsp_abl.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/nsp_abl.json", "w"), indent=1, ensure_ascii=False, default=float)
