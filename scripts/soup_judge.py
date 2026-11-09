#!/usr/bin/env python3
"""[체인 45 / 트랙 B2] 가중치 소프(CoTAM_HYB4_SOUP_L) 판정 — 5시드 풀링·최고 단일 시드·예측 앙상블 대비.
MMPD held-out(7~33) 공통 클립에서 per-clip Wilcoxon, 움직임 분할; 13축은 축별 vid 대응.
출력: results/soup_judge.json
"""
import json, numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"
SEEDS = ("", "_s200", "_s300", "_s400", "_s500")
def key(r): return (r["subject"], r["light"], r["motion"], r["exercise"], r["skin"])
def mm(m): return {key(r): r for r in json.load(open(f"{R}/agg_{m}_MMPD.json"))}
def wp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float("nan") if len(a) < 8 or np.allclose(a, b) else float(wilcoxon(a, b).pvalue)
out = {}
seeds = [mm(f"CoTAM_HYB4_L{s}") for s in SEEDS]
soup = mm("CoTAM_HYB4_SOUP_L")
ks_all = sorted(set.intersection(*[set(d) for d in seeds]))
missing = [k for k in ks_all if k not in soup]
print(f"소프 누락 클립 {len(missing)}: {missing[:10]}")
ks = [k for k in ks_all if k in soup and k[0] >= 7]
MO = {"Stationary": 1, "Rotation": 2, "Talking": 3, "Walking": 4}
mo = np.array([MO.get(k[2], 0) if isinstance(k[2], str) else k[2] for k in ks])
P = np.array([np.mean([d[k]["err"] for d in seeds]) for k in ks])   # 풀링
S = np.array([soup[k]["err"] for k in ks])
single = {s or "_s100": np.array([d[k]["err"] for k in ks]) for s, d in zip(SEEDS, seeds)}
best = min(single, key=lambda s: single[s].mean())
# 앙상블 per-clip (seed_ensemble.json에 있으면)
ens = None
try:
    rows_e = json.load(open(f"{R}/seed_ensemble.json"))["CoTAM_HYB4_L"]["MMPD"]["rows"]
    ens_map = {(r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]): r["err_ens"] for r in rows_e}
    ens = np.array([ens_map.get(k, np.nan) for k in ks])
except Exception as e:
    print("앙상블 per-clip 미사용:", e)
print(f"=== MMPD held-out 공통 n={len(ks)} ===")
rows = {}
for nm, sel in (("전체", mo > 0), ("정지(MO1)", mo == 1), ("움직임(MO2~4)", mo > 1), ("Walking(MO4)", mo == 4)):
    r = dict(n=int(sel.sum()), pooled=float(P[sel].mean()), best_single=best, best=float(single[best][sel].mean()),
             soup=float(S[sel].mean()), p_vs_pooled=wp(P[sel], S[sel]), p_vs_best=wp(single[best][sel], S[sel]))
    if ens is not None and not np.isnan(ens[sel]).any():
        r.update(ens=float(ens[sel].mean()), p_vs_ens=wp(ens[sel], S[sel]))
    rows[nm] = r
    print(f"{nm:14s} n={r['n']:3d} 풀링 {r['pooled']:6.2f} | 최고단일{best} {r['best']:6.2f} | 소프 {r['soup']:6.2f} "
          f"(p vs 풀링 {r['p_vs_pooled']:.4f}, vs 최고단일 {r['p_vs_best']:.4f})" + (f" | 앙상블 {r['ens']:6.2f} (p {r['p_vs_ens']:.4f})" if 'ens' in r else ""))
# 조건부 배치: 정지→TW, 움직임→소프
tw = [mm(f"CoTAF_TW_L{s}") for s in SEEDS]
T = np.array([np.mean([d[k]["err"] for d in tw]) for k in ks])
C_pool = np.where(mo > 1, P, T); C_soup = np.where(mo > 1, S, T)
print(f"조건부 배치(정지→TW): 풀링 {C_pool.mean():.2f} → 소프 {C_soup.mean():.2f} (p={wp(C_pool, C_soup):.4f})")
rows["cond_route"] = dict(pooled=float(C_pool.mean()), soup=float(C_soup.mean()), p=wp(C_pool, C_soup))
# 시드별 vs 소프
print("시드별 전체:", {s: round(float(v.mean()), 2) for s, v in single.items()}, "소프", round(float(S.mean()), 2))
rows["seeds_all"] = {s: float(v.mean()) for s, v in single.items()}
# 13축
print("=== 13축 (풀링 → 소프, vid 대응) ===")
ax_rows = {}
A = [json.load(open(f"{R}/agg_CoTAM_HYB4_L{s}.json")) for s in SEEDS]
Sx = json.load(open(f"{R}/agg_CoTAM_HYB4_SOUP_L.json"))
for ax in ("MCD_B", "MCD_A", "PURE", "REALTEST", "RTNI", "RTNC", "DLCN", "NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE", "C3", "C6", "E3"):
    ds = [{r["vid"]: r["err"] for r in a.get(ax, [])} for a in A]; sd = {r["vid"]: r["err"] for r in Sx.get(ax, [])}
    vs = sorted(set.intersection(*[set(d) for d in ds]) & set(sd))
    if not vs: continue
    p = np.array([np.mean([d[v] for d in ds]) for v in vs]); s = np.array([sd[v] for v in vs])
    ax_rows[ax] = dict(n=len(vs), pooled=float(p.mean()), soup=float(s.mean()), p=wp(p, s))
    print(f"{ax:11s} n={len(vs):3d} {p.mean():6.2f} → {s.mean():6.2f} (p={wp(p, s):.3f})")
out = dict(mmpd=rows, axes=ax_rows, missing_clips=[list(map(str, m)) for m in missing])
json.dump(out, open(f"{R}/soup_judge.json", "w"), indent=1, ensure_ascii=False)
