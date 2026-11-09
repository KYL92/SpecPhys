#!/usr/bin/env python3
"""[62 문서 §5, 사전 등록] 모델 구조 팔 vs 같은 시드 NF_SKREF_REP_L. 1단계(선별): python scripts/nsg_judge.py → results/nsg_judge.{txt,json}
2단계(확인): python scripts/nsg_judge.py --confirm ARM S1 S2 ... → results/nsg_confirm_ARM.{txt,json} (Holm 3: P1 잡음 영역 풀, P2 VIPL 전체, P3 MMPD 전체)."""
import glob, json, os, re, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, "/SPECPHYS_ROOT/scripts"); import mmpd_common_gt as _MGC
R = "/SPECPHYS_ROOT/_workspace/data/results"; LINES = []; OUT = {}
def out(s): print(s); LINES.append(s)
def f_of(cell, seed, suf):
    nm = cell if seed == 100 else f"{cell}_s{seed}"; p = f"{R}/agg_{nm}{suf}.json"; return p if os.path.exists(p) else None
def key(r): return (r["subject"], r["light"], r["motion"], r["exercise"], r["skin"])
def vipl(cell, S):
    acc = {}
    for s in S:
        f = f_of(cell, s, "_VIPL")
        if f:
            for r in json.load(open(f)): acc.setdefault(r["vid"], []).append(r)
    return {k: dict(dev=v[0]["gt_lit"], e=float(np.mean([x["err_lit"] for x in v])), sc=v[0].get("scenario")) for k, v in acc.items()}
def mmpd(cell, S):
    acc = {}
    for s in S:
        f = f_of(cell, s, "_MMPD")
        if f:
            for r in _MGC.load(f):
                if 7 <= r["subject"] <= 33: acc.setdefault(key(r), []).append(r)
    return {k: dict(gt=float(np.median([x["gt"] for x in v])), e=float(np.mean([x["err"] for x in v])), mo=v[0]["motion"]) for k, v in acc.items()}
def axis(cell, S, ax, sel=None):
    acc = {}
    for s in S:
        f = f_of(cell, s, "")
        if f:
            d = json.load(open(f))
            for r in (d.get(ax) or []):
                if sel is None or sel(r): acc.setdefault(r["vid"], []).append(r["err"])
    return {k: float(np.mean(v)) for k, v in acc.items()}
def cmp(a, b):
    k = sorted(set(a) & set(b), key=str)
    if len(k) < 5: return None
    x = np.array([a[i] for i in k]); y = np.array([b[i] for i in k]); d = x - y
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(x, y).pvalue)
    return dict(n=len(k), base=float(y.mean()), cell=float(x.mean()), d=float(d.mean()), p=p if np.isfinite(p) else 1.0)
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj
ve = lambda V, c=None: {k: v["e"] for k, v in V.items() if c is None or c(v)}
me = lambda M, c=None: {k: v["e"] for k, v in M.items() if c is None or c(v)}
def bundle(cell, S):
    V, M = vipl(cell, S), mmpd(cell, S); A = {a: axis(cell, S, a) for a in ("NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE", "DLCN", "MCD_A", "MCD_B", "RTNI", "RTNC", "REALTEST")}
    A["PURE"] = axis(cell, S, "PURE", lambda r: int(r["vid"]) >= 701)
    P1 = {**{("V", k): e for k, e in ve(V, lambda x: x["sc"] in ("v2", "v9")).items()}, **{("M",) + k: e for k, e in me(M, lambda x: x["mo"] in (2, 4)).items()}, **{("NS", k): e for k, e in A["NIRP_SMALL"].items()}, **{("NL", k): e for k, e in A["NIRP_LARGE"].items()}}
    tests = {"P1 noise regime pool": P1, "P2 VIPL-HR all": ve(V), "P3 MMPD 7-33 all": me(M),
             "pooled low": {**{("V", k): e for k, e in ve(V, lambda x: x["dev"] < 60).items()}, **{("M",) + k: e for k, e in me(M, lambda x: x["gt"] < 60).items()}},
             "pooled high": {**{("V", k): e for k, e in ve(V, lambda x: x["dev"] >= 120).items()}, **{("M",) + k: e for k, e in me(M, lambda x: x["gt"] >= 110).items()}},
             "VIPL <60": ve(V, lambda x: x["dev"] < 60), "MMPD <110": me(M, lambda x: x["gt"] < 110), "PURE 07-10": A["PURE"], "DLCN": A["DLCN"], "RTNI": A["RTNI"],
             "VIPL v2": ve(V, lambda x: x["sc"] == "v2"), "VIPL v7": ve(V, lambda x: x["sc"] == "v7"), "VIPL v9": ve(V, lambda x: x["sc"] == "v9"), "VIPL >=120": ve(V, lambda x: x["dev"] >= 120), "MMPD >=110": me(M, lambda x: x["gt"] >= 110),
             "MMPD walking": me(M, lambda x: x["mo"] == 4), "MMPD rotation": me(M, lambda x: x["mo"] == 2), "NIRP still": A["NIRP_STILL"], "NIRP small": A["NIRP_SMALL"], "NIRP large": A["NIRP_LARGE"], "RTNC": A["RTNC"], "MCD_A": A["MCD_A"], "MCD_B": A["MCD_B"], "UBFC 42-49": A["REALTEST"]}
    return tests
GUARD = {"pooled low": 0.3, "pooled high": 0.3, "VIPL <60": 0.3, "MMPD <110": 0.3, "PURE 07-10": 0.3, "DLCN": 0.3, "RTNI": 0.5}
PRIMARY = ("P1 noise regime pool", "P2 VIPL-HR all", "P3 MMPD 7-33 all")
BASE = "NF_SKREF_REP_L"
def judge(arm, S):
    cell = f"NF_{arm}_L"; T, B = bundle(cell, S), bundle(BASE, S); res = {}
    for k in T:
        r = cmp(T[k], B[k])
        if r: res[k] = r
    return res
def line(k, r, tail=""): out(f"  {k:24s} n={r['n']:5d} {r['base']:6.2f} -> {r['cell']:6.2f} Δ{r['d']:+6.2f} p={r['p']:.4f}{tail}")
_rng = np.random.default_rng(0)
def flip_p(a, b, n=200000):
    """영상 단위 부호 뒤집기 순열 검정(MAE 차이의 양측 p, 62 §8 2차 사전 등록)."""
    k = sorted(set(a) & set(b), key=str); d = np.array([a[i] - b[i] for i in k]); obs = abs(d.mean()); hit = 0
    for _ in range(n // 10000):
        hit += int(np.sum(np.abs((_rng.choice([-1.0, 1.0], size=(10000, len(d))) * d).mean(1)) >= obs - 1e-12))
    return (hit + 1) / (n + 1)
M_PRIMARY = ("P3 MMPD 7-33 all", "pooled high", "pooled low")
M_GUARD = {"P2 VIPL-HR all": 0.3, "VIPL <60": 0.3, "MMPD <110": 0.3, "PURE 07-10": 0.3, "DLCN": 0.3, "RTNI": 0.5}
if "--screen2" in sys.argv:
    S = (600, 700, 800); ARMS = ("NSP", "NSPA", "NSPF", "NSPL", "NSP2"); cand = []
    out(f"=== 62 §8 재선별: 투영 변형 vs 같은 시드 {BASE}, 시드 {S} (비보정, 채택 주장 금지). 점수 = ΔMMPD 전체 + ΔVIPL 전체")
    for arm in ARMS:
        if not all(f_of(f"NF_{arm}_L", s_, suf) for s_ in S for suf in ("_VIPL", "_MMPD", "")): out(f"--- {arm}: 미완료"); continue
        res = judge(arm, S); OUT[arm] = res; out(f"--- {arm}")
        T_, B_ = bundle(f"NF_{arm}_L", S), bundle(BASE, S)
        for k in ("P2 VIPL-HR all", "P3 MMPD 7-33 all", "pooled low", "pooled high", "P1 noise regime pool", "MMPD >=110", "MMPD walking", "VIPL >=120", "VIPL v2", "VIPL v9", "NIRP still", "NIRP small", "NIRP large", "RTNC"): line(k, res[k], f" perm={flip_p(T_[k], B_[k], 40000):.4f}")
        bad = []
        for k, m in {**M_GUARD, "pooled low": 0.3, "pooled high": 0.3}.items():
            ok = res[k]["d"] <= m; bad += [] if ok else [k]; line("가드 " + k, res[k], f" (≤+{m}) → {'유지' if ok else '위반'}")
        score = res["P3 MMPD 7-33 all"]["d"] + res["P2 VIPL-HR all"]["d"]; out(f"  점수 {score:+.2f} · 가드 위반 {bad or '없음'}"); OUT[arm + "_score"] = score; OUT[arm + "_guard_violations"] = bad
        if not bad: cand.append((score, arm))
    cand.sort(); pick = cand[0][1] if cand else None; out(f"재선별 규칙(가드 유지 중 점수 최소): {[(a, round(x, 2)) for x, a in cand]} → 3단계 {pick}"); OUT["pick"] = pick; name = "nsg_screen2"
elif "--confirm2" in sys.argv:
    i = sys.argv.index("--confirm2"); arm = sys.argv[i + 1]; S = [int(x) for x in sys.argv[i + 2:]]
    cell = f"NF_{arm}_L"; T_, B_ = bundle(cell, S), bundle(BASE, S); res = judge(arm, S); out(f"=== 62 §8 3단계 확인(평균 기준): {cell} vs {BASE}, 시드 {S} (MMPD 공통 정답; 영상 단위 부호 뒤집기 순열 검정, Holm 3)")
    ps = [flip_p(T_[k], B_[k]) for k in M_PRIMARY]; hs = holm(ps); sup = []
    for k, p_, h in zip(M_PRIMARY, ps, hs):
        ok = res[k]["d"] < 0 and h < .05; sup += [k] if ok else []; res[k]["perm"] = p_; res[k]["holm"] = h; line(k, res[k], f" (Wilcoxon) perm={p_:.4f} Holm={h:.4f} → {'지지' if ok else '미지지'}")
    bad = []
    for k, m in M_GUARD.items():
        ok = res[k]["d"] <= m; bad += [] if ok else [k]; line("가드 " + k, res[k], f" perm={flip_p(T_[k], B_[k], 40000):.4f} (≤+{m}) → {'유지' if ok else '위반'}")
    for k in res:
        if k not in M_PRIMARY and k not in M_GUARD: line("(서술) " + k, res[k], f" perm={flip_p(T_[k], B_[k], 40000):.4f}")
    verdict = "구조 이득 확인" if sup and not bad else "채택 안 함"; out(f"판정: 1차 지지 {sup} · 가드 위반 {bad} → {verdict}")
    OUT = dict(arm=arm, seeds=S, results=res, supported=sup, guard_violations=bad, verdict=verdict); name = f"nsg_confirm2_{arm}"
elif "--confirm" in sys.argv:
    i = sys.argv.index("--confirm"); arm = sys.argv[i + 1]; S = [int(x) for x in sys.argv[i + 2:]]
    res = judge(arm, S); out(f"=== 62 §5 2단계 확인: NF_{arm}_L vs {BASE}, 시드 {S} (MMPD 공통 정답)")
    hs = dict(zip(PRIMARY, holm([res[k]["p"] for k in PRIMARY]))); sup = []
    for k in PRIMARY:
        ok = res[k]["d"] < 0 and hs[k] < .05; sup += [k] if ok else []; line(k, res[k], f" Holm={hs[k]:.4f} → {'지지' if ok else '미지지'}")
    bad = []
    for k, m in GUARD.items():
        ok = res[k]["d"] <= m; bad += [] if ok else [k]; line("가드 " + k, res[k], f" (≤+{m}) → {'유지' if ok else '위반'}")
    for k in res:
        if k not in PRIMARY and k not in GUARD: line("(서술) " + k, res[k])
    orig = "구조 이득 확인" if sup and not bad else "채택 안 함"
    out(f"판정: 1차 지지 {sup} · 가드 위반 {bad} → {orig}")
    OUT = dict(arm=arm, seeds=S, results=res, holm=hs, supported=sup, guard_violations=bad, verdict=orig); name = f"nsg_confirm_{arm}"
else:
    S = (600, 700, 800); ARMS = ("NSG", "NSG0", "SGF", "NSP", "SNMF", "ORIGA"); cand = []
    out(f"=== 62 §5 1단계 선별: 구조 팔 vs 같은 시드 {BASE}, 시드 {S} (MMPD 공통 정답; 선별이므로 비보정, 채택 주장 금지)")
    for arm in ARMS:
        have = [s for s in S if f_of(f"NF_{arm}_L", s, "_VIPL") and f_of(f"NF_{arm}_L", s, "_MMPD") and f_of(f"NF_{arm}_L", s, "")]
        if len(have) < len(S): out(f"--- {arm}: 미완료(시드 {have})"); continue
        res = judge(arm, S); OUT[arm] = res; out(f"--- {arm}")
        for k in PRIMARY: line(k, res[k])
        bad = []
        for k, m in GUARD.items():
            ok = res[k]["d"] <= m; bad += [] if ok else [k]; line("가드 " + k, res[k], f" (≤+{m}) → {'유지' if ok else '위반'}")
        for k in res:
            if k not in PRIMARY and k not in GUARD: line("(서술) " + k, res[k])
        if arm != "ORIGA" and not bad and res[PRIMARY[0]]["d"] < 0: cand.append((res[PRIMARY[0]]["d"], res[PRIMARY[0]]["p"], arm))
        OUT[arm + "_guard_violations"] = bad; out(f"  가드 위반: {bad or '없음'}")
    cand.sort(); pick = [c[2] for c in cand[:1]] + [c[2] for c in cand[1:] if c[1] < .05][:1]
    out(f"선별 규칙 결과(가드 유지 + P1 Δ<0 중 P1 Δ 최대, 둘째는 P1 p<.05 일 때만): 후보 {[(a, round(d, 2), round(p, 4)) for d, p, a in cand]} → 2단계 {pick or '없음'}")
    OUT["pick"] = pick; name = "nsg_judge"
open(f"{R}/{name}.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/{name}.json", "w"), indent=1, ensure_ascii=False, default=float)
