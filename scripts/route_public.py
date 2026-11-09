#!/usr/bin/env python3
"""[60 문서 §23(a)] 영상 단위 라우팅: 우리 셀 OURS 의 심박 ≥ T1 이고 공개 모델 심박 ≥ T2 이고 공개 ≤ R×우리 이면 공개 모델 심박, 아니면 우리 심박.
  python scripts/route_public.py OUT OURS PUB_VIPL PUB_AXES [T1=95 T2=110 R=1.6]   (시드 접미사 _s200… 는 OURS 의 각 시드에 대해 자동 처리)"""
import glob, json, os, re, sys
R = "/SPECPHYS_ROOT/_workspace/data/results"
OUT, OURS, PV, PA = sys.argv[1:5]; T1, T2, RR = (float(x) for x in (sys.argv[5:8] if len(sys.argv) >= 8 else (95, 110, 1.6)))
def load(f):
    d = json.load(open(f)); return d.get("rows", d) if isinstance(d, dict) and "rows" in d else d
def rule(o, p): return p is not None and o >= T1 and p >= T2 and p <= RR * o
def fix(r, pred, pr_h=None):
    r = dict(r); r["pred"] = pred; r["err"] = abs(pred - r["gt"]); r["ratio"] = pred / r["gt"] if r.get("gt") else None
    if "gt_lit" in r: r["err_lit"] = abs(pred - r["gt_lit"])
    if pr_h is not None and "gt_h" in r: r["pred_h"] = pr_h; r["err_h"] = abs(pr_h - r["gt_h"])
    return r
pv = {r["vid"]: r for r in load(f"{R}/agg_{PV}_VIPL.json")}
pm = {(r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]): r for r in load(f"{R}/agg_{PA}_MMPD.json")}
pa = json.load(open(f"{R}/agg_{PA}.json"))
seeds = [""] + sorted(re.sub(rf"^agg_{re.escape(OURS)}(_s\d+)_VIPL\.json$", r"\1", os.path.basename(f)) for f in glob.glob(f"{R}/agg_{OURS}_s[0-9]*_VIPL.json") if re.match(rf"^agg_{re.escape(OURS)}_s\d+_VIPL\.json$", os.path.basename(f)))
for s in seeds:
    st = {}
    v = []; n = 0
    for r in load(f"{R}/agg_{OURS}{s}_VIPL.json"):
        p = pv.get(r["vid"]); ok = rule(r["pred"], p["pred"] if p else None); n += ok; v.append(fix(r, p["pred"]) if ok else r)
    json.dump(v, open(f"{R}/agg_{OUT}{s}_VIPL.json", "w")); st["VIPL"] = f"{n}/{len(v)}"
    m = []; n = 0
    for r in load(f"{R}/agg_{OURS}{s}_MMPD.json"):
        p = pm.get((r["subject"], r["light"], r["motion"], r["exercise"], r["skin"])); ok = rule(r["pred"], p["pred"] if p else None); n += ok; m.append(fix(r, p["pred"]) if ok else r)
    json.dump(m, open(f"{R}/agg_{OUT}{s}_MMPD.json", "w")); st["MMPD"] = f"{n}/{len(m)}"
    ax = json.load(open(f"{R}/agg_{OURS}{s}.json")); outax = {}
    for a, rows in ax.items():
        if not isinstance(rows, list): outax[a] = rows; continue
        pp = {r["vid"]: r for r in (pa.get(a) or [])}; n = 0; new = []
        for r in rows:
            p = pp.get(r["vid"]); ok = rule(r["pred"], p["pred"] if p else None); n += ok; new.append(fix(r, p["pred"], p.get("pred_h")) if ok else r)
        outax[a] = new; st[a] = f"{n}/{len(rows)}"
    json.dump(outax, open(f"{R}/agg_{OUT}{s}.json", "w"))
    print(f"[route] {OUT}{s}: routed to public {st}")
