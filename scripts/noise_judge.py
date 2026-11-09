#!/usr/bin/env python3
"""통합 트랙 LN(31 문서) 판정: 두 목표 축의 쌍대 비교.
  python scripts/noise_judge.py vipl MODEL [REF ...]   VIPL fold held-out: 전체(문헌 GT 주·정본 부)·휴대폰 source2·v8/v9 부분집합, 영상 단위 Wilcoxon vs REF(기본 VF1_TW_L VF1_HYB_L)
  python scripts/noise_judge.py nirp MODEL [REF ...]   MR-NIRP driving STILL/SMALL/LARGE/움직임 합산: 피험자 단위 Wilcoxon vs REF(기본 CoTAF_TW_L VF1_TW_L POS)
                                                     NF{k}_* 모델은 held-out fold k 피험자(nirp_folds.json)로 제한(in-domain 표기)
  python scripts/noise_judge.py nirpcv TAG [REF ...]   NF1..NF5_{TAG} 를 각자 held-out 피험자로 합산(피험자 배타 5-fold in-domain) vs REF(zero-shot) 동일 피험자
  python scripts/noise_judge.py all                    위를 결과 파일이 있는 모든 셀에 대해 → results/noise_judge.json
입력: results/agg_{M}_VIPL.json(agg_vipl.py rows), results/agg_{M}.json(agg_model2 axes), results/agg_POS_NIRP.json. 출력은 표 + json."""
import glob, json, os, re, sys
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"; PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"
FOLDS = {int(k): set(v) for k, v in json.load(open(f"{PFA}/nirp_folds.json")).items()}


def wp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 6 or np.allclose(x, y): return float("nan")
    return float(wilcoxon(x, y).pvalue)


def vipl_rows(m):
    f = f"{R}/agg_{m}_VIPL.json"
    return {r["vid"]: r for r in json.load(open(f))} if os.path.exists(f) else None


def vipl(M, refs):
    a = vipl_rows(M)
    if a is None: print(f"[vipl] {M}: no agg"); return {}
    subsets = {"전체": lambda r: True, "휴대폰 source2": lambda r: r["source"] == 2, "휴대폰 v8/v9": lambda r: r["scenario"] in ("v8", "v9"),
               "휴대폰 v9 움직임": lambda r: r["scenario"] == "v9", "웹캠 source1": lambda r: r["source"] == 1, "RealSense source3": lambda r: r["source"] == 3}
    out = {}
    print(f"=== VIPL held-out (fold {sorted(set(r['fold'] for r in a.values()))}): {M}  [문헌 GT MAE | 정본 GT MAE]")
    for nm, sel in subsets.items():
        ks = [k for k, r in a.items() if sel(r) and r["err_lit"] is not None]
        e_lit = np.array([a[k]["err_lit"] for k in ks]); e = np.array([a[k]["err"] for k in ks])
        line = f"  {nm:16s} n={len(ks):3d}  {M} {e_lit.mean():6.2f} | {e.mean():6.2f}"; out[nm] = dict(n=len(ks), mae_lit=float(e_lit.mean()), mae=float(e.mean()), refs={})
        for ref in refs:
            b = vipl_rows(ref)
            if b is None: continue
            kk = [k for k in ks if k in b]
            x = np.array([b[k]["err_lit"] for k in kk]); y = np.array([a[k]["err_lit"] for k in kk])
            p = wp(x, y); line += f" | vs {ref} {x.mean():6.2f} (Δ{y.mean()-x.mean():+.2f} p={p:.3f})"; out[nm]["refs"][ref] = dict(n=len(kk), ref_mae_lit=float(x.mean()), delta=float(y.mean() - x.mean()), p=p)
        print(line)
    return out


def nirp_axes(m):
    if m == "POS":
        f = f"{R}/agg_POS_NIRP.json"
        if not os.path.exists(f): return None
        d = {}
        for r in json.load(open(f)): d.setdefault(r["axis"], {})[int(r["vid"])] = r["err"]
        return d
    f = f"{R}/agg_{m}.json"
    if not os.path.exists(f): return None
    d = json.load(open(f)); return {ax: {int(r["vid"]): r["err"] for r in d.get(ax, [])} for ax in ("NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE")}


GROUPS = {"정지 STILL": ["NIRP_STILL"], "소동작 SMALL": ["NIRP_SMALL"], "대동작 LARGE": ["NIRP_LARGE"], "움직임 SMALL+LARGE": ["NIRP_SMALL", "NIRP_LARGE"], "전체 3축": ["NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE"]}


def nirp_table(label, A, refs, subj=None, out=None):
    """A = {axis: {subject: err}} (이미 held-out 제한 적용). refs = [(name, axes)]"""
    out = {} if out is None else out
    print(f"=== MR-NIRP driving (피험자 쌍대{', in-domain held-out ' + str(sorted(subj)) if subj else ', zero-shot'}): {label}")
    for g, axes in GROUPS.items():
        ys = [(ax, s, A[ax][s]) for ax in axes for s in sorted(A.get(ax, {})) if (subj is None or s in subj)]
        if not ys: continue
        y = np.array([v for _, _, v in ys]); line = f"  {g:18s} n={len(ys):2d}  {label} {y.mean():6.2f}"; out[g] = dict(n=len(ys), mae=float(y.mean()), refs={})
        for rn, B in refs:
            if B is None: continue
            pair = [(v, B[ax][s]) for ax, s, v in ys if s in B.get(ax, {})]
            if not pair: continue
            yy, xx = np.array(pair).T; p = wp(xx, yy)
            line += f" | vs {rn} {xx.mean():6.2f} (Δ{yy.mean()-xx.mean():+.2f} p={p:.3f})"; out[g]["refs"][rn] = dict(n=len(pair), ref_mae=float(xx.mean()), delta=float(yy.mean() - xx.mean()), p=p)
        print(line)
    return out


def nirp(M, refs):
    A = nirp_axes(M)
    if A is None: print(f"[nirp] {M}: no agg"); return {}
    m = re.match(r"NF(\d)_", M); subj = FOLDS[int(m.group(1))] if m else None
    return nirp_table(M, A, [(r, nirp_axes(r)) for r in refs], subj)


def nirpcv(tag, refs):
    A = {}; subj = set()
    for k in range(1, 6):
        Ak = nirp_axes(f"NF{k}_{tag}")
        if Ak is None: print(f"[nirpcv] NF{k}_{tag} missing"); continue
        for ax, d in Ak.items():
            for s, v in d.items():
                if s in FOLDS[k]: A.setdefault(ax, {})[s] = v; subj.add(s)
    if not A: return {}
    return nirp_table(f"CV_{tag}(in-domain)", A, [(r, nirp_axes(r)) for r in refs], subj)


if __name__ == "__main__":
    mode = sys.argv[1]; args = sys.argv[2:]
    VREF = ["VF1_TW_L", "VF1_HYB_L"]; NREF = ["CoTAF_TW_L", "VF1_TW_L", "POS"]
    if mode == "vipl": vipl(args[0], args[1:] or VREF)
    elif mode == "nirp": nirp(args[0], args[1:] or NREF)
    elif mode == "nirpcv": nirpcv(args[0], args[1:] or NREF)
    elif mode == "all":
        res = {"vipl": {}, "nirp": {}, "nirpcv": {}}
        cells = [os.path.basename(f)[4:-10] for f in sorted(glob.glob(f"{R}/agg_VF1_*_VIPL.json"))]
        for c in cells: res["vipl"][c] = vipl(c, [r for r in VREF if r != c])
        ncells = [os.path.basename(f)[4:-5] for f in sorted(glob.glob(f"{R}/agg_VF1_*.json") + glob.glob(f"{R}/agg_NF?_*.json")) if not f.endswith("_VIPL.json")]
        for c in ncells: res["nirp"][c] = nirp(c, [r for r in NREF if r != c])
        for tag in sorted(set(re.sub(r"^NF\d_", "", c) for c in ncells if c.startswith("NF"))): res["nirpcv"][tag] = nirpcv(tag, NREF)
        json.dump(res, open(f"{R}/noise_judge.json", "w"), indent=1); print(f"→ {R}/noise_judge.json")
    else: raise SystemExit(__doc__)
