#!/usr/bin/env python3
"""[체인 45 / 트랙 B1] 시드 예측 앙상블 — 기존 평가 pickle에서 5시드 예측 파형을 청크 정렬·연결 후
시드별 z-표준화 → 평균 → 정본 프로토콜 HR → per-clip 오차. 재평가 없음.
- "5시드 풀링" = 시드별 오차의 평균 (pool5_judge 정의) vs "앙상블" = 평균 예측의 오차 — 다른 양, 둘 다 보고.
- MMPD held-out(7~33) 움직임 분할 + 13축(pickle이 있는 축). 정본: 실=toolbox 0.6–3.3, 합성=hi-res(agg_model2와 동일).
출력: _workspace/data/results/seed_ensemble.json
사용: python seed_ensemble.py [MODEL ...]  (기본 CoTAM_HYB4_L CoTAF_TW_L)
"""
import glob, json, pickle, re, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from agg_model2 import cat_chunks, toolbox_hr, hires_hr, vidnum, PF, REAL_AXES, SYN_AXES

RES = f"{PF}/../results"
SEEDS = ("", "_s200", "_s300", "_s400", "_s500")
MODELS = sys.argv[1:] or ["CoTAM_HYB4_L", "CoTAF_TW_L"]
MMPD_RE = re.compile(r"subject(\d+)_L(\d)_MO(\d)_E(\d)_S(\d)_GE(\d)_GL(\d)_H(\d)_MA(\d)")

def z(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def wp(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 8 or np.allclose(a, b): return float("nan")
    return float(wilcoxon(a, b).pvalue)

def load_axis(model, ax):
    """ax='MMPD' → B1..6 병합. 반환 {vid: (pred, label)}"""
    pat = f"{PF}/runs/ev_{model}_MMPD_B*/*/saved_test_outputs/*.pickle" if ax == "MMPD" \
        else f"{PF}/runs/ev_{model}_{ax}/*/saved_test_outputs/*.pickle"
    out = {}
    for pk in sorted(glob.glob(pat)):
        d = pickle.load(open(pk, "rb"))
        for vid in d["predictions"]:
            out[str(vid)] = (cat_chunks(d["predictions"][vid]), cat_chunks(d["labels"][vid]))
    return out

def hr_fn(ax):
    return hires_hr if ax in SYN_AXES else toolbox_hr

result = {}
for model in MODELS:
    result[model] = {}
    for ax in ["MMPD"] + REAL_AXES + SYN_AXES:
        per_seed = []
        for s in SEEDS:
            d = load_axis(f"{model}{s}", ax)
            if d: per_seed.append(d)
        if len(per_seed) < 2:
            continue
        vids = sorted(set.intersection(*[set(d) for d in per_seed]))
        f = hr_fn(ax)
        rows = []
        for v in vids:
            lab = per_seed[0][v][1]
            # 라벨은 시드 간 동일해야 함(같은 전처리) — 길이 불일치 시 최소 길이로 절단
            L = min(len(d[v][0]) for d in per_seed)
            preds = np.stack([z(d[v][0][:L]) for d in per_seed])
            ens = preds.mean(0)
            gt = f(lab[:L])
            e_seed = [abs(f(p) - gt) for p in preds]
            e_ens = abs(f(ens) - gt)
            row = dict(vid=v, gt=gt, err_seed=e_seed, err_pool=float(np.mean(e_seed)), err_ens=float(e_ens),
                       pred_ens=float(f(ens)))
            if ax == "MMPD":
                m = MMPD_RE.match(v)
                row.update(subject=int(m.group(1)), light=int(m.group(2)), motion=int(m.group(3)),
                           exercise=int(m.group(4)), skin=int(m.group(5)))
            else:
                row["vidnum"] = vidnum(v)
            rows.append(row)
        result[model][ax] = dict(n_seeds=len(per_seed), rows=rows)
        E = np.array([r["err_seed"] for r in rows])  # (n, seeds)
        pool = E.mean(1); ens = np.array([r["err_ens"] for r in rows])
        best = E.mean(0).argmin()
        line = (f"{model:14s} {ax:10s} n={len(rows):3d} seeds={len(per_seed)} 풀링={pool.mean():6.2f} "
                f"시드별={' '.join(f'{x:.2f}' for x in E.mean(0))} 최고단일={E.mean(0).min():.2f} "
                f"앙상블={ens.mean():6.2f} (vs풀링 p={wp(pool, ens):.4f}, vs최고단일 p={wp(E[:, best], ens):.4f})")
        if ax == "MMPD":
            ho = np.array([r["subject"] >= 7 for r in rows]); mo = np.array([r["motion"] for r in rows])
            line += "\n   held-out(7-33):"
            for nm, sel in (("전체", ho), ("정지", ho & (mo == 1)), ("움직임", ho & (mo > 1)), ("Walking", ho & (mo == 4))):
                line += (f" {nm} 풀링 {pool[sel].mean():.2f}→앙상블 {ens[sel].mean():.2f} (p={wp(pool[sel], ens[sel]):.4f});")
        print(line, flush=True)

json.dump(result, open(f"{RES}/seed_ensemble.json", "w"))
print("ENSEMBLE_DONE")
