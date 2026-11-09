#!/usr/bin/env python3
"""체인 38c: LODO 뱅크 확장 최종 요약 — 폴드별 CoTAF_LO{F}_L(확장 뱅크, 대상 GT 제외) vs CoTAF_TW_L(뱅크 v1).

공정성: DLCN/MMPD/NIRP/MCD 폴드는 양쪽 모두 대상 셋 GT를 뱅크·학습에 안 씀 → 대칭.
PURE 폴드는 TW의 v1 뱅크에 PURE GT가 포함(누수 이점) → 비대칭이므로 보조로 CoTAF_L(뱅크 무) 대비도 병기.
"""
import json
import numpy as np
from scipy.stats import wilcoxon

R = "/SPECPHYS_ROOT/_workspace/data/results"
FOLD_AXES = {"DLCN": ["DLCN"], "MMPD": None, "NIRP": ["NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE"],
             "MCD": ["MCD_A", "MCD_B"], "PURE": ["PURE"]}

def load_ax(model, ax):
    d = json.load(open(f"{R}/agg_{model}.json"))
    return {r["vid"]: r["err"] for r in d.get(ax, [])}

def load_mmpd(model, sub_min=1):
    d = json.load(open(f"{R}/agg_{model}_MMPD.json"))
    return {r["vid"]: r["err"] for r in d if r["subject"] >= sub_min}

def pair(ka, kb):
    ks = sorted(set(ka) & set(kb))
    return np.array([ka[k] for k in ks]), np.array([kb[k] for k in ks])

def wp(x, y):
    if len(x) < 8 or np.allclose(x, y): return float("nan")
    return float(wilcoxon(x, y).pvalue)

pool = []
print("=== LODO 요약: 기준(TW, 뱅크 v1) -> 확장 뱅크(LO — 대상 데이터셋 GT 제외) ===")
for fold, axes in FOLD_AXES.items():
    M = f"CoTAF_LO{fold}_L"
    try:
        if fold == "MMPD":
            x, y = pair(load_mmpd("CoTAF_TW_L"), load_mmpd(M))
            xh, yh = pair(load_mmpd("CoTAF_TW_L", 7), load_mmpd(M, 7))
            print(f"[{fold}] n={len(x)} TW {x.mean():.2f} -> LO {y.mean():.2f} p={wp(x, y):.4f} "
                  f"(held-out 7~33 n={len(xh)}: {xh.mean():.2f}->{yh.mean():.2f} p={wp(xh, yh):.4f})")
        else:
            xs, ys, parts = [], [], []
            for ax in axes:
                x, y = pair(load_ax("CoTAF_TW_L", ax), load_ax(M, ax))
                if not len(x): parts.append(f"{ax}:없음"); continue
                parts.append(f"{ax} n={len(x)} {x.mean():.2f}->{y.mean():.2f} p={wp(x, y):.3f}")
                xs.append(x); ys.append(y)
            x = np.concatenate(xs); y = np.concatenate(ys)
            line = f"[{fold}] n={len(x)} TW {x.mean():.2f} -> LO {y.mean():.2f} p={wp(x, y):.4f} | " + "  ".join(parts)
            if fold == "PURE":
                xb, yb = pair(load_ax("CoTAF_L", "PURE"), load_ax(M, "PURE"))
                line += f" | (비대칭 주의: TW 뱅크에 PURE 포함 — 보조: CoTAF_L {xb.mean():.2f}->LO {yb.mean():.2f} p={wp(xb, yb):.3f})"
            print(line)
        pool.append((x, y))
    except FileNotFoundError as e:
        print(f"[{fold}] agg 없음: {e}")
if pool:
    X = np.concatenate([p[0] for p in pool]); Y = np.concatenate([p[1] for p in pool])
    d = Y - X
    print(f"=== 풀링(전 폴드 per-clip) n={len(X)} TW {X.mean():.3f} -> LO {Y.mean():.3f} p={wp(X, Y):.4f} "
          f"| 개선 {int((d < 0).sum())} 악화 {int((d > 0).sum())} ===")
print("LODO_SUMMARY_DONE")
