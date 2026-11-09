#!/usr/bin/env python3
"""[65 문서 §6, 탐색 — 사전 등록 밖] sepx_lag.json(모델별 고정 지연 L*)으로 분리 시험 r 을 지연 보정 전(r0)·후(rL) 갈래끼리 비교. 클립별 5시드 평균, 정확 부호 뒤집기 순열(양측, 보정 없음).
  python scripts/sepx_lag_cmp.py → results/sepx_lag_cmp.txt"""
import itertools, json
import numpy as np
R = "/SPECPHYS_ROOT/_workspace/data/results"; d = json.load(open(f"{R}/sepx_lag.json")); L = []
def out(s): print(s); L.append(s)
meta = {s: c for v in d.values() for s, c in v["clips"].items()}
def arm_clip(arm, key):
    acc = {}
    for v in d.values():
        if v["arm"] == arm:
            for s, c in v["clips"].items(): acc.setdefault(s, []).append(c[key])
    return {s: float(np.mean(x)) for s, x in acc.items()}
def exact(dd):
    dd = np.asarray(dd, float); obs = abs(dd.mean()); hit = tot = 0
    for sg in itertools.product((-1.0, 1.0), repeat=len(dd)): tot += 1; hit += abs((np.array(sg) * dd).mean()) >= obs - 1e-12
    return hit / tot
out("셀별 고정 지연 L*(프레임, 양수 = 출력이 정답보다 늦음): " + ", ".join(f"{k}={v['lag']}" for k, v in d.items()))
for key, nm in (("r0", "지연 보정 전"), ("rL", "지연 보정 후")):
    out(f"=== {nm} ({key})")
    for a, b in (("LS", "F"), ("L0", "F"), ("LS", "L0")):
        A, B = arm_clip(a, key), arm_clip(b, key)
        for st, kind, lab in (("STRESS_SEPX24", "G", "새 시험 전역 플리커 근접"), ("STRESS_SEPX24", "L", "새 시험 피부 국소 플리커 근접"), ("STRESS_OFFBAND6", "G", "기존 플리커 먼 거리"), ("TEST_C3", "G", "기존 TEST_C3")):
            subs = sorted(s for s in A if s in B and meta[s]["set"] == st and meta[s]["kind"] == kind); dd = [A[s] - B[s] for s in subs]
            out(f"  {a} vs {b} {lab:18s} n={len(subs):2d} {b} {np.mean([B[s] for s in subs]):.3f} / {a} {np.mean([A[s] for s in subs]):.3f} Δ{np.mean(dd):+.3f} 정확 순열 p={exact(dd):.4f}")
open(f"{R}/sepx_lag_cmp.txt", "w").write("\n".join(L) + "\n")
