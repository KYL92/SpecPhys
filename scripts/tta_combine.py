#!/usr/bin/env python3
"""[60 문서 §22] 반속 TTA 결합(사전 고정 규칙): 기본 모델 심박 ≥ T 이고 기본 ≤ 2×반속 ≤ min(1.6×기본, 200) 이면 2×반속 심박, 아니면 기본 심박(정상 심박을 반속하면 판독 대역 하단을 벗어나 고조파를 읽으므로 일관성 조건 필수). VIPL·MMPD 행만 결합, 실 축(agg_{BASE}.json)은 그대로 복사(TTA 미적용).
  python scripts/tta_combine.py OUT BASE SLOW [T=95]"""
import json, shutil, sys
R = "/SPECPHYS_ROOT/_workspace/data/results"
OUT, BASE, SLOW = sys.argv[1:4]; T = float(sys.argv[4]) if len(sys.argv) > 4 else 95.0
def load(f):
    d = json.load(open(f)); return d.get("rows", d) if isinstance(d, dict) else d
for ax in ("VIPL", "MMPD"):
    b = load(f"{R}/agg_{BASE}_{ax}.json"); s = {r["vid"]: r for r in load(f"{R}/agg_{SLOW}_{ax}.json")}; out = []; ng = 0
    for r in b:
        r = dict(r); sr = s.get(r["vid"])
        if sr is not None and r["pred"] >= T and r["pred"] <= 2 * sr["pred"] <= min(1.6 * r["pred"], 200.0):
            r["pred"] = 2 * sr["pred"]; ng += 1
            r["err"] = abs(r["pred"] - r["gt"]); r["ratio"] = r["pred"] / r["gt"] if r.get("gt") else None
            if "gt_lit" in r: r["err_lit"] = abs(r["pred"] - r["gt_lit"])
        out.append(r)
    json.dump(out, open(f"{R}/agg_{OUT}_{ax}.json", "w")); print(f"[tta] {OUT} {ax}: {ng}/{len(b)} videos routed to 2×slow (T={T})")
shutil.copy(f"{R}/agg_{BASE}.json", f"{R}/agg_{OUT}.json"); print(f"[tta] {OUT}: real axes copied from {BASE} (no TTA)")
