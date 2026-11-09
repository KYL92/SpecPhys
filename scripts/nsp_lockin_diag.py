#!/usr/bin/env python3
"""[62 문서 §5 기전 점검, 학습 없음] 잡음 부분공간 투영이 실패 유형을 어떻게 바꾸는가: 53 문서의 실패 분류(selfsup_phase0_diag.classify, 영상별 잡음 피크 캐시)를 같은 시드의 기준·투영 셀에 적용.
  python scripts/nsp_lockin_diag.py → results/nsp_lockin_diag.{txt,json}"""
import json, sys
import numpy as np
sys.path.insert(0, "/SPECPHYS_ROOT/scripts"); import selfsup_phase0_diag as D
R = D.R; PK = json.load(open(f"{R}/selfsup_nuis_peaks.json")); LINES = []; OUT = {}
def emit(s): print(s); LINES.append(s)
SEEDS = (600, 700, 800, 1600, 1700, 1800, 1900, 2000)
CELLS = {"기준 레시피": "NF_SKREF_REP_L", "투영 λ=0.01": "NF_NSP2_L", "원본 어텐션(1600~2000)": "NF_ORIGA_L"}
import re, mmpd_common_gt as _MGC
def mmpd_key(v):
    g = re.match(r"subject(\d+)_L(\d+)_MO(\d+)_E(\d+)_S(\d+)", v); return tuple(int(x) for x in g.groups())
_MK = {mmpd_key(v): v for v in PK["MMPD"]}
def mmpd_preds(ml):
    """새 집계의 vid 이름이 피크 캐시와 달라 (피험자, 조명, 움직임, 운동, 피부) 로 잇는다. 공통 정답 사용."""
    out = {}
    for m in ml:
        for r in _MGC.load(f"{R}/agg_{m}_MMPD.json"):
            v = _MK.get((r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]))
            if v: out.setdefault(v, []).append((float(r["pred"]), float(r["gt"])))
    return out
GROUPS = {"VIPL": {"VIPL v7(운동 후)": lambda v, k: "_v7_" in v, "VIPL ≥120": lambda v, k: (k.get("gt") or 0) >= 120, "VIPL v2·v9(움직임)": None},
          "MMPD": {"MMPD 보행": lambda v, k: k.get("motion") == 4 or (k.get("t") or {}).get("motion") == 4, "MMPD 회전": lambda v, k: k.get("motion") == 2 or (k.get("t") or {}).get("motion") == 2, "MMPD 대화": lambda v, k: k.get("motion") == 3}}
GROUPS["VIPL"].pop("VIPL v2·v9(움직임)")
for axis, grp in GROUPS.items():
    for gname, sel in grp.items():
        vids = [v for v, k in PK[axis].items() if sel(v, k)]
        emit(f"--- {gname} (n={len(vids)})"); OUT[gname] = {"n": len(vids)}
        for nm, cell in CELLS.items():
            sd = [s for s in SEEDS if not (cell == "NF_ORIGA_L" and s in (600, 700, 800))]; ml = [f"{cell}_s{s}" for s in sd]; preds = mmpd_preds(ml) if axis == "MMPD" else D.load_preds(axis, ml)
            cnt = {}; tot = 0.0; errs = []
            for v in vids:
                pg = preds.get(v)
                if not pg: continue
                gt_ax = PK[axis][v].get("gt")
                for pred, gt in pg:
                    gtv = gt if (gt_ax is None or axis == "MMPD") else gt_ax
                    c, _ = D.classify(pred, gtv, PK[axis][v]["pk_g"], PK[axis][v]["pk_m"]); cnt[c] = cnt.get(c, 0.0) + 1.0 / len(pg); tot += 1.0 / len(pg); errs.append(abs(pred - gtv))
            fails = tot - cnt.get("ok", 0.0); fr = {c: 100.0 * cnt.get(c, 0.0) / max(tot, 1e-9) for c in ("ok", "half_double", "lockin", "prior", "other")}
            emit(f"  {nm:22s} MAE {np.mean(errs):6.2f} | ok {fr['ok']:4.1f}%  half/double {fr['half_double']:4.1f}%  lock-in {fr['lockin']:4.1f}%  prior {fr['prior']:4.1f}%  other {fr['other']:4.1f}% | 실패 중 lock-in {100 * cnt.get('lockin', 0.0) / max(fails, 1e-9):4.1f}% (n={tot:.0f}, seeds {len(sd)})")
            OUT[gname][nm] = dict(mae=float(np.mean(errs)), frac=fr, fail_lockin=float(cnt.get("lockin", 0.0) / max(fails, 1e-9)))
open(f"{R}/nsp_lockin_diag.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/nsp_lockin_diag.json", "w"), indent=1, ensure_ascii=False)
