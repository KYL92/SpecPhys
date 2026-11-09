#!/usr/bin/env python3
"""논문 v2(PhysFormer/RhythmFormer 양식) 표 생성: agg_*.json → MAE/RMSE/MAPE/r(+dbl/hlv) 마크다운.
출력: _workspace/data/results/tables_v2.md  (MMPD/CoTAM 결과가 생기면 재실행으로 자동 반영)"""
import json, os, glob, numpy as np
RES = "/SPECPHYS_ROOT/_workspace/data/results"
def load(model):
    d = {}
    for suf in ("", "_newsets"):
        p = f"{RES}/agg_{model}{suf}.json"
        if os.path.exists(p): d.update(json.load(open(p)))
    p = f"{RES}/agg_{model.replace('_arch', '')}_MMPD.json"
    if os.path.exists(p):
        rows = json.load(open(p)); d["MMPD"] = [r for r in rows if r["subject"] >= 7]; d["MMPD_ALL"] = rows
        for key, vals in (("light", {1: "LEDlow", 2: "LEDhigh", 3: "Incand", 4: "Nature"}), ("motion", {1: "Stat", 2: "Rot", 3: "Talk", 4: "Walk"}), ("skin", {3: "S3", 4: "S4", 5: "S5", 6: "S6"})):
            for v, nm in vals.items(): d[f"MMPD_{nm}"] = [r for r in d["MMPD"] if r[key] == v]
        d["MMPD_Exer"] = [r for r in d["MMPD"] if r["exercise"] == 1]; d["MMPD_NoExer"] = [r for r in d["MMPD"] if r["exercise"] == 2]
    return d
def met(rows, sub=None):
    if not rows: return None
    if sub: rows = [r for r in rows if sub(r)]
    if not rows: return None
    g = np.array([r["gt"] for r in rows]); p = np.array([r["pred"] for r in rows]); e = np.abs(g - p)
    r = float(np.corrcoef(g, p)[0, 1]) if len(rows) > 2 and g.std() > 0 and p.std() > 0 else float("nan")
    return dict(n=len(rows), MAE=e.mean(), RMSE=np.sqrt((e**2).mean()), MAPE=(e/g).mean()*100, r=r,
                dbl=sum(abs(x["ratio"]-2) < 0.2 for x in rows), hlv=sum(abs(x["ratio"]-0.5) < 0.1 for x in rows))
def pos_rows(ax):
    p = f"{RES}/pos_pervideo_{ax}.json"
    if not os.path.exists(p): return None
    d = json.load(open(p)); return [dict(err=v) for v in d.values()]
def cell(m, full=True):
    if m is None: return "–"
    return f"{m['MAE']:.2f} / {m['RMSE']:.2f} / {m['MAPE']:.1f} / {m['r']:.2f}" if full else f"{m['MAE']:.2f}"
MODELS = ["base", "CoTNH", "CoTNHM", "CoTA", "CoTAF", "CoTAF2", "CoTAM", "CoTAF_L", "CoTAM_L", "CoTN_L", "CoTNH_L", "CoTNHM_L", "CoTA_L", "CoTAF2_L", "CoTNC_L", "CoTNW_L", "CoTNH2_L", "CoTB_L", "CoTBH_L", "CoTAF_T_L", "CoTAF_TW_L", "CoTAF_TS_L", "CoTAF_TC_L", "CoTAF_TWM_L", "CoTAF_TW_L_s200", "CoTAF_TW_L_s300", "CoTAF_L_s200", "CoTAF_L_s300", "RhythmFormer_arch", "RFA", "FactorizePhys_arch", "FFA", "EfficientPhys_arch", "PhysNet_arch"]
NAMES = {"base": "PhysFormer (UBFC)", "CoTNH": "PhysFormer + ours-H", "CoTNHM": "PhysFormer + ours-HM", "CoTA": "PhysFormer + ours-A", "CoTAF": "**PhysFormer + ours (CoTAF)**", "CoTAF2": "PhysFormer + ours-F2", "CoTAM": "**PhysFormer + ours + MMPD (CoTAM)**", "CoTAF_L": "PhysFormer + ours, label-fix (CoTAF-L)", "CoTAM_L": "PhysFormer + ours + MMPD, label-fix (CoTAM-L)", "CoTN_L": "L: +noise co-training (CoTN-L)", "CoTNH_L": "L: +harmonics (CoTNH-L)", "CoTNHM_L": "L: +high-HR+MCD (CoTNHM-L)", "CoTA_L": "L: +PURE (CoTA-L)", "CoTAF2_L": "L: +driving stress (CoTAF2-L)", "CoTNC_L": "L: ablation compression-only (CoTNC-L)", "CoTNW_L": "L: ablation waveform-only (CoTNW-L)", "CoTNH2_L": "L: ablation real-matched harmonics (CoTNH2-L)", "CoTB_L": "L: balanced pool (CoTB-L)", "CoTBH_L": "L: balanced pool + harmonics (CoTBH-L)",
         "RhythmFormer_arch": "RhythmFormer (UBFC)", "RFA": "RhythmFormer + ours", "FactorizePhys_arch": "FactorizePhys (UBFC)", "FFA": "FactorizePhys + ours", "EfficientPhys_arch": "EfficientPhys (UBFC)", "PhysNet_arch": "PhysNet (UBFC)"}
NAMES.update({"CoTAF_TW_L": "**PhysFormer + ours-TW (CoTAF-TW-L, adopted)**", "CoTAF_T_L": "twin v1: bank+specular+codec (CoTAF-T-L)", "CoTAF_TS_L": "twin abl.: specular-only (CoTAF-TS-L)", "CoTAF_TC_L": "twin abl.: codec-only (CoTAF-TC-L)", "CoTAF_TWM_L": "twin abl.: +real-driving motion (CoTAF-TWM-L)", "CoTAF_TW_L_s200": "CoTAF-TW-L seed200", "CoTAF_TW_L_s300": "CoTAF-TW-L seed300", "CoTAF_L_s200": "CoTAF-L seed200", "CoTAF_L_s300": "CoTAF-L seed300"})
D = {m: load(m) for m in MODELS}
AX = [("PURE", "PURE held-out 07–10", lambda r: r["vid"] >= 700), ("PURE_ALL", "PURE all 01–10", None), ("MCD_B", "MCD stable", None), ("MCD_A", "MCD post-exercise", None),
      ("DLCN", "DLCN (zero-shot)", None), ("NIRP_STILL", "MR-NIRP still", None), ("NIRP_SMALL", "MR-NIRP small", None), ("NIRP_LARGE", "MR-NIRP large", None),
      ("MMPD", "MMPD held-out 7–33", None), ("MMPD_ALL", "MMPD all 1–33", None), ("REALTEST", "UBFC held-out 42–49", None), ("RTNI", "UBFC+flicker", None), ("RTNC", "UBFC+color-temp", None)]
out = ["# Tables v2 (auto) — MAE / RMSE / MAPE(%) / Pearson r ; dbl=harmonic doubling count, hlv=halving\n", "정본 프로토콜: toolbox(cumsum→detrend λ=100→butter1 0.6–3.3Hz→periodogram), per-video. 자동 생성 — 수정 금지, 재실행: `python scripts/paper_tables_v2.py`.\n"]
def table(axes, models, full=True):
    hdr = "| Method | " + " | ".join(a[1] for a in axes) + " |"; lines = [hdr, "|" + "---|" * (len(axes) + 1)]
    for m in models:
        cells = []
        for key, _, sub in axes:
            k = "PURE" if key == "PURE_ALL" else key
            mm = met(D[m].get(k), sub if key == "PURE" else None); cells.append(cell(mm, full))
        lines.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    # POS from per-video files
    cells = []
    for key, _, _ in axes:
        k = {"PURE": "PURE", "PURE_ALL": "PURE", "MMPD": "MMPD", "MMPD_ALL": "MMPD"}.get(key, key)
        if k == "MMPD" and os.path.exists(f"{RES}/agg_POS_MMPD.json"):
            rows = json.load(open(f"{RES}/agg_POS_MMPD.json")); rows = rows if key == "MMPD_ALL" else [r for r in rows if r["subject"] >= 7]; cells.append(cell(met(rows), full)); continue
        pr = pos_rows(k)
        if pr is None: cells.append("–"); continue
        if key == "PURE": pr = [dict(err=v) for kk, v in json.load(open(f"{RES}/pos_pervideo_PURE.json")).items() if int(kk) >= 700]
        e = np.array([r["err"] for r in pr]); cells.append((f"{e.mean():.2f} / {np.sqrt((e**2).mean()):.2f} / – / –") if full else f"{e.mean():.2f}")
    lines.append("| POS (classical) | " + " | ".join(cells) + " |")
    return "\n".join(lines)
out.append("\n## T1. Cross-dataset, train UBFC(+ours) → test X  [MAE/RMSE/MAPE/r]\n"); out.append(table(AX[:1] + AX[2:9], ["base", "CoTNH", "CoTA", "CoTAF", "CoTAF2", "CoTAM", "CoTAF_L", "CoTAM_L", "CoTAF_TW_L", "CoTN_L", "CoTNH_L", "CoTNHM_L", "CoTA_L", "CoTAF2_L", "CoTNC_L", "CoTNW_L", "CoTNH2_L", "CoTB_L", "CoTBH_L", "RhythmFormer_arch", "RFA", "FactorizePhys_arch", "FFA", "EfficientPhys_arch", "PhysNet_arch"]))
out.append("\n\n## T2. In-domain / held-out (identity- or subject-disjoint) [MAE only]\n"); out.append(table([AX[10], AX[0], AX[2], AX[3], AX[8]], ["base", "CoTA", "CoTAF", "CoTAM", "CoTAF_L", "CoTAM_L", "CoTAF_TW_L", "RhythmFormer_arch", "RFA"], full=False))
out.append("\n\n## T3. Real nuisance on UBFC held-out (flicker / colour-temperature) [MAE]\n"); out.append(table(AX[10:13], ["base", "CoTNH", "CoTAF", "CoTAF2", "CoTAM", "RhythmFormer_arch", "RFA"], full=False))
MM = [(f"MMPD_{n}", n, None) for n in ["LEDlow", "LEDhigh", "Incand", "Nature", "Stat", "Rot", "Talk", "Walk", "Exer", "NoExer", "S3", "S4", "S5", "S6"]]
if D["base"].get("MMPD"):
    out.append("\n\n## T4. MMPD held-out (7–33) by condition [MAE]\n"); out.append(table(MM, ["base", "CoTAF", "CoTAM", "CoTAF_L", "CoTAM_L", "CoTAF_TW_L", "RhythmFormer_arch", "RFA", "FactorizePhys_arch", "FFA"], full=False))
else: out.append("\n\n## T4. MMPD by condition — (chain8 대기 중)\n")
out.append("\n\n## T5. Harmonic doubling / halving counts (canonical) \n")
lines = ["| Method | " + " | ".join(a[1] for a in AX[:1] + AX[2:9]) + " |", "|" + "---|" * 9]
for m in ["base", "CoTNH", "CoTA", "CoTAF", "CoTAM", "CoTAF_L", "CoTAF_TW_L", "RhythmFormer_arch", "RFA", "FactorizePhys_arch", "FFA"]:
    cells = []
    for key, _, sub in AX[:1] + AX[2:9]:
        mm = met(D[m].get(key), sub); cells.append("–" if mm is None else f"{mm['dbl']}/{mm['hlv']} (n={mm['n']})")
    lines.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
out.append("\n".join(lines))
out.append("\n\n## T6. Label-fix lineage (all PhysFormer models retrained with RPPG_HR_LABEL=cumsum, batch 4) [MAE/RMSE/MAPE/r]\n"); out.append(table(AX[:1] + AX[2:9], ["base", "CoTN_L", "CoTNH_L", "CoTNHM_L", "CoTA_L", "CoTAF_L", "CoTAF2_L", "CoTAM_L", "CoTNC_L", "CoTNW_L", "CoTNH2_L", "CoTB_L", "CoTBH_L"]))

out.append("\n\n## T7. Digital-twin factor decomposition (pool 203 fixed; baseline CoTAF-L) [MAE/RMSE/MAPE/r]\n")
out.append(table(AX[:1] + AX[2:9], ["CoTAF_L", "CoTAF_T_L", "CoTAF_TW_L", "CoTAF_TS_L", "CoTAF_TC_L", "CoTAF_TWM_L"]))
out.append("\n\n## T8. Multi-seed confirmation (RPPG_SEED 100/200/300) — MAE mean \u00b1 sd\n")
ax8 = [("MMPD", "MMPD held-out 7\u201333", None), ("PURE", "PURE held-out 07\u201310", lambda r: r["vid"] >= 700), ("MCD_A", "MCD post-exercise", None), ("DLCN", "DLCN (zero-shot)", None), ("NIRP_LARGE", "MR-NIRP large", None)]
l8 = ["| Method | " + " | ".join(a[1] for a in ax8) + " |", "|" + "---|" * (len(ax8) + 1)]
for bm, nm in [("CoTAF_L", "CoTAF-L (baseline)"), ("CoTAF_TW_L", "**CoTAF-TW-L (adopted)**")]:
    cells = []
    for key, _, sub in ax8:
        vals = [mm["MAE"] for s in ("", "_s200", "_s300") for mm in [met(D[bm + s].get(key), sub)] if mm]
        cells.append(f"{np.mean(vals):.2f} \u00b1 {np.std(vals, ddof=1):.2f} (n={len(vals)})" if len(vals) >= 2 else "\u2013")
    l8.append(f"| {nm} | " + " | ".join(cells) + " |")
out.append("\n".join(l8))
open(f"{RES}/tables_v2.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
