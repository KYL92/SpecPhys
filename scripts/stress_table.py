#!/usr/bin/env python3
"""[45 문서 ⑨] 공개 가중치 통제 스트레스 표: 13축 MAE·배가/반감 편수 + 채택 모델(시드 100)·분해 모델 대비 영상 쌍대 Wilcoxon → results/stress_table.{md,json}
모델: base(PhysFormer-UBFC) RhythmFormer FactorizePhys EfficientPhys_arch PhysNet_arch DeepPhys_arch Tscan_arch(MA-UBFC) + CoTAF_TW_L, CoTAM_HYB4_L"""
import json, os, numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"
MODELS = [("base", "PhysFormer (UBFC)"), ("RhythmFormer", "RhythmFormer (UBFC)"), ("FactorizePhys", "FactorizePhys (UBFC)"), ("PhysNet_arch", "PhysNet (UBFC)"), ("EfficientPhys_arch", "EfficientPhys (UBFC)"), ("DeepPhys_arch", "DeepPhys (UBFC)"), ("Tscan_arch", "TS-CAN (MA-UBFC)"), ("CoTAF_TW_L", "채택 모델(공식 코퍼스)"), ("CoTAM_HYB4_L", "분해 모델")]
AXES = ["REALTEST", "RTNI", "RTNC", "C3", "C6", "E3", "PURE", "MCD_A", "MCD_B", "DLCN", "NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE"]
def load(m):
    f = f"{R}/agg_{m}.json"; return json.load(open(f)) if os.path.exists(f) else {}
D = {m: load(m) for m, _ in MODELS}; ours = D["CoTAF_TW_L"]
def wp(x, y): return float("nan") if len(x) < 6 or np.allclose(x, y) else float(wilcoxon(x, y).pvalue)
rows = []; out = {}
for m, label in MODELS:
    cells = []; out[m] = {}
    for ax in AXES:
        r = D[m].get(ax)
        if not r: cells.append("–"); continue
        e = np.array([x["err"] for x in r]); dbl = sum(abs(x["ratio"] - 2) < 0.2 for x in r); hlv = sum(abs(x["ratio"] - 0.5) < 0.1 for x in r)
        lock = sum(abs(x["pred"] - 30 * 60 / 30) < 1.0 for x in r)  # 플리커 lock-in: 예측이 60 bpm(1 Hz 플리커 하모닉) 근처? (참고용)
        p = ""
        if m != "CoTAF_TW_L" and ours.get(ax):
            o = {str(x["vid"]): x["err"] for x in ours[ax]}; ks = [x for x in r if str(x["vid"]) in o]
            if len(ks) >= 6: p = wp([o[str(x["vid"])] for x in ks], [x["err"] for x in ks]); p = f" (p={p:.3f})" if p == p else ""
        out[m][ax] = dict(mae=float(e.mean()), n=len(e), dbl=int(dbl), hlv=int(hlv)); cells.append(f"{e.mean():.2f}{'·' + str(dbl) + 'd' if dbl else ''}{'·' + str(hlv) + 'h' if hlv else ''}{p}")
    rows.append(f"| {label} | " + " | ".join(cells) + " |")
hdr = "| 모델 | " + " | ".join(AXES) + " |\n|" + "---|" * (len(AXES) + 1)
md = "## T18. 공개 가중치 통제 스트레스 표 (정본 프로토콜 MAE; 숫자 뒤 ·Nd/·Nh = 배가/반감 편수; 괄호 p = 채택 모델 대비 영상 쌍대 Wilcoxon, 비보정)\n" + hdr + "\n" + "\n".join(rows) + "\n"
open(f"{R}/stress_table.md", "w").write(md); json.dump(out, open(f"{R}/stress_table.json", "w"), indent=1, ensure_ascii=False); print(md)
