#!/usr/bin/env python3
"""[69 문서] 기여 근거 그림(본문 후보): (a) 어려운 통제 심박 시험, (b) 쉬운 통제 심박 시험 — 목표 심박별 절대 오차, (c) 같은 사람의 심박 변화 추종 기울기(VIPL-HR·MMPD).
숫자는 results/contrib_judge.json 에서만 읽는다(정본 추적). 팔레트는 dataviz 검증기 통과(대비 낮은 색은 범례·표식 모양으로 보완).
  python scripts/contrib_figs.py → results/figs_nf/Fig_contrib.{png,pdf}"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "/SPECPHYS_ROOT/_workspace/data/results"; OUT = f"{R}/figs_nf"; os.makedirs(OUT, exist_ok=True)
J = json.load(open(f"{R}/contrib_judge.json"))
INK, MUTED, GRID = "#1f2328", "#5b6470", "#e3e6ea"
# 고정 순서의 범주 색(검증: node validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4,#6250d6" --mode light → ALL PASS)
S = [("SpecPhys", "SpecPhys (ours)", "#2a78d6", "o", "-", 2.2),
     ("previous PhysFormer + corpus", "PhysFormer + our corpus", "#eb6834", "s", "-", 1.4),
     ("SpecPhys without synthetic clips", "SpecPhys w/o synthetic clips", "#1baf7a", "D", "--", 1.6),
     ("public PhysMamba", "PhysMamba (public)", "#eda100", "v", "-", 1.4),
     ("public RhythmFormer", "RhythmFormer (public)", "#e87ba4", "^", "-", 1.4),
     ("public FactorizePhys", "FactorizePhys (public)", "#6250d6", "X", "-", 1.4)]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})
f = plt.figure(figsize=(7.0, 2.75)); gs = f.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.30, left=0.07, right=0.99, top=0.77, bottom=0.16)
A = J["A"]["levels"]; tags = [t for t in A if t.startswith("hard")] + [t for t in A if t.startswith("easy")]
titles = ["(a) Weak pulse, strong noise", "(b) Clean pulse"]
for i, tag in enumerate(tags):
    ax = f.add_subplot(gs[0, i]); L = A[tag]
    for key, lab, col, mk, ls, lw in S:
        if key not in L: continue
        hr = sorted(int(h) for h in L[key]["per_hr"]); y = [L[key]["per_hr"][str(h)] for h in hr]
        ax.plot(hr, y, ls=ls, lw=lw, color=col, marker=mk, ms=3.6 if key != "SpecPhys" else 4.2, mec="white", mew=0.5, label=lab, zorder=5 if key == "SpecPhys" else 3)
    ax.set_title(titles[i], fontsize=8, color=INK, loc="left", pad=3); ax.set_xlabel("injected heart rate (bpm)"); ax.set_xticks([45, 75, 105, 135, 165])
    if i == 0: ax.set_ylabel("absolute error (bpm)")
    ax.set_ylim(-2, 100); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    m = L["SpecPhys"]["mae"]; ax.text(0.03, 0.99, f"SpecPhys mean {m:.2f} bpm", transform=ax.transAxes, ha="left", va="top", fontsize=6.6, color=INK)
ax = f.add_subplot(gs[0, 2]); B = [("VIPL-HR", J["B_VIPL-HR"]), ("MMPD", J["B_MMPD 7-33"])]
models = [s for s in S if s[0] != "SpecPhys without synthetic clips"]; n = len(models); w = 0.8 / n; xs = np.arange(len(B))
for j, (key, lab, col, mk, ls, lw) in enumerate(models):
    vals = [b["levels"][key]["slope"] for _, b in B]; xx = xs + (j - (n - 1) / 2) * w
    ax.bar(xx, vals, w * 0.86, color=col, edgecolor="white", linewidth=0.6, zorder=3)
    if key == "SpecPhys":
        lo = [min(b["seed_slopes"]) for _, b in B]; hi = [max(b["seed_slopes"]) for _, b in B]
        ax.errorbar(xx, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6, zorder=4)
        for x_, v in zip(xx, vals): ax.text(x_, v + 0.06, f"{v:.2f}", ha="center", va="bottom", fontsize=6.3, color=INK)
ax.set_xticks(xs); ax.set_xticklabels([b for b, _ in B]); ax.set_ylim(0, 1.0); ax.set_ylabel("within-subject slope")
ax.axhline(1.0, color=MUTED, lw=0.6, ls=":"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.set_title("(c) Within-person tracking", fontsize=8, color=INK, loc="left", pad=3)
h, l = f.axes[0].get_legend_handles_labels(); f.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.8, handlelength=2.4, columnspacing=1.6, bbox_to_anchor=(0.5, 1.0))
f.savefig(f"{OUT}/Fig_contrib.png", dpi=220, facecolor="white"); f.savefig(f"{OUT}/Fig_contrib.pdf", facecolor="white"); plt.close(f); print("wrote", f"{OUT}/Fig_contrib.png")
