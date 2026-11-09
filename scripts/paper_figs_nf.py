#!/usr/bin/env python3
"""[56 문서] 논문 그림 사양(패널형 다이어그램, FactorizePhys/NeurIPS 스타일) — 한 사양에서 두 렌더러:
  ① matplotlib → results/figs_nf/fig*.png (검수용 미리보기) + .pdf (LaTeX 용)  ② JSON → scripts/paper_figs_nf.js(pptxgenjs) 가 편집 가능한 PPT 도형으로 그림
  python scripts/paper_figs_nf.py   → results/figs_nf/{Fig0_teaser,Fig1_overview,Fig2_signals,Fig3_results,Table2_compact}.{png,pdf}, results/figs_nf/fig_spec.json  (Fig0 = CVPR 티저, 58 문서)
좌표 단위 = 인치(슬라이드 13.33×7.5). 아이템: panel(둥근 배경+라벨), box(둥근 모듈), text, circle, arrow, line, axis(스펙트럼 스케치), chart(막대), table.
숫자는 results/paper_tables_nf.json 에서만 읽는다(정본 추적)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, FancyArrowPatch

R = "/SPECPHYS_ROOT/_workspace/data/results"; OUT = f"{R}/figs_nf"; os.makedirs(OUT, exist_ok=True)
T = json.load(open(f"{R}/paper_tables_nf.json"))
FT = json.load(open(f"{R}/final_config_tables.json"))   # [62 §14] 최종 설정(SpecPhys) 수치, MMPD 공통 정답
W, H = 13.33, 7.5
# 팔레트(참조 그림: 연노랑·연초록·연보라 패널, 라벤더 모듈, 색 토큰)
C = dict(yellow="FBF4DC", green="E9F3E6", blue="EAF0F8", lav="D9E1F2", lav2="E6ECF7", white="FFFFFF", ink="1F2328", muted="5B6470",
         tokY="F4D35E", tokG="9BC995", tokO="E8A87C", tokB="A8C4E5", red="C0392B", gray="BFC5CE", accent="2F3C7E", teal="1F7A8C")
FONT = "Arial"


def it(kind, **kw): d = dict(kind=kind); d.update(kw); return d


def panel(x, y, w, h, fill, label, dash=False): return it("panel", x=x, y=y, w=w, h=h, fill=fill, label=label, dash=dash)
def box(x, y, w, h, text, fill="lav", fs=10.5, bold=False, border=None): return it("box", x=x, y=y, w=w, h=h, text=text, fill=fill, fs=fs, bold=bold, border=border)
def txt(x, y, w, h, text, fs=10, bold=False, color="ink", align="left", italic=False): return it("text", x=x, y=y, w=w, h=h, text=text, fs=fs, bold=bold, color=color, align=align, italic=italic)
def circ(x, y, d, fill, text="", fs=9): return it("circle", x=x, y=y, d=d, fill=fill, text=text, fs=fs)
def arrow(x1, y1, x2, y2, color="ink", dash=False, width=1.5): return it("arrow", x1=x1, y1=y1, x2=x2, y2=y2, color=color, dash=dash, width=width)
def line(x1, y1, x2, y2, color="ink", dash=False, width=1.2): return it("line", x1=x1, y1=y1, x2=x2, y2=y2, color=color, dash=dash, width=width)


# ======================= Figure 0: CVPR teaser (full width, 13.33 x 4.3 in) =======================
def fig0():
    I = []; cross = T["cross"]; vb = T["vipl_bands"]
    def pub(ax): return cross[ax]["FactorizePhys"]["mae"]
    def ours(ax): return cross[ax][next(k for k in cross[ax] if k.startswith("SpecPhys (ours, final)"))]["mae"]
    mm_fin = T["mmpd_motion"]["SpecPhys (ours, final)"]
    # (a) problem
    I.append(panel(0.25, 0.25, 4.15, 3.8, "yellow", "(a) Public weights follow the nuisance"))
    I.append(it("axis", x=0.5, y=0.55, w=3.65, h=1.75, hr=75, band=6, peaks=[(66, 1.0), (75, 0.35), (132, 0.2)], label="output spectrum of a public checkpoint: flicker (66 bpm) wins over HR"))
    I.append(txt(0.45, 2.5, 3.8, 1.1, "Controlled flicker (RTNI): MAE ≈ 30 bpm for PhysNet, EfficientPhys, RhythmFormer and FactorizePhys (constant prediction).\nSame family in the wild: walking (MMPD), post-exercise heart rates (VIPL-HR), driving (MR-NIRP).", fs=9))
    # (b) recipe
    I.append(panel(4.6, 0.25, 4.3, 3.8, "green", "(b) Training signal and model"))
    I.append(box(4.8, 0.5, 3.9, 0.46, "generated face, animated; real PPG injected per skin pixel along P_{bv} (<1 %)", fill="lav", fs=9))
    I.append(arrow(6.75, 0.96, 6.75, 1.08)); I.append(box(4.8, 1.08, 3.9, 0.46, "render twice (pulse on / off, same nuisance draw) → exact nuisance trace n(t)", fill="tokY", fs=9))
    I.append(arrow(6.75, 1.54, 6.75, 1.66)); I.append(box(4.8, 1.66, 3.9, 0.46, "366-clip pool: real clips (PPG labels) + synthetic clips (PPG + nuisance labels)", fill="lav2", fs=9))
    I.append(arrow(6.75, 2.12, 6.75, 2.24)); I.append(box(4.8, 2.24, 3.9, 0.62, "FactorizePhys trunk + spectral losses; doubling penalty; skin-weighted input;\noutput excluded from background, non-skin and motion frequencies", fill="tokG", fs=9))
    I.append(arrow(6.75, 2.86, 6.75, 2.98)); I.append(box(4.8, 2.98, 3.9, 0.62, "SpecPhys: project the clip's noise references out of the temporal features\n(no new parameters; side-band colour and global motion)", fill="tokB", fs=9, bold=True))
    # (c) outcome
    I.append(panel(9.1, 0.25, 3.98, 3.8, "blue", "(c) Both ends, and what remains"))
    fin = "SpecPhys (ours, final)"; pubv = "FactorizePhys (public)"
    cats = ["VIPL\nall", "VIPL\n<60", "VIPL\n≥120", "MMPD\n7–33", "NIRP\nsmall", "RTNI\nflicker"]
    pv = [pub("VIPL"), vb[pubv]["<60 bpm"], vb[pubv]["≥120"], pub("MMPD"), pub("NIRP_SMALL"), pub("RTNI")]; ov = [ours("VIPL"), vb[fin]["<60 bpm"], vb[fin]["≥120"], ours("MMPD"), ours("NIRP_SMALL"), ours("RTNI")]
    I.append(it("chart", x=9.2, y=0.45, w=3.8, h=2.45, cats=cats, series=[("public FactorizePhys", pv, "gray"), ("SpecPhys", ov, "accent")], ylabel="MAE (bpm)"))
    L = FT["levels"]
    I.append(txt(9.3, 2.95, 3.65, 1.0, f"Remains: MMPD ≥110 bpm {L['SpecPhys (final)']['mhi']:.1f} vs. {L['public RhythmFormer']['mhi']:.1f} for the high-rate-biased public RhythmFormer; walking {mm_fin['walking']:.1f} bpm; a face parser stays at the input. Target labels lower MMPD further at a low-rate cost.", fs=9))
    return dict(title="Fig0_teaser", items=I, W=13.33, H=4.3)


# ======================= Figure 1: overview =======================
def fig1():
    I = []
    # ---- (a) corpus
    I.append(panel(0.25, 0.3, 6.3, 3.45, "yellow", "(a) PPG-injected corpus with observable nuisance labels"))
    I.append(box(0.45, 0.6, 1.25, 0.62, "Identity\nSDXL / SFHQ face", "white"))
    I.append(arrow(1.70, 0.91, 1.95, 0.91))
    I.append(box(1.95, 0.6, 1.35, 0.62, "LivePortrait\nanimation", "white"))
    I.append(arrow(3.30, 0.91, 3.55, 0.91))
    I.append(box(3.55, 0.52, 1.6, 0.78, "Per-pixel PPG injection\nalong P_{bv} = [0.33, 0.77, 0.53]\nreal PPG bank waveform", "lav", fs=9.5))
    I.append(arrow(5.15, 0.91, 5.4, 0.91))
    I.append(box(5.4, 0.6, 1.0, 0.62, "Lossless\nFFV1 clip", "white"))
    I.append(box(0.45, 1.5, 2.85, 0.78, "Nuisance generator\ndrift · flicker 0.9–1.3 Hz · head motion · sensor noise", "lav2", fs=9.2))
    I.append(arrow(1.9, 1.3, 1.9, 1.5, width=1.2))
    I.append(arrow(3.3, 1.89, 3.55, 1.89))
    I.append(box(3.55, 1.5, 2.85, 0.78, "Render twice: pulse on / pulse off\n→ exact nuisance trace n(t) (dual-render sidecar)", "lav2", fs=9.2))
    I.append(txt(0.45, 2.38, 6.0, 0.3, "Hybrid training pool (366 clips) — UBFC checkpoint fine-tuned, 10 epochs", fs=10.5, bold=True))
    toks = [("tokB", "real UBFC 34"), ("tokB", "real PURE 35"), ("tokB", "real MCD 48"), ("tokB", "real MMPD 1–6 120"), ("tokG", "real + synthetic noise 29"), ("tokY", "synthetic 80"), ("tokY", "codec copies 20"), ("tokO", "dual-render sidecars 48")]
    for i, (col, lab) in enumerate(toks):
        x = 0.5 + (i % 4) * 1.5; y = 2.68 + (i // 4) * 0.27
        I.append(circ(x, y, 0.2, col)); I.append(txt(x + 0.24, y - 0.03, 1.25, 0.28, lab, fs=8))
    I.append(txt(0.45, 3.2, 6.0, 0.28, "Real clips carry PPG/HR labels but no nuisance label; only the 48 synthetic sidecars have n(t).", fs=8.5, italic=True, color="muted"))
    # ---- (b) backbone
    I.append(panel(6.75, 0.3, 6.33, 3.45, "green", "(b) Spectrum-guided factorization backbone"))
    I.append(box(6.95, 0.6, 0.86, 0.62, "x\n3×161×72²", "white", fs=9))
    I.append(arrow(7.81, 0.91, 7.93, 0.91))
    I.append(box(7.93, 0.6, 0.8, 0.62, "Δt + inst.\nnorm", "lav", fs=9))
    I.append(arrow(8.73, 0.91, 8.85, 0.91))
    I.append(box(8.85, 0.6, 0.72, 0.62, "3D-CNN\ntrunk", "lav", fs=9))
    I.append(arrow(9.57, 0.91, 9.69, 0.91))
    I.append(box(9.69, 0.6, 0.78, 0.62, "F\n16×T×7²", "white", fs=9))
    I.append(arrow(10.47, 0.91, 10.59, 0.91))
    I.append(box(10.59, 0.52, 1.34, 0.78, "noise-subspace\nprojection F → F′\n(no parameters)", "tokB", fs=9, bold=True))
    I.append(arrow(11.93, 0.91, 12.05, 0.91))
    I.append(box(12.05, 0.6, 0.9, 0.62, "BVP\nhead", "lav"))
    I.append(arrow(12.5, 1.22, 12.5, 1.5, width=1.2))
    I.append(box(12.05, 1.5, 0.9, 0.5, "rPPG ŷ(t)", "white", fs=10, bold=True))
    # NF-FSAM block
    I.append(it("rect", x=6.95, y=1.5, w=4.9, h=1.55, fill="white", border="gray", dash=False))
    I.append(txt(7.05, 1.52, 4.7, 0.3, "NF-FSAM: non-negative matrix factorization of F′ with two temporal bases", fs=10, bold=True))
    I.append(circ(7.15, 1.95, 0.26, "tokY")); I.append(txt(7.47, 1.9, 2.3, 0.35, "pulse slot s_{p}   (init 1/√T)", fs=9.5))
    I.append(circ(7.15, 2.35, 0.26, "tokB")); I.append(txt(7.47, 2.3, 2.5, 0.35, "nuisance slot s_{n}   (init: global intensity trace g′)", fs=9.5))
    I.append(box(9.95, 1.9, 1.8, 0.5, "4 multiplicative updates\n+ 1 differentiable update", "lav2", fs=8.5))
    I.append(box(9.95, 2.48, 1.8, 0.5, "attention from pulse slot only\n⊙ F′ + residual", "lav2", fs=8.5))
    I.append(arrow(11.26, 1.3, 11.26, 1.5, width=1.2))
    I.append(arrow(11.75, 2.73, 12.3, 1.22, width=1.2))
    I.append(txt(7.05, 2.72, 2.8, 0.3, "s_{p}, s_{n} → spectral losses in (c)", fs=9, italic=True, color="muted"))
    I.append(txt(6.95, 3.1, 5.95, 0.4, "Module names match the public FactorizePhys checkpoint. The projection removes the clip's side-band colour and global-motion references from F. On its own the two-slot structure shows no measurable contribution (Table 4).", fs=8.5, italic=True, color="muted"))
    # ---- (c) spectral supervision
    I.append(panel(0.25, 4.0, 6.3, 3.25, "blue", "(c) Spectral guidance and nuisance targets"))
    I.append(it("axis", x=0.5, y=4.35, w=2.6, h=1.35, hr=78, peaks=[(78, 1.0), (156, 0.35), (66, 0.45)], band=3, label="P̂_{ŷ}(f), 30–200 bpm (1 bpm DFT grid)"))
    I.append(box(3.3, 4.35, 3.05, 0.5, "L_{spec} = −log Σ_{|f−HR|≤3} P̂_{ŷ}(f)      (w = 1.0)", "lav", fs=9.5))
    I.append(box(3.3, 4.93, 3.05, 0.5, "L_{half}: mass near HR/2 vs HR      (w = 0.5)", "lav", fs=9.5))
    I.append(box(3.3, 5.51, 3.05, 0.5, "L_{ps}: same spectral NLL on the pulse slot s_{p}   (w = 0.5)", "lav", fs=9.5))
    I.append(box(0.5, 5.85, 2.75, 0.62, "L_{nuis} = 1 − cos(P̂(s_{n}), P̂(n))   (w = 1.0)\nn = sidecar (synthetic) or gated global trace (real)", "lav2", fs=8.8))
    I.append(box(3.3, 6.09, 3.05, 0.4, "L_{decor} = corr²(s_{p}, s_{n})  (w = 0.25)  +  Neg-Pearson(ŷ, PPG)", "lav2", fs=9))
    I.append(txt(0.5, 6.55, 5.9, 0.5, "Chunk HR label = FFT peak of the label chunk (device HR for VIPL). A readout-consistent label was tested post hoc: no effect (Table 5).", fs=8.5, italic=True, color="muted"))
    # ---- (d) protocol
    I.append(panel(6.75, 4.0, 6.33, 3.25, "white", "(d) Evaluation protocol", dash=True))
    I.append(txt(6.95, 4.3, 3.0, 0.3, "Zero-shot (no target labels)", fs=10.5, bold=True, color="accent"))
    zs = ["VIPL-HR 2,378 videos (device HR)", "MMPD held-out subjects 7–33 (540)", "PURE 07–10 · UBFC 42–49", "MR-NIRP RGB: still / small / large motion (19 each)", "DLCN (20) · MCD-rPPG A/B (20 each)", "controlled flicker/drift (RTNI), + motion/noise (RTNC)"]
    for i, s in enumerate(zs): I.append(txt(7.0, 4.62 + 0.3 * i, 3.1, 0.3, "• " + s, fs=9))
    I.append(txt(10.2, 4.3, 2.8, 0.3, "In-domain, two-stage (target labels)", fs=10.5, bold=True, color="teal"))
    ind = ["VIPL-HR subject-exclusive 5-fold", "MMPD 5-fold (660 clips)", "MR-NIRP 5-fold (19 subjects)", "corpus ckpt → fold adaptation (lr 5e-5, 10 ep)", "always reported in separate tables"]
    for i, s in enumerate(ind): I.append(txt(10.25, 4.62 + 0.3 * i, 2.8, 0.3, "• " + s, fs=9))
    I.append(box(6.95, 6.38, 5.95, 0.5, "Canonical HR readout: (cumsum for diff labels) → detrend λ=100 → Butterworth 0.6–3.3 Hz → periodogram peak\nPaired Wilcoxon per video · Holm within each pre-registered family · ≥3 seeds", "lav2", fs=8.6))
    return dict(title="Fig1_overview", items=I)


# ======================= Figure 2: learning-signal study =======================
def fig2():
    I = []
    I.append(panel(0.25, 0.3, 6.3, 3.45, "yellow", "(a) Global-trace self-supervision"))
    I.append(box(0.45, 0.65, 1.55, 0.7, "clip's own proxies\ng′(t), v_{x}(t), v_{y}(t)", "white", fs=9.5))
    I.append(txt(0.45, 1.38, 1.7, 0.45, "global intensity diff + Lucas–Kanade translation (signed)", fs=8, italic=True, color="muted"))
    I.append(arrow(2.0, 1.0, 2.3, 1.0))
    I.append(box(2.3, 0.65, 1.9, 0.7, "nuisance mask M(f)\nmax power, HR bands zeroed", "lav", fs=9.5))
    I.append(arrow(4.2, 1.0, 4.5, 1.0))
    I.append(box(4.5, 0.55, 1.9, 0.5, "L_{excl} = Σ_f P̂_{ŷ}(f) M(f)", "lav2", fs=9.5))
    I.append(box(4.5, 1.12, 1.9, 0.5, "s_{n} ← P̂(g′)·(1 − HR band)", "lav2", fs=9.5))
    I.append(it("axis", x=0.5, y=1.95, w=2.6, h=1.2, hr=72, peaks=[(72, 0.8), (96, 1.0), (192, 0.3)], band=5, label="mask protects HR ± 5 bpm; nuisance peaks elsewhere"))
    I.append(txt(3.3, 1.95, 3.05, 1.5, "Applied to every chunk without a sidecar; soft gate min(1, g_std/1.0).\nRejected: VIPL v7 −0.16 (ns), MMPD walking +1.24 (p=.05), no lock-in reduction; VIPL overall −0.15 (p=.02).", fs=8.8))
    I.append(panel(6.75, 0.3, 6.33, 3.45, "green", "(b) Attention spatial regularization"))
    I.append(it("grid", x=7.0, y=0.65, w=1.4, h=1.4, n=7, label="pulse-slot loading p_{att} (7×7)"))
    I.append(it("grid", x=8.7, y=0.65, w=1.4, h=1.4, n=7, label="skin map S (YCrCb)", kind2="skin"))
    I.append(box(10.4, 0.65, 2.5, 0.42, "L_{skin} = Σ p_{att} (1 − S)   (0.5)", "lav2", fs=9.5))
    I.append(box(10.4, 1.14, 2.5, 0.42, "L_{nreg} = Σ p_{att} · r_{nuis}   (0.5)", "lav2", fs=9.5))
    I.append(box(10.4, 1.63, 2.5, 0.42, "L_{dis} = cos(p_{att}, n_{att})   (0.25)", "lav2", fs=9.5))
    I.append(txt(7.0, 2.5, 5.9, 1.1, "Loss values stayed constant over all 10 epochs in every seed (skin 0.520→0.519, nuisance-region 0.704→0.703): the NMF coefficient maps (4 no-grad multiplicative updates + 1 differentiable step) give the trunk almost no gradient path. The signal cannot act in this architecture.", fs=9))
    I.append(panel(0.25, 4.0, 6.3, 3.25, "blue", "(c) Local nuisance proxy"))
    I.append(it("grid", x=0.5, y=4.35, w=1.3, h=1.3, n=6, label="6×6 cells (luminance diff, LK motion)"))
    I.append(arrow(1.85, 5.0, 2.15, 5.0))
    I.append(box(2.15, 4.6, 2.1, 0.8, "spatial incoherence spectrum\nmean_c P̂_c(f) × std_c/mean_c", "lav", fs=9.5))
    I.append(arrow(4.25, 5.0, 4.55, 5.0))
    I.append(box(4.55, 4.6, 1.85, 0.8, "M = max(M_global, M_local)\n→ L_{excl}", "lav2", fs=9.5))
    I.append(txt(0.5, 6.1, 5.9, 1.0, "Targets local shadows, hands, hair and local motion that the global trace cannot represent.\nRejected: MMPD walking +1.46 (Holm p=.046, worse), VIPL v7 −0.11 (ns), MR-NIRP +0.42 (ns).", fs=9))
    I.append(panel(6.75, 4.0, 6.33, 3.25, "white", "(d) Augmentation consistency", dash=True))
    I.append(box(6.95, 4.4, 1.0, 0.55, "x", "white", fs=11, bold=True))
    I.append(arrow(7.95, 4.67, 8.3, 4.67))
    I.append(box(8.3, 4.3, 2.3, 0.75, "inject known nuisance\nflicker · drift · shift 1–4 px · noise", "lav", fs=9))
    I.append(arrow(10.6, 4.67, 10.95, 4.67))
    I.append(box(10.95, 4.4, 1.0, 0.55, "x′", "white", fs=11, bold=True))
    I.append(arrow(7.45, 4.95, 7.45, 5.4, width=1.2)); I.append(arrow(11.45, 4.95, 11.45, 5.4, width=1.2))
    I.append(box(6.95, 5.4, 1.0, 0.5, "f(x) → ŷ", "lav2", fs=9.5)); I.append(box(10.95, 5.4, 1.0, 0.5, "f(x′) → ŷ′", "lav2", fs=9.5))
    I.append(it("dblarrow", x1=7.95, y1=5.65, x2=10.95, y2=5.65, color="red", dash=True, label="1 − Pearson(ŷ, ŷ′)"))
    I.append(txt(6.95, 6.05, 6.0, 1.1, "+ exclusion of the injected spectrum in ŷ′, nuisance slot ← injected spectrum, supervised losses on x′.  Rejected: v7 +0.43, walking +0.68, MR-NIRP +2.31 (all ns); side effect: full robustness to the trained 1.1 Hz flicker (RTNI 1.46 → 0.00).", fs=9))
    return dict(title="Fig2_signals", items=I)


# ======================= Figure 3: results charts =======================
def fig3():
    I = []
    vb = T["vipl_bands"]; mm = T["mmpd_motion"]; p0 = T["phase0"]; ss = T["selfsup"]
    bands = ["<60 bpm", "60–90", "90–120", "≥120"]
    I.append(panel(0.25, 0.3, 6.3, 3.45, "yellow", "(a) VIPL-HR zero-shot MAE by heart-rate band"))
    I.append(it("chart", x=0.45, y=0.6, w=5.95, h=3.05, cats=bands, series=[("FactorizePhys (public)", [vb["FactorizePhys (public)"][b] for b in bands], "gray"), ("PhysFormer + corpus", [vb["PhysFormer + corpus"][b] for b in bands], "tokO"), ("SpecPhys-B", [vb["SpecPhys-B (ours)"][b] for b in bands], "tokB"), ("SpecPhys (final)", [vb["SpecPhys (ours, final)"][b] for b in bands], "accent")], ylabel="MAE (bpm)"))
    mot = ["stationary", "rotation", "talking", "walking"]
    I.append(panel(6.75, 0.3, 6.33, 3.45, "green", "(b) MMPD held-out subjects 7–33 by motion label"))
    I.append(it("chart", x=6.95, y=0.6, w=5.95, h=3.05, cats=mot, series=[("FactorizePhys (public)", [mm["FactorizePhys (public)"][m] for m in mot], "gray"), ("PhysFormer + corpus", [mm["PhysFormer + corpus"][m] for m in mot], "tokO"), ("SpecPhys-B", [mm["SpecPhys-B (ours)"][m] for m in mot], "tokB"), ("SpecPhys (final)", [mm["SpecPhys (ours, final)"][m] for m in mot], "accent")], ylabel="MAE (bpm)"))
    regs = [("VIPL v7", "VIPL v7 post-exercise"), ("VIPL ≥90", "VIPL HR ≥90 bpm (excl. v7)"), ("VIPL ≥120", "VIPL HR ≥120 bpm"), ("MMPD walk", "MMPD walking"), ("NIRP large", "NIRP_LARGE large motion")]
    obs = [100 * p0[k]["fail_lockin"] for _, k in regs]; nul = [100 * p0[k]["null"]["shuffle_mean"] for _, k in regs]
    I.append(panel(0.25, 4.0, 6.3, 3.25, "blue", "(c) Lock-in share of failures vs. shuffled-prediction null"))
    I.append(it("chart", x=0.45, y=4.3, w=5.95, h=2.85, cats=[r for r, _ in regs], series=[("observed lock-in share", obs, "accent"), ("shuffled-prediction null", nul, "gray")], ylabel="% of failures (|err| > 5 bpm)"))
    I.append(panel(6.75, 4.0, 6.33, 3.25, "white", "(d) Self-supervised signals: Δ MAE vs. SpecPhys-B (lower is better)", dash=True))
    cells = [("global trace", "+ global-trace self-supervision"), ("attention reg.", "+ attention spatial regularization"), ("both", "+ both"), ("local proxy", "+ local + global nuisance proxy"), ("aug. consistency", "+ augmentation consistency")]
    base = ss["SpecPhys-B baseline"]
    I.append(it("chart", x=6.95, y=4.3, w=5.95, h=2.85, cats=[c for c, _ in cells], series=[("VIPL v7 post-exercise", [ss[k]["VIPL v7 post-exercise"] - base["VIPL v7 post-exercise"] for _, k in cells], "tokO"), ("MMPD walking", [ss[k]["MMPD walking"] - base["MMPD walking"] for _, k in cells], "accent"), ("MR-NIRP large", [ss[k]["MR-NIRP large"] - base["MR-NIRP large"] for _, k in cells], "teal")], ylabel="Δ MAE (bpm)", zero=True))
    return dict(title="Fig3_results", items=I)



# ======================= Table slide: compact cross-dataset MAE =======================
def tab1():
    I = []; cross = T["cross"]
    cols = [("VIPL-HR", "VIPL"), ("MMPD 7–33", "MMPD"), ("PURE 07–10", "PURE_HO"), ("UBFC 42–49", "REALTEST"), ("NIRP still", "NIRP_STILL"), ("NIRP small", "NIRP_SMALL"), ("NIRP large", "NIRP_LARGE"), ("DLCN", "DLCN"), ("MCD-A", "MCD_A"), ("MCD-B", "MCD_B"), ("RTNI flicker", "RTNI"), ("RTNC", "RTNC")]
    models = ["POS [classical]", "PhysNet", "EfficientPhys", "PhysFormer", "RhythmFormer", "FactorizePhys", "PhysFormer + PPG-injected corpus", "FactorizePhys + hybrid corpus, decomposition loss", "SpecPhys-B (ours): FactorizePhys + hybrid corpus + spectral guidance", "SpecPhys (ours, final): SpecPhys-L + noise-subspace projection"]
    short = {"SpecPhys (ours, final): SpecPhys-L + noise-subspace projection": "SpecPhys (ours, final, 10 seeds)", "POS [classical]": "POS (classical)", "PhysNet": "PhysNet (public)", "EfficientPhys": "EfficientPhys (public)", "PhysFormer": "PhysFormer (public)", "RhythmFormer": "RhythmFormer (public)", "FactorizePhys": "FactorizePhys (public)", "PhysFormer + PPG-injected corpus": "PhysFormer + corpus (5 seeds)", "FactorizePhys + hybrid corpus, decomposition loss": "FactorizePhys + corpus, decomposition loss (3)", "SpecPhys-B (ours): FactorizePhys + hybrid corpus + spectral guidance": "SpecPhys-B (ours, 3 seeds)"}
    header = ["Model"] + [c for c, _ in cols]; rows = []
    for m in models:
        r = [short[m]]
        for _, ax in cols:
            mt = cross.get(ax, {}).get(m); r.append("–" if not mt else f"{mt['mae']:.2f}")
        rows.append(r)
    # bold best per column
    best = []
    for j, (_, ax) in enumerate(cols):
        vals = [(float(r[j + 1]), i) for i, r in enumerate(rows) if r[j + 1] != "–"]; best.append(min(vals)[1] if vals else -1)
    I.append(txt(0.3, 0.25, 12.7, 0.4, "Table 2 (compact). Zero-shot heart-rate MAE (bpm) per evaluation set — lower is better; best per column in bold", fs=12, bold=True))
    I.append(it("table", x=0.3, y=0.75, w=12.73, h=4.6, header=header, rows=rows, best=best, fs=9))
    I.append(txt(0.3, 5.5, 12.7, 1.2, "Public weights = rPPG-Toolbox UBFC-rPPG checkpoints (training data differs). PURE restricted to held-out subjects 07–10; MMPD to held-out subjects 7–33; VIPL-HR uses the device HR ground truth. Multi-seed models: mean over seeds. Full tables with RMSE, MAPE, correlation and standard errors: results/paper_tables_nf.md.", fs=9, italic=True, color="muted"))
    return dict(title="Table2_compact", items=I)

# ======================= matplotlib renderer =======================
def hexc(k): return "#" + C.get(k, k)
def parse_runs(s):
    """'a_{b}c' → mathtext-ish; keep simple: convert _{x} to $_{x}$ via mathtext for matplotlib"""
    out = ""; i = 0
    while i < len(s):
        if s[i] == "_" and i + 1 < len(s) and s[i + 1] == "{":
            j = s.index("}", i); out += "$_{\\mathrm{" + s[i + 2:j].replace(" ", "\\ ") + "}}$"; i = j + 1
        else: out += s[i]; i += 1
    return out
def render_mpl(fig, path):
    W, H = fig.get("W", 13.33), fig.get("H", 7.5)   # [58 문서] 그림별 캔버스(티저 13.33×4.3)
    f = plt.figure(figsize=(W, H), dpi=200); ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    for o in fig["items"]:
        k = o["kind"]
        if k == "panel":
            ax.add_patch(FancyBboxPatch((o["x"], o["y"]), o["w"], o["h"], boxstyle="round,pad=0,rounding_size=0.12", fc=hexc(o["fill"]), ec=hexc("gray") if not o["dash"] else hexc("ink"), lw=1.0, ls="--" if o["dash"] else "-"))
            ax.text(o["x"] + 0.15, o["y"] + o["h"] - 0.12, o["label"], fontsize=11.5, fontweight="bold", color=hexc("ink"), va="bottom", ha="left", family=FONT)
        elif k == "rect":
            ax.add_patch(Rectangle((o["x"], o["y"]), o["w"], o["h"], fc=hexc(o["fill"]), ec=hexc(o["border"]), lw=0.8))
        elif k == "box":
            ax.add_patch(FancyBboxPatch((o["x"], o["y"]), o["w"], o["h"], boxstyle="round,pad=0,rounding_size=0.08", fc=hexc(o["fill"]), ec=hexc(o["border"] or "gray"), lw=0.8))
            ax.text(o["x"] + o["w"] / 2, o["y"] + o["h"] / 2, parse_runs(o["text"]), fontsize=o["fs"] * 0.78, fontweight="bold" if o["bold"] else "normal", ha="center", va="center", color=hexc("ink"), family=FONT, linespacing=1.15)
        elif k == "text":
            import textwrap
            cpl = max(8, int(o["w"] / (0.52 * o["fs"] * 0.78 / 72)))
            wrapped = "\n".join(textwrap.fill(ln, cpl) if ln else "" for ln in o["text"].split("\n"))
            ax.text(o["x"] + (o["w"] / 2 if o["align"] == "center" else 0.03), o["y"] + 0.05, parse_runs(wrapped), fontsize=o["fs"] * 0.78, fontweight="bold" if o["bold"] else "normal", style="italic" if o["italic"] else "normal", ha=o["align"], va="top", color=hexc(o["color"]), family=FONT, wrap=True, linespacing=1.2)
        elif k == "circle":
            ax.add_patch(Ellipse((o["x"] + o["d"] / 2, o["y"] + o["d"] / 2), o["d"], o["d"], fc=hexc(o["fill"]), ec=hexc("ink"), lw=0.6))
        elif k in ("arrow", "line", "dblarrow"):
            st = "-|>" if k == "arrow" else ("<|-|>" if k == "dblarrow" else "-")
            ax.add_patch(FancyArrowPatch((o["x1"], o["y1"]), (o["x2"], o["y2"]), arrowstyle=st, mutation_scale=10, lw=o.get("width", 1.2), color=hexc(o["color"]), ls="--" if o["dash"] else "-"))
            if o.get("label"): ax.text((o["x1"] + o["x2"]) / 2, o["y1"] - 0.08, parse_runs(o["label"]), fontsize=8, ha="center", va="bottom", color=hexc(o["color"]), family=FONT)
        elif k == "axis":
            x0, y0, w, h = o["x"], o["y"], o["w"], o["h"]; ax.add_patch(Rectangle((x0, y0), w, h, fc="white", ec=hexc("gray"), lw=0.6))
            f0, f1 = 30, 200; fx = lambda f: x0 + 0.15 + (f - f0) / (f1 - f0) * (w - 0.3)
            ax.add_patch(Rectangle((fx(o["hr"] - o["band"]), y0 + 0.15), fx(o["hr"] + o["band"]) - fx(o["hr"] - o["band"]), h - 0.45, fc=hexc("tokY"), ec="none", alpha=0.6))
            fs = np.linspace(f0, f1, 400); yv = np.zeros_like(fs)
            for pf, pa in o["peaks"]: yv += pa * np.exp(-0.5 * ((fs - pf) / 4.0) ** 2)
            yv += 0.05; ax.plot([fx(v) for v in fs], y0 + h - 0.3 - yv / (yv.max() + 1e-9) * (h - 0.5), color=hexc("accent"), lw=1.2)
            ax.plot([x0 + 0.15, x0 + w - 0.15], [y0 + h - 0.3] * 2, color=hexc("ink"), lw=0.8)
            for fq in (30, 60, 120, 200): ax.text(fx(fq), y0 + h - 0.26, str(fq), fontsize=6.5, ha="center", va="top", family=FONT, color=hexc("muted"))
            ax.text(fx(o["hr"]), y0 + 0.12, "HR", fontsize=7, ha="center", va="bottom", family=FONT, color=hexc("ink"))
            ax.text(x0 + w / 2, y0 + h + 0.02, parse_runs(o["label"]), fontsize=7.5, ha="center", va="top", family=FONT, color=hexc("muted"))
        elif k == "grid":
            n = o["n"]; cw = o["w"] / n; rng = np.random.default_rng(3 if o.get("kind2") == "skin" else 1)
            for i in range(n):
                for j in range(n):
                    if o.get("kind2") == "skin":
                        v = 1.0 if (abs(i - n / 2 + 0.5) < n * 0.33 and abs(j - n / 2 + 0.5) < n * 0.4) else 0.15
                    else:
                        v = float(np.exp(-0.5 * (((i - n / 2 + 0.5) / (n * 0.28)) ** 2 + ((j - n / 2 + 0.2) / (n * 0.3)) ** 2))) * (0.7 + 0.3 * rng.random())
                    ax.add_patch(Rectangle((o["x"] + j * cw, o["y"] + i * cw), cw, cw, fc=(0.95 - 0.6 * v, 0.95 - 0.25 * v, 0.6 + 0.2 * (1 - v)) if o.get("kind2") != "skin" else (1 - 0.5 * v, 1 - 0.25 * v, 1 - 0.1 * v), ec="white", lw=0.5))
            ax.text(o["x"] + o["w"] / 2, o["y"] + o["h"] + 0.03, parse_runs(o["label"]), fontsize=7.5, ha="center", va="top", family=FONT, color=hexc("muted"), wrap=True)
        elif k == "table":
            nr = len(o["rows"]) + 1; nc = len(o["header"]); rh = o["h"] / nr; cw0 = o["w"] * 0.26; cw = (o["w"] - cw0) / (nc - 1)
            for ri in range(nr):
                vals = o["header"] if ri == 0 else o["rows"][ri - 1]
                for ci, v in enumerate(vals):
                    x = o["x"] + (0 if ci == 0 else cw0 + (ci - 1) * cw); wdt = cw0 if ci == 0 else cw
                    ax.add_patch(Rectangle((x, o["y"] + ri * rh), wdt, rh, fc=hexc("lav") if ri == 0 else (hexc("F6F8FC") if ri % 2 else "#FFFFFF"), ec=hexc("gray"), lw=0.4))
                    bold = ri == 0 or (ci > 0 and o["best"][ci - 1] == ri - 1)
                    ax.text(x + (0.06 if ci == 0 else wdt / 2), o["y"] + ri * rh + rh / 2, str(v), fontsize=o["fs"] * 0.78, fontweight="bold" if bold else "normal", ha="left" if ci == 0 else "center", va="center", family=FONT, color=hexc("ink"))
        elif k == "chart":
            cx = f.add_axes([(o["x"] + 0.55) / W, 1 - (o["y"] + o["h"] - 0.55) / H, (o["w"] - 0.7) / W, (o["h"] - 0.85) / H])
            cats = o["cats"]; ns = len(o["series"]); bw = 0.8 / ns; xs = np.arange(len(cats))
            _all = [v for _, vs, _ in o["series"] for v in vs if v is not None]; off = 0.012 * (max(_all) - min(0, min(_all)))   # 값 범위에 비례한 라벨 간격(작은 범위 차트에서 라벨이 축 밖으로 나가지 않게)
            for si, (nm, vals, col) in enumerate(o["series"]):
                vals = [0 if v is None else v for v in vals]; bars = cx.bar(xs + (si - (ns - 1) / 2) * bw, vals, bw * 0.92, label=nm, color=hexc(col), edgecolor="white", linewidth=0.4)
                for b, v in zip(bars, vals): cx.text(b.get_x() + b.get_width() / 2, v + (off if v >= 0 else -off), f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=6.2, family=FONT, color=hexc("ink"))
            if min(_all) < 0: cx.set_ylim(bottom=min(_all) - 9 * off)
            if o.get("zero"): cx.axhline(0, color=hexc("ink"), lw=0.8)
            cx.set_xticks(xs); cx.set_xticklabels(cats, fontsize=7.5, family=FONT); cx.tick_params(axis="y", labelsize=7.5); cx.set_ylabel(o["ylabel"], fontsize=8, family=FONT)
            for sp in ("top", "right"): cx.spines[sp].set_visible(False)
            cx.grid(axis="y", color=hexc("gray"), lw=0.4, alpha=0.6); cx.set_axisbelow(True); cx.legend(fontsize=7, frameon=False, ncol=ns if ns <= 3 else 2, loc="upper left")
    f.savefig(path + ".png", dpi=200, facecolor="white"); f.savefig(path + ".pdf", facecolor="white"); plt.close(f)


if __name__ == "__main__":
    spec = {"W": W, "H": H, "colors": C, "font": FONT, "figures": []}
    for fn in (fig0, fig1, fig2, fig3, tab1):
        fg = fn(); spec["figures"].append(fg); render_mpl(fg, f"{OUT}/{fg['title']}"); print("rendered", fg["title"], len(fg["items"]), "items")
    json.dump(spec, open(f"{OUT}/fig_spec.json", "w"), indent=0, ensure_ascii=False); print("->", OUT)
