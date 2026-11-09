#!/usr/bin/env python3
"""개요도 5종을 편집 가능한 네이티브 도형으로 재구성한 PPT 생성.
박스·화살표·텍스트는 전부 PowerPoint 도형(선택·이동·수정 가능),
얼굴/파형/스펙트럼만 그림으로 삽입."""
from pptx import Presentation
from pptx.util import Inches as IN, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn

S = "/tmp/claude-1000/-home-user-Desktop-Virtual-Face/5f46862b-4884-4c14-9c4d-2201c79585ec/scratchpad"
OUT = "/SPECPHYS_ROOT/overview_figures.pptx"

EDGE = RGBColor(0x5B, 0x67, 0x70); BOX = RGBColor(0xF4, 0xF6, 0xF8)
ACC = RGBColor(0x1F, 0x6F, 0x43); RED = RGBColor(0xB2, 0x43, 0x2F)
BLUE = RGBColor(0x2B, 0x5D, 0x8A); AMB = RGBColor(0xB0, 0x7D, 0x2E)
INK = RGBColor(0x22, 0x22, 0x22); GREY = RGBColor(0x55, 0x55, 0x55)
G_BG = RGBColor(0xE9, 0xF2, 0xEC); R_BG = RGBColor(0xF6, 0xE9, 0xE7)
A_BG = RGBColor(0xF7, 0xF1, 0xE6); B_BG = RGBColor(0xDD, 0xE8, 0xF3)

prs = Presentation()
prs.slide_width = IN(13.333); prs.slide_height = IN(7.5)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def rrect(sl, x, y, w, h, fill, line, lw=1.5, dash=None):
    sh = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, IN(x), IN(y), IN(w), IN(h))
    sh.adjustments[0] = 0.07
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line; sh.line.width = Pt(lw)
    if dash:
        d = sh.line._get_or_add_ln(); e = d.makeelement(qn('a:prstDash'), {'val': 'dash'}); d.append(e)
    sh.shadow.inherit = False
    return sh


def text(sl, x, y, w, h, s, size=11, bold=False, color=INK, align=PP_ALIGN.CENTER,
         italic=False):
    tb = sl.shapes.add_textbox(IN(x), IN(y), IN(w), IN(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = s.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = ln
        f = r.font; f.size = Pt(size); f.bold = bold; f.italic = italic
        f.color.rgb = color; f.name = "Arial"
    return tb


def arrow(sl, x0, y0, x1, y1, color=EDGE, w=2.5):
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, IN(x0), IN(y0), IN(x1), IN(y1))
    c.line.color.rgb = color; c.line.width = Pt(w)
    ln = c.line._get_or_add_ln()
    e = ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'})
    ln.append(e)
    return c


def pic(sl, path, x, y, w, h=None):
    return sl.shapes.add_picture(path, IN(x), IN(y), IN(w), IN(h) if h else None)


def title(sl, s, sub=""):
    text(sl, 0.4, 0.18, 12.5, 0.5, s, size=20, bold=True)
    if sub:
        text(sl, 0.4, 0.62, 12.5, 0.35, sub, size=11, color=GREY, italic=True)


def panel(sl, x, y, w, h, t, sub=""):
    rrect(sl, x, y, w, h, BOX, EDGE, 1.2)
    text(sl, x + 0.1, y + 0.12, w - 0.2, 0.35, t, size=13, bold=True)
    if sub:
        text(sl, x + 0.1, y + 0.44, w - 0.2, 0.3, sub, size=8.5, color=GREY, italic=True)


# ─────────────────────────────── 표지 ───────────────────────────────
sl = slide()
text(sl, 0.8, 1.1, 11.7, 0.8, "Overview Figure — 5 Versions", size=32, bold=True)
text(sl, 0.8, 1.9, 11.7, 0.4, "Controllable Noise Injection and Denoising-Style Fine-Tuning for Robust rPPG on Virtual Faces",
     size=13, color=GREY, italic=True)
items = [("V1  Pipeline", "생성 → 주입 → 디노이징 학습 → 제로샷 검증 — 전체 흐름", BLUE),
         ("V2  Every factor is a knob", "노이즈를 독립변수로 — 문제 설정의 참신성", ACC),
         ("V3  Causal attribution", "대조군/처치군 분기 — 실험 설계의 엄밀함", AMB),
         ("V4  Failure modes", "발견한 실패 3종과 일괄 수리 — 진단 도구 가치", RED),
         ("V5  Sim-to-real story", "제로샷 성공 → 균열 → 진단 → 해소 — 4막 서사", EDGE)]
for i, (t, d, c) in enumerate(items):
    y = 2.75 + i * 0.82
    rrect(sl, 0.9, y, 11.5, 0.68, RGBColor(0xFF, 0xFF, 0xFF), c, 1.4)
    text(sl, 1.2, y + 0.1, 3.6, 0.4, t, size=14, bold=True, color=c, align=PP_ALIGN.LEFT)
    text(sl, 5.0, y + 0.14, 7.2, 0.4, d, size=11, color=GREY, align=PP_ALIGN.LEFT)
text(sl, 0.8, 7.0, 11.7, 0.35, "모든 도형·텍스트·화살표는 개별 편집 가능 / 모든 수치는 실제 실험값", size=9.5, color=GREY)

# ─────────────────────────────── V1 파이프라인 ───────────────────────────────
sl = slide()
title(sl, "V1 — Pipeline: virtual faces → pulse injection → denoising training → zero-shot validation")
Y, H = 1.15, 5.3
panel(sl, 0.25, Y, 3.0, H, "1. Virtual-face videos", "many non-existent identities (SFHQ / SDXL-Turbo) + LivePortrait")
for i in range(6):  # 정체성 6명 그리드 (2행 x 3열)
    r, c = divmod(i, 3)
    pic(sl, f"{S}/id_{i}.png", 0.45 + c * 0.92, Y + 0.85 + r * 0.92, 0.82, 0.82)
text(sl, 0.35, Y + 2.72, 2.85, 0.28, "synthetic identities (no real person)", size=8, color=GREY)
text(sl, 0.35, Y + 3.05, 2.85, 0.28, "animate each: 30 fps · 34 s · lossless", size=8, color=GREY)
for i in range(3):  # 서로 다른 3명의 애니메이션 프레임
    pic(sl, f"{S}/anim_{i}.png", 0.45 + i * 0.92, Y + 3.4, 0.82, 0.82)
text(sl, 0.35, Y + 4.25, 2.85, 0.28, "animated frames (t = 15 s)", size=7.5, color=GREY)
arrow(sl, 3.3, Y + 2.6, 3.62, Y + 2.6)
panel(sl, 3.7, Y, 3.35, H, "2. Pixel-wise injection", "known GT + controlled corruption")
pic(sl, f"{S}/ov_0.png", 3.95, Y + 0.95, 1.25, 1.25)
pic(sl, f"{S}/ov_1.png", 5.45, Y + 0.95, 1.25, 1.25)
text(sl, 3.8, Y + 2.25, 3.0, 0.3, "per-identity amplitude maps A(x,y) — patchy perfusion", size=7.5, color=GREY)
pic(sl, f"{S}/corr_1.png", 4.7, Y + 2.62, 0.75, 0.75)
text(sl, 5.5, Y + 2.75, 1.5, 0.5, "corrupted frame\n(illum+motion+sensor)", size=7.5, color=GREY)
pic(sl, f"{S}/w_gt.png", 3.95, Y + 3.5, 2.85, 0.65)
text(sl, 3.85, Y + 4.15, 3.05, 0.28, "injected pulse p(t) = ground truth (per-identity BPM)", size=8, color=ACC)
text(sl, 3.8, Y + 4.55, 3.15, 0.4, "Ic ← Ic (1 + α·A(x,y)·p(t)·Pbvc)", size=10.5, bold=True)
arrow(sl, 7.1, Y + 2.6, 7.42, Y + 2.6)
panel(sl, 7.5, Y, 3.1, H, "3. Denoising fine-tuning", "corrupted input, clean pulse target")
rrect(sl, 7.7, Y + 1.2, 1.05, 0.8, RGBColor(0xFF, 0xFF, 0xFF), EDGE, 1.2)
text(sl, 7.7, Y + 1.42, 1.05, 0.5, "corrupted\nvideo", size=8.5)
rrect(sl, 9.0, Y + 1.2, 1.35, 0.8, B_BG, BLUE, 1.6)
text(sl, 9.0, Y + 1.47, 1.35, 0.4, "PhysFormer", size=10, bold=True, color=BLUE)
arrow(sl, 8.78, Y + 1.6, 8.98, Y + 1.6)
pic(sl, f"{S}/w_pred.png", 9.05, Y + 2.45, 1.3, 0.6)
text(sl, 8.95, Y + 3.05, 1.5, 0.25, "predicted", size=8, color=BLUE)
pic(sl, f"{S}/w_gt_small.png", 9.05, Y + 3.55, 1.3, 0.6)
text(sl, 8.85, Y + 4.15, 1.7, 0.25, "clean ground truth", size=8, color=ACC)
arrow(sl, 9.7, Y + 2.4, 9.7, Y + 2.1, RED, 1.8)
text(sl, 7.6, Y + 2.55, 1.3, 0.9, "loss pulls\nprediction to\nclean target", size=8, color=RED)
arrow(sl, 10.65, Y + 2.6, 10.97, Y + 2.6)
panel(sl, 11.05, Y, 2.05, H, "4. Zero-shot real", "no real face seen in training")
rows = [("UBFC-rPPG 42 subj", "1.78 BPM · r 0.974", ACC, False),
        ("real + flicker", "30.32 → 0.00*", ACC, False),
        ("PURE", "awaiting access", EDGE, True),
        ("MMPD", "awaiting access", EDGE, True)]
for i, (n, r, c, dash) in enumerate(rows):
    yy = Y + 0.85 + i * 1.05
    rrect(sl, 11.2, yy, 1.75, 0.85, RGBColor(0xFF, 0xFF, 0xFF), c, 1.2, dash=dash)
    text(sl, 11.25, yy + 0.08, 1.65, 0.35, n, size=8.5, bold=True, align=PP_ALIGN.LEFT)
    text(sl, 11.25, yy + 0.42, 1.65, 0.35, r, size=8, color=c if not dash else GREY, align=PP_ALIGN.LEFT)
text(sl, 10.0, Y + 5.32, 3.2, 0.3, "*dual-domain noise co-training", size=7.5, color=GREY)

# ─────────────────────────────── V2 노브 ───────────────────────────────
sl = slide()
title(sl, "V2 — Noise as an independent variable, not an augmentation trick")
Y, H = 1.15, 5.3
rrect(sl, 0.25, Y, 4.1, H, R_BG, RED, 1.4)
text(sl, 0.35, Y + 0.12, 3.9, 0.35, "A real recording entangles everything", size=13, bold=True)
text(sl, 0.35, Y + 0.44, 3.9, 0.3, "no noise-free counterfactual exists", size=8.5, color=GREY, italic=True)
pic(sl, f"{S}/p_face.png", 0.55, Y + 1.1, 1.5, 1.5)
for i, q in enumerate(["illumination?", "head motion?", "perfusion geometry?", "sensor noise?"]):
    yy = Y + 1.0 + i * 0.78
    rrect(sl, 2.3, yy, 1.85, 0.55, RGBColor(0xFF, 0xFF, 0xFF), RED, 1.0, dash=True)
    text(sl, 2.3, yy + 0.14, 1.85, 0.3, q, size=9, color=RED)
text(sl, 0.35, Y + 4.65, 3.9, 0.4, "which nuisance caused the error? — unanswerable", size=9, color=RED, italic=True)
arrow(sl, 4.42, Y + 2.6, 4.75, Y + 2.6)
panel(sl, 4.82, Y, 4.6, H, "Our testbed: every factor is a knob", "many virtual identities + pulse injected by construction")
pic(sl, f"{S}/ov_0.png", 5.0, Y + 1.05, 1.15, 1.15)
pic(sl, f"{S}/ov_1.png", 5.0, Y + 2.35, 1.15, 1.15)
text(sl, 4.9, Y + 3.55, 1.55, 0.75, "pulse placed in the\npixels of every identity\n— A(x,y) per face", size=8, color=GREY)
knobs = [("illumination drift / flicker", 0.55), ("rigid head motion", 0.75),
         ("sensor noise", 0.35), ("perfusion coverage", 0.62)]
for i, (n, frac) in enumerate(knobs):
    yy = Y + 1.0 + i * 0.85
    text(sl, 6.7, yy, 2.5, 0.28, n, size=9, align=PP_ALIGN.LEFT)
    rrect(sl, 6.7, yy + 0.3, 2.4, 0.12, RGBColor(0xDD, 0xDD, 0xDD), RGBColor(0xDD, 0xDD, 0xDD), 0.5)
    rrect(sl, 6.7, yy + 0.3, 2.4 * frac, 0.12, BLUE, BLUE, 0.5)
    kn = sl.shapes.add_shape(MSO_SHAPE.OVAL, IN(6.7 + 2.4 * frac - 0.09), IN(yy + 0.27), IN(0.18), IN(0.18))
    kn.fill.solid(); kn.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    kn.line.color.rgb = BLUE; kn.line.width = Pt(1.5); kn.shadow.inherit = False
text(sl, 4.95, Y + 4.6, 4.35, 0.5, "ground truth known exactly →\nevery condition has its counterfactual", size=9, color=ACC)
arrow(sl, 9.5, Y + 2.6, 9.82, Y + 2.6)
panel(sl, 9.9, Y, 3.2, H, "What this unlocks", "causal questions become measurable")
qs = [("decompose degradation\nby nuisance type & strength", BLUE),
      ("attribute recovery: noise learning\nvs domain adaptation", ACC),
      ("audit losses & find shortcuts\ninvisible in real data", AMB)]
for i, (q, c) in enumerate(qs):
    yy = Y + 0.95 + i * 1.15
    rrect(sl, 10.1, yy, 2.8, 0.9, RGBColor(0xFF, 0xFF, 0xFF), c, 1.4)
    text(sl, 10.15, yy + 0.16, 2.7, 0.6, q, size=8.5)
text(sl, 9.95, Y + 4.55, 3.1, 0.6, "noise: an independent variable,\nnot an augmentation trick", size=9.5, bold=True)

# ─────────────────────────────── V3 인과 귀인 ───────────────────────────────
sl = slide()
title(sl, "V3 — Causal attribution: which mechanism creates robustness?")
rrect(sl, 0.35, 3.1, 2.1, 1.0, B_BG, BLUE, 1.6)
text(sl, 0.35, 3.3, 2.1, 0.6, "pretrained\nPhysFormer", size=11, bold=True, color=BLUE)
text(sl, 0.25, 4.2, 2.3, 0.6, "identical identities, beat\nrates, seeds in both branches", size=8, color=GREY)
arrow(sl, 2.5, 3.35, 3.35, 2.35, AMB)
arrow(sl, 2.5, 3.85, 3.35, 4.85, ACC)
rrect(sl, 3.45, 1.7, 3.4, 1.25, A_BG, AMB, 1.5)
text(sl, 3.55, 1.85, 3.2, 0.35, "CONTROL: fine-tune on clean synthetic", size=10, bold=True, color=AMB)
text(sl, 3.55, 2.25, 3.2, 0.55, "learns the domain,\nnever sees a nuisance", size=8.5, color=GREY)
rrect(sl, 3.45, 4.3, 3.4, 1.25, G_BG, ACC, 1.5)
text(sl, 3.55, 4.45, 3.2, 0.35, "TREATMENT: fine-tune on noisy synthetic", size=10, bold=True, color=ACC)
text(sl, 3.55, 4.85, 3.2, 0.55, "denoising supervision:\ncorrupted input, clean pulse target", size=8.5, color=GREY)
arrow(sl, 7.0, 3.6, 7.35, 3.6)
text(sl, 8.2, 1.0, 4.5, 0.35, "held-out evaluation (HR MAE, BPM)", size=12, bold=True)
cols = ["condition", "pretrained", "clean-only\ncontrol", "noise-\ntrained"]
for j, c in enumerate(cols):
    text(sl, 7.55 + j * 1.45, 1.45, 1.4, 0.5, c, size=9, bold=True)
rows = [("head motion", "33.56", "0.09", "0.15"),
        ("patchy perfusion", "32.22", "0.09", "16.10"),
        ("illumination flicker", "10.54", "10.48", "0.14"),
        ("stronger illumination", "16.11", "27.57", "0.09")]
for i, (name, *vals) in enumerate(rows):
    yy = 2.1 + i * 0.85
    text(sl, 7.45, yy + 0.15, 1.6, 0.5, name, size=8.5)
    for j, v in enumerate(vals):
        good = float(v) < 1.0
        c = ACC if good else RED
        rrect(sl, 9.15 + j * 1.45, yy, 1.15, 0.6, G_BG if good else R_BG, c, 1.1)
        text(sl, 9.15 + j * 1.45, yy + 0.16, 1.15, 0.3, v, size=9.5, bold=True, color=c)
rrect(sl, 7.45, 5.85, 5.5, 1.0, G_BG, ACC, 1.8)
text(sl, 7.6, 6.0, 5.2, 0.75,
     "control repairs motion & perfusion (= domain adaptation)\nbut inherits the illumination failure\n→ illumination robustness is CAUSED by noise learning",
     size=9, bold=True, color=RGBColor(0x1A, 0x4D, 0x30))

# ─────────────────────────────── V4 실패 모드 ───────────────────────────────
sl = slide()
title(sl, "V4 — The testbed exposes how rPPG fails, and shows the failures are learnable")
cards = [(0.3, "Failure 1: flicker lock-in", "1.1 Hz ambient flicker inside the cardiac band"),
         (4.6, "Failure 2: harmonic doubling", "the 2nd harmonic outguns the fundamental"),
         (8.9, "Failure 3: patchy perfusion", "nose & fatty regions barely pulse")]
for x, t, s in cards:
    rrect(sl, x, 1.2, 4.1, 4.3, R_BG, RED, 1.4)
    text(sl, x + 0.1, 1.35, 3.9, 0.35, t, size=12.5, bold=True, color=RED)
    text(sl, x + 0.1, 1.7, 3.9, 0.3, s, size=8.5, color=GREY, italic=True)
pic(sl, f"{S}/spec_flicker.png", 0.6, 2.3, 3.5)
text(sl, 0.5, 4.6, 3.7, 0.7, "model reads 66 BPM (= the light!)\ntrue pulse: 77 BPM (dashed)", size=9, color=RED)
pic(sl, f"{S}/spec_harm.png", 4.9, 2.3, 3.5)
text(sl, 4.8, 4.6, 3.7, 0.7, "true 64 BPM → reported 128\n(192 under patchy perfusion)", size=9, color=RED)
pic(sl, f"{S}/ov_0.png", 9.25, 2.25, 1.35, 1.35)
pic(sl, f"{S}/ov_1.png", 10.7, 2.25, 1.35, 1.35)
text(sl, 12.1, 2.5, 0.85, 1.2, "doubles\nthe\nerror:", size=9)
text(sl, 9.2, 4.15, 3.6, 0.4, "16.16 → 32.22 BPM", size=13, bold=True, color=RED)
text(sl, 9.2, 4.6, 3.6, 0.6, "(spatial amplitude map A(x,y)\nmakes this a controllable factor)", size=8, color=GREY)
rrect(sl, 0.3, 5.85, 12.7, 0.95, G_BG, ACC, 1.8)
text(sl, 0.5, 6.05, 12.3, 0.6,
     "denoising-style training repairs all three:  noisy-condition mean 16.32 → 0.13 BPM   |   SNR −2.7 → +6…+10 dB   |   extrapolates to unseen noise types & strengths",
     size=10.5, bold=True, color=RGBColor(0x1A, 0x4D, 0x30))

# ─────────────────────────────── V5 sim-to-real ───────────────────────────────
sl = slide()
title(sl, "V5 — Sim-to-real: zero-shot success, a hidden crack, and the dual-domain fix")
beats = [(0.3, "1. zero-shot to real faces", "1.78 BPM", "train: synthetic only\ntest: UBFC-rPPG, 42 real subjects\nPearson 0.974", ACC, G_BG),
         (3.55, "2. stress test finds a crack", "30.32 BPM", "real video + controlled flicker:\nthe co-trained model collapses\nexactly like the baseline", RED, R_BG),
         (6.8, "3. diagnosis: hidden shortcut", "clean real data", "the model learned \"trust\nillumination on real faces\" —\ninvariance was domain-conditional", AMB, A_BG),
         (10.05, "4. dual-domain denoising fix", "0.00 BPM", "corrupt BOTH domains, keep clean\ntargets: flicker repaired, clean 1.43\nkept, synthetic mean 0.10", ACC, G_BG)]
for x, t, big, sub, c, bg in beats:
    rrect(sl, x, 1.6, 2.95, 4.1, bg, c, 1.6)
    text(sl, x + 0.08, 1.8, 2.8, 0.6, t, size=11, bold=True)
    text(sl, x + 0.08, 3.0, 2.8, 0.5, big, size=17, bold=True, color=c)
    text(sl, x + 0.08, 4.1, 2.8, 1.3, sub, size=8.5, color=GREY)
# 1막에 가상 정체성 스트립 (합성만으로 학습했음을 시각화)
for i in range(4):
    pic(sl, f"{S}/id_{i}.png", 0.52 + i * 0.63, 2.42, 0.55, 0.55)
for x0 in (3.27, 6.52, 9.77):
    arrow(sl, x0, 3.65, x0 + 0.26, 3.65)
text(sl, 0.5, 6.1, 12.3, 0.5,
     "a failure mode invisible to any evaluation that leaves real video uncorrupted —\nmade visible, explained, and eliminated by the controlled testbed",
     size=10, color=GREY, italic=True)

prs.save(OUT)
print("saved:", OUT)
print("slides:", len(prs.slides.__iter__.__self__._sldIdLst))
