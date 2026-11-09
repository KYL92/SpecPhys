// [63 문서] SpecPhys 구조 그림 4장: 편집 가능한 PPT 도형(참고 그림: FactorizePhys·EfficientPhys 논문 그림 형식).
// 실행: cd scripts && NODE_PATH=$CLAUDE_JOB_DIR/tmp/node/node_modules node paper_arch_figs.js [out.pptx]
//   → python scripts/paper_arch_figs_group.py out.pptx (도형 묶기) → _workspace/slides/specphys_architecture_figs.pptx
// 재료: python scripts/paper_arch_figs_data.py → results/figs_arch/ (실제 체크포인트 출력, 얼굴은 생성 얼굴만). 수치: results/final_config_tables.json.
const pptxgen = require("pptxgenjs"); const fs = require("fs");
const ROOT = "/SPECPHYS_ROOT", RES = `${ROOT}/_workspace/data/results`, A = `${RES}/figs_arch`;
const SKILL = "/path/to/pptx-skill";
const { applyTheme } = require(`${SKILL}/scripts/apply_theme.js`);
const D = JSON.parse(fs.readFileSync(`${A}/arch_data.json`)); const FT = JSON.parse(fs.readFileSync(`${RES}/final_config_tables.json`));
const OUTF = process.argv[2] || `${ROOT}/_workspace/slides/specphys_architecture_figs.pptx`;
const THEME = { name: "SpecPhys figures", headFontFace: "Arial", bodyFontFace: "Arial", colors: { dk1: "1F2328", lt1: "FFFFFF", dk2: "2F3C7E", lt2: "EEF3FB", accent1: "2F3C7E", accent2: "C0392B", accent3: "D9822B", accent4: "1F7A8C", accent5: "A98A81", accent6: "6B7280", hlink: "2F3C7E", folHlink: "6B7280" } };
const pres = new pptxgen(); pres.layout = "LAYOUT_WIDE"; pres.title = "SpecPhys architecture figures"; pres.author = "SpecPhys"; pres.subject = "Architecture figures"; pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };
const S = pres.shapes; const FONT = "Arial", MATH = "Times New Roman";
// 색: 주황 = 잡음, 빨강 = 맥파, 남색 = 이 연구에서 더한 부분, 회갈색 = 공개 모델과 같은 몸통
const K = { ink: "1F2328", mute: "5B6470", line: "2B2F36", gray: "BFC5CE", gray2: "E4E7EC", gray3: "F3F4F6", ours: "2F3C7E", oursF: "DCE6F5", oursT: "EEF3FB", noise: "D9822B", noiseF: "FBEBD9", pulse: "C0392B", pulseF: "F9E3E0", teal: "1F7A8C", front: "EFDCD6", top: "E2C9C1", side: "CFB0A7", edge: "A98A81", arrow: "7A7F87", white: "FFFFFF", warm: "F6F4EE" };
let CUR = null, GRP = "", N = 0;
const nm = (n) => `${GRP ? GRP + "|" : ""}${n || "s"}${++N}`;
function runs(str, o) {   // 표기: $수식$(이탤릭 세리프), _{아래}, ^{위}, **굵게**, \n
  const out = []; let math = false, bold = !!o.b, buf = "";
  const base = () => ({ fontFace: math ? MATH : (o.font || FONT), fontSize: (o.fs || 12) + (math ? 1 : 0), color: o.color || K.ink, italic: math || !!o.i, bold });
  const flush = (extra) => { if (buf) { out.push({ text: buf, options: { ...base(), ...(extra || {}) } }); buf = ""; } };
  for (let i = 0; i < str.length; i++) {
    const c = str[i], d = str[i + 1];
    if (c === "$") { flush(); math = !math; }
    else if (c === "*" && d === "*") { flush(); bold = !bold; i++; }
    else if ((c === "_" || c === "^") && d === "{") { flush(); const j = str.indexOf("}", i); buf = str.slice(i + 2, j); flush(c === "_" ? { subscript: true } : { superscript: true }); i = j; }
    else if (c === "\n") { flush(); if (out.length) out[out.length - 1].options.breakLine = true; }
    else buf += c;
  }
  flush(); return out;
}
function txt(str, x, y, w, h, o = {}) {
  const b = { x, y, w, h }; if (o.rot) { b.x = x + w / 2 - h / 2; b.y = y + h / 2 - w / 2; b.w = h; b.h = w; b.rotate = o.rot; }
  CUR.addText(runs(str, o), { ...b, margin: 0, align: o.align || "center", valign: o.valign || "middle", isTextBox: true, objectName: nm(o.name || "text") });
}
function box(str, x, y, w, h, o = {}) {
  const opt = { shape: o.rect ? S.RECTANGLE : S.ROUNDED_RECTANGLE, x, y, w, h, margin: o.margin === undefined ? 2 : o.margin, align: o.align || "center", valign: o.valign || "middle", isTextBox: true, objectName: nm(o.name || "box"), line: o.noline ? undefined : { color: o.line || K.line, width: o.lw || 1, dashType: o.dash ? "dash" : "solid" } };
  if (o.fill !== null) opt.fill = { color: o.fill || K.white };
  if (!o.rect) opt.rectRadius = o.r === undefined ? Math.min(0.08, Math.min(w, h) * 0.2) : o.r;
  if (o.rot) { opt.x = x + w / 2 - h / 2; opt.y = y + h / 2 - w / 2; opt.w = h; opt.h = w; opt.rotate = o.rot; }
  CUR.addText(str ? runs(str, o) : "", opt);
}
function frame(x, y, w, h, o = {}) { CUR.addShape(S.RECTANGLE, { x, y, w, h, objectName: nm(o.name || "frame"), fill: o.fill ? { color: o.fill, transparency: o.alpha || 0 } : undefined, line: o.noline ? undefined : { color: o.line || K.line, width: o.lw || 1, dashType: o.dash ? "dash" : "solid" } }); }
function line(x1, y1, x2, y2, o = {}) {
  const x = Math.min(x1, x2), y = Math.min(y1, y2), w = Math.abs(x2 - x1), h = Math.abs(y2 - y1);
  CUR.addShape(S.LINE, { x, y, w, h, flipH: x2 < x1, flipV: y2 < y1, objectName: nm(o.name || "line"), line: { color: o.color || K.line, width: o.lw || 1.25, dashType: o.dash ? "dash" : "solid", endArrowType: o.head === false ? "none" : (o.head || "triangle"), beginArrowType: "none" } });
}
function seg(x1, y1, x2, y2, o = {}) { line(x1, y1, x2, y2, { head: false, ...o }); }
function poly(pts, o = {}) {   // pts: [[x,y],...] 또는 여러 경로 [[[x,y],...],...]
  const paths = Array.isArray(pts[0][0]) ? pts : [pts]; const all = paths.flat();
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]); const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const w = Math.max(x1 - x0, 0.001), h = Math.max(y1 - y0, 0.001); const points = [];
  for (const p of paths) { p.forEach((q, i) => points.push({ x: q[0] - x0, y: q[1] - y0, ...(i === 0 ? { moveTo: true } : {}) })); if (o.close) points.push({ close: true }); }
  CUR.addShape(S.CUSTOM_GEOMETRY, { x: x0, y: y0, w, h, points, objectName: nm(o.name || "poly"), fill: o.fill ? { color: o.fill, transparency: o.alpha || 0 } : undefined, line: o.noline ? undefined : { color: o.color || K.line, width: o.lw || 1.25, dashType: o.dash ? "dash" : "solid", endArrowType: o.head || "none", beginArrowType: "none" } });
}
function img(file, x, y, w, h, o = {}) { CUR.addImage({ path: `${A}/${file}`, x, y, w, h, altText: o.alt || file.replace(".png", ""), objectName: nm(o.name || "image") }); if (o.border !== false) frame(x, y, w, h, { line: o.bc || K.white, lw: o.blw || 0.75, name: "imgframe" }); }
function stack(files, x, y, s, off, o = {}) { const n = files.length; for (let i = n - 1; i >= 0; i--) img(files[i], x + off * i, y - off * i, s, s, { alt: o.alt || "Generated face frame", ...o }); }
function cuboid(x, y, w, h, dx, dy, o = {}) {  // (x,y,w,h) = 앞면; 깊이는 오른쪽 위로 (dx,-dy)
  const c = o.pal || { front: K.front, top: K.top, side: K.side, edge: K.edge }; const lw = o.lw || 0.75;
  poly([[x, y], [x + dx, y - dy], [x + w + dx, y - dy], [x + w, y]], { close: true, fill: c.top, color: c.edge, lw, name: "top" });
  poly([[x + w, y], [x + w + dx, y - dy], [x + w + dx, y + h - dy], [x + w, y + h]], { close: true, fill: c.side, color: c.edge, lw, name: "side" });
  frame(x, y, w, h, { fill: c.front, line: c.edge, lw, name: "front" });
  if (o.grid) {
    const [nw, nh, nd] = o.grid; const g = [];
    for (let i = 1; i < nw; i++) { const a = x + w * i / nw; g.push([[a, y], [a, y + h]]); g.push([[a, y], [a + dx, y - dy]]); }
    for (let i = 1; i < nh; i++) { const b = y + h * i / nh; g.push([[x, b], [x + w, b]]); g.push([[x + w, b], [x + w + dx, b - dy]]); }
    for (let i = 1; i < nd; i++) { const t = i / nd; g.push([[x + dx * t, y - dy * t], [x + w + dx * t, y - dy * t]]); g.push([[x + w + dx * t, y - dy * t], [x + w + dx * t, y + h - dy * t]]); }
    poly(g, { color: o.gridColor || c.edge, lw: o.glw || 0.4, name: "grid" });
  }
}
function wave(arr, x, y, w, h, o = {}) { const a = o.n ? arr.slice(o.from || 0, (o.from || 0) + o.n) : arr; const lo = Math.min(...a), hi = Math.max(...a); poly(a.map((v, i) => [x + w * i / (a.length - 1), y + h - h * (v - lo) / (hi - lo + 1e-9)]), { color: o.color || K.pulse, lw: o.lw || 1.5, name: o.name || "wave" }); }
function op(ch, cx, cy, d = 0.3, o = {}) { CUR.addText(runs(ch, { fs: o.fs || 14, b: true }), { shape: S.OVAL, x: cx - d / 2, y: cy - d / 2, w: d, h: d, margin: 0, align: "center", valign: "middle", isTextBox: true, fill: { color: K.white }, line: { color: K.line, width: 1.25 }, objectName: nm("op") }); }
function blockArrow(x, y, w, h, o = {}) { CUR.addShape(S.RIGHT_ARROW, { x, y, w, h, fill: { color: o.color || K.arrow }, line: { color: o.color || K.arrow, width: 0.5 }, objectName: nm("arrow") }); }
function brace(x0, x1, y, h, o = {}) { // 아래로 벌어진 중괄호: 끝점 (x0,y+h),(x1,y+h), 뾰족한 곳 ((x0+x1)/2 또는 o.tip, y)
  const r = h / 2, t = o.tip === undefined ? (x0 + x1) / 2 : o.tip; const arc = (cx, cy, a0, a1) => { const p = []; for (let i = 0; i <= 8; i++) { const a = a0 + (a1 - a0) * i / 8; p.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); } return p; };
  poly([[x0, y + h], ...arc(x0 + r, y + h, Math.PI, 1.5 * Math.PI), ...arc(t - r, y, 0.5 * Math.PI, 0), ...arc(t + r, y, Math.PI, 0.5 * Math.PI), ...arc(x1 - r, y + h, 1.5 * Math.PI, 2 * Math.PI), [x1, y + h]], { color: o.color || K.ours, lw: o.lw || 1.5, name: "brace" });
}
function ramp(x, y, s) { frame(x, y, s, s, { fill: K.white, lw: 0.75, name: "ramp" }); seg(x + s * 0.5, y + 0.03, x + s * 0.5, y + s - 0.03, { lw: 0.5, dash: true, color: K.mute }); seg(x + 0.03, y + s * 0.75, x + s - 0.03, y + s * 0.75, { lw: 0.5, dash: true, color: K.mute }); poly([[x + 0.05, y + s * 0.75], [x + s * 0.5, y + s * 0.75], [x + s - 0.05, y + 0.05]], { lw: 1.25, name: "ramp" }); }
function spectrumPlot(x, y, w, h, f, sers, o = {}) { // f(bpm), sers: [{y,color,lw}], marks: [{v,color}], bands: [{a,b,color,alpha}]
  const lo = o.lo || 30, hi = o.hi || 200; const X = (v) => x + w * (v - lo) / (hi - lo);
  const m = o.ymax || Math.max(...sers.flatMap((s) => s.y.filter((_, i) => f[i] >= lo && f[i] <= hi)));
  (o.bands || []).forEach((b) => frame(X(Math.max(b.a, lo)), y, X(Math.min(b.b, hi)) - X(Math.max(b.a, lo)), h, { fill: b.color, alpha: b.alpha === undefined ? 45 : b.alpha, noline: true, name: "band" }));
  (o.marks || []).forEach((k) => seg(X(k.v), y, X(k.v), y + h, { color: k.color, lw: 1, dash: true, name: "mark" }));
  sers.forEach((s) => { const P = []; f.forEach((v, i) => { if (v >= lo && v <= hi) P.push([X(v), y + h - h * Math.min(s.y[i] / m, 1)]); }); poly(s.closed ? [[X(lo), y + h], ...P, [X(hi), y + h]] : P, { color: s.color, lw: s.lw || 1.5, fill: s.fill, alpha: s.alpha, close: !!s.closed, noline: s.noline, name: "series" }); });
  seg(x, y + h, x + w, y + h, { color: K.line, lw: 1, name: "axis" });
  (o.ticks || [60, 120, 180]).forEach((t) => { seg(X(t), y + h, X(t), y + h + 0.04, { color: K.line, lw: 0.75, name: "tick" }); txt(`${t}`, X(t) - 0.25, y + h + 0.05, 0.5, 0.17, { fs: o.tfs || 10, color: K.mute, name: "ticklabel" }); });
  if (o.xlabel) txt(o.xlabel, x, y + h + 0.21, w, 0.18, { fs: o.tfs || 10, color: K.mute, name: "xlabel" });
}
function begin(title, notes) { pres.addSection({ title }); CUR = pres.addSlide({ sectionTitle: title }); CUR.background = { color: "FFFFFF" }; CUR.addNotes(notes); GRP = ""; }
function group(name, fn) { const g = GRP; GRP = name; fn(); GRP = g; }
const argmax = (a) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0); const peak = (f, a) => f[argmax(a)];
const LV = FT.levels, finL = LV["SpecPhys (final)"], pubL = LV["public FactorizePhys"], AX = FT["axes_public FactorizePhys"];

// 작은 파이프라인 칸
function mini(items, x, y, h, gap) { let cx = x; items.forEach((it, i) => { box(it.t, cx, y, it.w, h, { fill: it.fill || K.front, line: it.line || K.edge, fs: it.fs || 11, b: !!it.b, color: it.color, margin: 1, name: "mini" }); cx += it.w; if (i < items.length - 1) { line(cx, y + h / 2, cx + gap, y + h / 2, { lw: 1, name: "miniarrow" }); cx += gap; } }); return cx; }
// 라벨이 붙은 복셀 더미
function vox(x, y, pal, o = {}) {
  const dxx = o.dx || 0.72, dyy = o.dy || 0.72, w = o.w || 0.46, h = o.h || 0.32; const grid = [4, 3, 10]; const ys = [y + 1.58, y + 1.08, y + 0.26];  // 앞면 윗변 y (아래, 중간, 위)
  // 뒤쪽 조각부터: 위 → 점 → 중간 → 아래 (오른쪽 위로 뻗으므로 아래 조각이 가장 앞)
  cuboid(x, ys[2], w, h, dxx, dyy, { pal, grid, glw: 0.3, lw: 0.6 });
  [0, 1, 2].forEach((i) => CUR.addShape(S.OVAL, { x: x + w + 0.03 + i * 0.09, y: ys[2] + h + 0.2 - i * 0.09, w: 0.07, h: 0.07, fill: { color: K.mute }, objectName: nm("dot") }));
  cuboid(x, ys[1], w, h, dxx, dyy, { pal, grid, glw: 0.3, lw: 0.6 }); cuboid(x, ys[0], w, h, dxx, dyy, { pal, grid, glw: 0.3, lw: 0.6 });
  const q = o.labels; if (q) { txt(q[2], x + w + dxx + 0.03, ys[2] - dyy - 0.02, 0.36, 0.22, { fs: 11, align: "left" }); txt(q[1], x + w + dxx + 0.03, ys[1] - dyy + 0.1, 0.34, 0.22, { fs: 11, align: "left" }); txt(q[0], x + w + dxx + 0.03, ys[0] - dyy + 0.1, 0.34, 0.22, { fs: 11, align: "left" }); }
}

// ======================= 그림 1: 공개 체크포인트와 SpecPhys 의 높은 수준 비교 =======================
function figCompare() {
  const pB = peak(D.f, D.sp), oB = peak(D.f, D.so);
  begin("Figure 1: Public checkpoint vs. SpecPhys", `Figure 1. High-level comparison on the same input. The input is a generated-face clip that is not in the training pool, with heart rate ${D.hr} bpm and an illumination flicker at ${D.nuis} bpm, outside the 54-78 bpm flicker range of the pool. The spectra and pulse traces are real outputs on one 160-frame chunk: the public FactorizePhys weights peak at ${pB.toFixed(1)} bpm and SpecPhys (seed 1600) at ${oB.toFixed(1)} bpm. One clip, shown for illustration. Zero-shot MAE from the final tables: VIPL-HR with device labels, RTNI = real UBFC-rPPG clips with 1.1 Hz flicker; SpecPhys over ten seeds.`);
  group("input", () => {
    txt("**Video frames**", 0.15, 2.2, 1.95, 0.28, { fs: 14 });
    stack(["face_0.png", "face_1.png", "face_2.png"], 0.2, 3.0, 1.45, 0.16);
    line(0.3, 4.58, 0.76, 5.04, { lw: 1.25, name: "time" }); txt("time", 0.8, 4.78, 0.5, 0.22, { fs: 11, color: K.mute, align: "left" });
    txt("Generated face\nheart rate **" + D.hr + "** bpm\nflicker **" + D.nuis + "** bpm", 0.1, 5.14, 2.0, 0.7, { fs: 12 });
  });
  poly([[2.0, 3.6], [2.22, 3.6], [2.22, 2.05], [2.42, 2.05]], { head: "triangle", name: "in1" }); poly([[2.0, 3.6], [2.22, 3.6], [2.22, 5.73], [2.42, 5.73]], { head: "triangle", name: "in2" });
  const colA = 2.55, wA = 2.0, colB = 4.65, wB = 4.2, colC = 8.95, wC = 2.24, colD = 11.36;
  // 윗줄
  frame(2.42, 0.14, 10.78, 3.55, { dash: true, lw: 1.25, name: "outer1" });
  txt("**1.**  Public checkpoint: FactorizePhys weights trained on one dataset", 2.55, 0.2, 10.4, 0.34, { fs: 15, align: "left" });
  frame(colA, 0.6, wA, 2.95, { dash: true, lw: 0.75, name: "A1" }); txt("**A.** Training data", colA + 0.07, 0.65, wA - 0.1, 0.28, { fs: 13, align: "left" });
  group("db1", () => { CUR.addShape(S.FLOWCHART_MAGNETIC_DISK, { x: colA + 0.5, y: 1.1, w: 1.0, h: 0.98, fill: { color: K.gray2 }, line: { color: K.line, width: 1 }, objectName: nm("db") }); txt("UBFC-rPPG", colA, 2.2, wA, 0.26, { fs: 12.5, b: true }); txt("one dataset,\nwaveform loss", colA + 0.05, 2.52, wA - 0.1, 0.6, { fs: 12, color: K.mute }); });
  frame(colB, 0.6, wB, 2.95, { dash: true, lw: 0.75, name: "B1" }); txt("**B.** Model", colB + 0.07, 0.65, wB - 0.1, 0.28, { fs: 13, align: "left" });
  group("pipe1", () => { mini([{ t: "Frame\ndiff +\nnorm", w: 0.82, fs: 11.5 }, { t: "3D-CNN\ntrunk", w: 0.9, fs: 11.5 }, { t: "FSAM\nrank 1", w: 0.8, fs: 11.5 }, { t: "Head", w: 0.7, fs: 11.5 }], colB + 0.14, 1.22, 0.92, 0.12); txt("Temporal receptive field: 25 frames\nAttention: a residual side path\nNMF bases get no gradient", colB + 0.05, 2.36, wB - 0.1, 0.96, { fs: 12.5, color: K.mute }); });
  frame(colC, 0.6, wC, 2.95, { dash: true, lw: 0.75, name: "C1" }); txt("**C.** Output spectrum", colC + 0.07, 0.65, wC - 0.1, 0.28, { fs: 13, align: "left" });
  const specLab = (x, w, y) => { const X = (v) => x + w * (v - 30) / 170; txt("HR", X(D.hr) - 0.2, y, 0.4, 0.2, { fs: 11, color: K.pulse, b: true }); txt("flicker", X(D.nuis) + 0.03, y, 0.62, 0.2, { fs: 11, color: K.noise, b: true, align: "left" }); };
  group("spec1", () => { spectrumPlot(colC + 0.13, 1.14, wC - 0.26, 1.48, D.f, [{ y: D.sp, color: K.line, lw: 1.75 }], { marks: [{ v: D.hr, color: K.pulse }, { v: D.nuis, color: K.noise }], xlabel: "bpm" }); specLab(colC + 0.13, wC - 0.26, 0.94); });
  txt("peak **" + pB.toFixed(0) + "** bpm:\nfollows the flicker", colC + 0.03, 3.0, wC - 0.06, 0.48, { fs: 12, name: "peak1" });
  line(colC + wC, 2.05, colD, 2.05);
  group("out1", () => { wave(D.pub, colD + 0.03, 1.5, 1.74, 0.72, { color: K.line, n: 100, name: "pulse_public" }); txt("Estimated pulse", colD, 1.14, 1.8, 0.26, { fs: 12.5, b: true }); });
  txt(`Zero-shot MAE\nVIPL-HR **${pubL.vipl.toFixed(2)}** bpm\nRTNI flicker **${AX.RTNI.base.toFixed(2)}** bpm`, colD - 0.03, 2.4, 1.84, 0.92, { fs: 12, name: "mae1" });
  // 아랫줄
  frame(2.42, 3.82, 10.78, 3.55, { dash: true, lw: 1.25, line: K.ours, name: "outer2" });
  txt("**2.**  SpecPhys: same trunk, same 52,169 parameters", 2.55, 3.88, 10.4, 0.34, { fs: 15, align: "left", color: K.ours });
  frame(colA, 4.28, wA, 2.95, { dash: true, lw: 0.75, name: "A2" }); txt("**A.** Training data", colA + 0.07, 4.33, wA - 0.1, 0.28, { fs: 13, align: "left" });
  group("pool2", () => { [["real clips", 237, K.gray2], ["real + noise", 29, K.noiseF], ["PPG-injected", 80, K.pulseF], ["lossy copies", 20, K.gray3]].forEach((r, i) => { const y = 4.66 + i * 0.44; box(`**${r[1]}**`, colA + 0.08, y, 0.5, 0.36, { fill: r[2], line: K.edge, fs: 12, margin: 0 }); txt(r[0], colA + 0.64, y, wA - 0.68, 0.36, { fs: 12, align: "left" }); }); txt("48 synthetic clips\nrendered twice:\nexact noise trace", colA + 0.04, 6.44, wA - 0.08, 0.72, { fs: 12, color: K.pulse }); });
  frame(colB, 4.28, wB, 2.95, { dash: true, lw: 0.75, name: "B2" }); txt("**B.** Model and training", colB + 0.07, 4.33, wB - 0.1, 0.28, { fs: 13, align: "left" });
  group("pipe2", () => { mini([{ t: "Skin\nweight", w: 0.56, fill: K.oursF, line: K.ours, fs: 10.5, color: K.ours }, { t: "Diff +\nnorm", w: 0.54, fs: 10.5 }, { t: "3D-CNN\ntrunk", w: 0.66, fs: 10.5 }, { t: "Noise-\nsubspace\nprojection", w: 0.84, fill: K.ours, line: K.ours, color: K.white, fs: 10.5, b: true }, { t: "Two-slot\nFSAM", w: 0.7, fill: K.ours, line: K.ours, color: K.white, fs: 10.5, b: true }, { t: "Head", w: 0.44, fs: 10.5 }], colB + 0.07, 4.72, 0.92, 0.056); box("**Spectrum-guided losses**\nheart-rate band · halving · doubling\nnoise-reference exclusion", colB + 0.14, 5.92, wB - 0.28, 1.16, { fill: K.oursT, line: K.ours, fs: 12.5 }); line(colB + wB / 2, 5.92, colB + wB / 2, 5.66, { color: K.ours, lw: 1.25, name: "loss_to_model" }); });
  frame(colC, 4.28, wC, 2.95, { dash: true, lw: 0.75, name: "C2" }); txt("**C.** Output spectrum", colC + 0.07, 4.33, wC - 0.1, 0.28, { fs: 13, align: "left" });
  group("spec2", () => { spectrumPlot(colC + 0.13, 4.82, wC - 0.26, 1.48, D.f, [{ y: D.so, color: K.ours, lw: 1.75 }], { marks: [{ v: D.hr, color: K.pulse }, { v: D.nuis, color: K.noise }], xlabel: "bpm" }); specLab(colC + 0.13, wC - 0.26, 4.62); });
  txt("peak **" + oB.toFixed(0) + "** bpm:\nfollows the pulse", colC + 0.03, 6.68, wC - 0.06, 0.48, { fs: 12, name: "peak2" });
  line(colC + wC, 5.73, colD, 5.73);
  group("out2", () => { wave(D.out, colD + 0.03, 5.18, 1.74, 0.72, { color: K.pulse, n: 100, name: "pulse_specphys" }); txt("Estimated pulse", colD, 4.82, 1.8, 0.26, { fs: 12.5, b: true }); });
  txt(`Zero-shot MAE\nVIPL-HR **${finL.vipl.toFixed(2)}** bpm\nRTNI flicker **${AX.RTNI.cell.toFixed(2)}** bpm`, colD - 0.03, 6.08, 1.84, 0.92, { fs: 12, color: K.ours, name: "mae2" });
}

// ======================= 그림 2: 텐서 차원이 있는 전체 구조 =======================
function figNetwork() {
  const fb = peak(D.f, D.feat_before), fa = peak(D.f, D.feat_after);
  begin("Figure 2: SpecPhys network", `Figure 2. (A) SpecPhys network with tensor sizes T×H×W×C and, below each layer, kernel size; stride. The sizes are read from a forward pass of the final model. Blue blocks are what SpecPhys adds to the public FactorizePhys model; trunk and head are the same and the parameter count is unchanged (52,169). The original FactorizePhys attention is rank 1; FSAM here has two temporal bases. (B) Noise references computed from the input of the same chunk, with a real example: a generated-face clip that is not in the training pool, heart rate ${D.hr} bpm, illumination flicker ${D.nuis} bpm, seed 1600. The side-band colour traces carry the flicker, and the projection moves the peak of the mean temporal-feature spectrum from ${fb.toFixed(0)} to ${fa.toFixed(0)} bpm. One clip, shown for illustration.`);
  const base = 3.08, dx = 0.86, dy = 0.78; const hOf = (s) => 0.175 * Math.sqrt(s), wOf = (s) => 0.60 * hOf(s); const yc = 2.85;
  txt("**A**", 0.12, 0.06, 0.32, 0.32, { fs: 18, name: "panelA" });
  group("legend", () => { frame(9.5, 0.16, 0.22, 0.16, { fill: K.ours, line: K.ours, name: "key" }); txt("added by SpecPhys", 9.78, 0.12, 1.5, 0.24, { fs: 11, align: "left" }); frame(11.3, 0.16, 0.22, 0.16, { fill: K.front, line: K.edge, name: "key" }); txt("as in FactorizePhys", 11.58, 0.12, 1.66, 0.24, { fs: 11, align: "left" }); });
  group("input", () => { txt("161×72×72×3", 0.12, 0.98, 1.2, 0.22, { fs: 10.5 }); stack(["face_0.png", "face_1.png", "face_2.png"], 0.2, 1.62, 0.92, 0.09); txt("**Raw frames**", 0.08, 2.6, 1.28, 0.22, { fs: 12 }); txt("generated face", 0.08, 2.82, 1.28, 0.2, { fs: 10, color: K.mute }); });
  line(1.34, 2.14, 1.48, 2.14);
  group("skinw", () => { img("p_skin.png", 1.62, 0.98, 0.52, 0.52, { bc: K.ours, blw: 1, alt: "Skin probability from the face parser" }); txt("$p$_{skin}", 1.52, 0.72, 0.72, 0.24, { fs: 12 }); box("Skin\nweighting", 1.48, 1.78, 0.8, 0.72, { fill: K.oursF, line: K.ours, fs: 10.5, b: true, margin: 0, color: K.ours }); line(1.88, 1.5, 1.88, 1.78, { color: K.ours, lw: 1 }); txt("× (0.2 + 0.8 $p$)", 1.28, 2.54, 1.2, 0.24, { fs: 10.5, color: K.ours }); });
  line(2.28, 2.14, 2.4, 2.14);
  const L = [["d", 72, 3, 2.4, 0.52, 0.58], ["8", 72, 8, 2.92, null, 0.36], ["35", 35, 12, 3.81, null, 1.0], ["33", 33, 12, 4.43, null, 0.78], ["31", 31, 12, 5.03, null, 1.02], ["15", 15, 16, 5.61, null, 1.4], ["13", 13, 16, 6.11, null, 1.18], ["11", 11, 16, 6.61, null, 1.46], ["9", 9, 16, 7.11, null, 1.24], ["7", 7, 16, 7.61, null, 1.54]];
  const stride = { "8": "1×1×1", "35": "1×2×2", "33": "1×1×1", "31": "1×1×1", "15": "1×2×2", "13": "1×1×1", "11": "1×1×1", "9": "1×1×1", "7": "1×1×1" };
  L.forEach((b) => group(`block_${b[0]}`, () => { const h = hOf(b[1]), w = b[4] || wOf(b[1]); const y = base - h; if (b[0] === "d") cuboid(b[3], y, w, h, dx, dy, { pal: { front: K.white, top: K.white, side: K.white, edge: K.gray }, grid: [12, 1, 16], gridColor: K.gray }); else cuboid(b[3], y, w, h, dx, dy);
    const lab = b[0] === "7" ? "$F$: 160×7×7×16" : `160×${b[1]}×${b[1]}×${b[2]}`; txt(lab, b[3] + w / 2 + dx * 0.95 - 0.55, b[5], 1.1, 0.2, { fs: 10, name: "dims" }); if (b[0] === "d") txt("Frame diff\nInstance norm", 1.72, base + 0.08, 1.2, 0.36, { fs: 10, color: K.mute, align: "right" }); else txt(`3×3×3;\n${stride[b[0]]}`, b[3] + w / 2 - 0.27, base + 0.08, 0.54, 0.36, { fs: 10, color: K.mute, name: "kernel" }); }));
  line(7.89, yc, 8.06, yc, { lw: 1.5 });
  group("projection", () => box("Noise-subspace\nprojection\n160×7×7×16", 8.06, yc - 0.42, 1.16, 0.84, { fill: K.ours, line: K.ours, color: K.white, fs: 10.5, b: true, margin: 0 }));
  line(9.22, yc, 9.38, yc, { lw: 1.5 });
  group("fsam", () => box("Two-slot\nFSAM\n+ residual", 9.38, yc - 0.42, 0.9, 0.84, { fill: K.ours, line: K.ours, color: K.white, fs: 10.5, b: true, margin: 0 }));
  line(10.28, yc, 10.44, yc, { lw: 1.5 });
  [["5", 5, 12, 10.44, 1.74], ["3", 3, 8, 10.94, 1.52], ["1", 1, 1, 11.44, 1.9]].forEach((b) => group(`head_${b[0]}`, () => { const h = hOf(b[1]), w = wOf(b[1]); const y = base - h; cuboid(b[3], y, w, h, dx, dy); txt(`160×${b[1]}×${b[1]}×${b[2]}`, b[3] + w / 2 + dx * 0.95 - 0.55, b[4], 1.1, 0.2, { fs: 10, name: "dims" }); txt("3×3×3;\n1×1×1", b[3] + w / 2 - 0.27, base + 0.08, 0.54, 0.36, { fs: 10, color: K.mute, name: "kernel" }); }));
  line(11.56, 3.0, 12.08, 3.0, { lw: 1.25 });
  group("output", () => { wave(D.out, 12.1, 2.62, 1.12, 0.5, { n: 70, name: "pulse" }); txt("**Estimated\nBVP signal**", 12.06, 3.16, 1.2, 0.42, { fs: 11 }); });
  seg(0.15, 3.68, 13.18, 3.68, { color: K.gray, lw: 2.25, name: "sep" });
  txt("**B**", 0.12, 3.74, 0.32, 0.32, { fs: 18, name: "panelB" });
  txt("**Noise references of one chunk**, computed from the frames the model receives", 0.5, 3.76, 8.4, 0.3, { fs: 13, align: "left" });
  const by = 4.3;
  group("bands", () => { const s = 1.42, x = 0.32, y = by + 0.3; img("skin_weighted.png", x, y, s, s, { alt: "Skin-weighted generated face frame with side bands highlighted" }); frame(x, y, s / 9, s, { fill: K.noise, alpha: 45, line: K.noise, lw: 1, name: "bandL" }); frame(x + s - s / 9, y, s / 9, s, { fill: K.noise, alpha: 45, line: K.noise, lw: 1, name: "bandR" }); txt("Side bands: outer 1/9\n0.4–2.3 % skin", 0.12, y + s + 0.06, 1.82, 0.42, { fs: 11, color: K.noise }); });
  line(1.78, by + 0.72, 2.1, by + 0.72, { color: K.noise }); line(1.78, by + 1.6, 2.1, by + 1.6);
  group("traces", () => {
    txt("Side-band colour, difference-normalized", 2.14, by - 0.06, 2.72, 0.24, { fs: 11, align: "left" });
    [["R", D.refs.R, "C0392B"], ["G", D.refs.G, "2E8B57"], ["B", D.refs.B, "2F6DB5"]].forEach((c, i) => { txt(c[0], 2.12, by + 0.2 + i * 0.34, 0.2, 0.3, { fs: 11, color: c[2], b: true }); wave(c[1], 2.34, by + 0.22 + i * 0.34, 2.4, 0.28, { color: c[2], lw: 1, name: `ref_${c[0]}` }); });
    txt("Lucas–Kanade global motion", 2.14, by + 1.26, 2.72, 0.24, { fs: 11, align: "left" });
    [["$v_{x}$", D.refs.vx], ["$v_{y}$", D.refs.vy]].forEach((c, i) => { txt(c[0], 2.08, by + 1.5 + i * 0.36, 0.26, 0.3, { fs: 11 }); wave(c[1], 2.34, by + 1.52 + i * 0.36, 2.4, 0.28, { color: K.mute, lw: 1, name: "ref_motion" }); });
  });
  line(4.78, by + 1.12, 4.92, by + 1.12);
  group("gate", () => box("Standardize\n× amplitude gate\n$a\\,/\\,(a+a_{0})$".replace(/\\,/g, " "), 4.92, by + 0.62, 1.2, 1.0, { fill: K.white, fs: 10.5, margin: 1 }));
  line(6.12, by + 1.12, 6.26, by + 1.12);
  group("Rmat", () => { img("R_matrix.png", 6.26, by + 0.62, 1.14, 1.0, { bc: K.line, blw: 0.75, alt: "Reference matrix R: five references and their Hilbert transforms over 160 frames" }); txt("$R$: 5 references + Hilbert\n10×160", 5.92, by + 1.66, 1.82, 0.42, { fs: 11 }); });
  line(7.4, by + 1.12, 7.54, by + 1.12);
  group("ridge", () => box("Per temporal feature $x$ of $F$:\n$x′ = x − x̃R^{T}(RR^{T} + λTI)^{−1}R$\n$λ$ = 0.01", 7.54, by + 0.62, 2.44, 1.0, { fill: K.oursF, line: K.ours, fs: 10.5, margin: 2 }));
  line(8.64, by + 0.62, 8.64, yc + 0.42, { color: K.ours, lw: 1.5, name: "toprojection" });
  group("effect", () => {
    txt("Mean temporal-feature spectrum", 10.14, by - 0.08, 3.06, 0.24, { fs: 11.5, b: true });
    spectrumPlot(10.24, by + 0.36, 2.86, 1.54, D.f, [{ y: D.feat_before, color: K.arrow, lw: 1.5 }, { y: D.feat_after, color: K.ours, lw: 1.75 }], { marks: [{ v: D.hr, color: K.pulse }, { v: D.nuis, color: K.noise }], xlabel: "bpm" });
    const X = (v) => 10.24 + 2.86 * (v - 30) / 170; txt("HR", X(D.hr) - 0.2, by + 0.16, 0.4, 0.2, { fs: 11, color: K.pulse, b: true }); txt("flicker", X(D.nuis) - 0.1, by + 0.16, 0.6, 0.2, { fs: 11, color: K.noise, b: true, align: "left" });
    seg(11.68, by + 0.56, 11.9, by + 0.56, { color: K.arrow, lw: 1.5, name: "legend" }); txt(`before, peak ${fb.toFixed(0)}`, 11.94, by + 0.46, 1.26, 0.2, { fs: 10, align: "left" }); seg(11.68, by + 0.8, 11.9, by + 0.8, { color: K.ours, lw: 1.75, name: "legend" }); txt(`after, peak ${fa.toFixed(0)}`, 11.94, by + 0.7, 1.26, 0.2, { fs: 10, align: "left", color: K.ours });
  });
  txt(`Example: generated face, not in the training pool, heart rate ${D.hr} bpm, flicker ${D.nuis} bpm. One 160-frame chunk.`, 0.3, 7.02, 12.8, 0.28, { fs: 11, color: K.mute, align: "left", i: true, name: "note" });
}

// ======================= 그림 3: 잡음 부분공간 투영 + 두 슬롯 분해 모듈 =======================
function figModule() {
  const pp = D.peaks || {};
  begin("Figure 3: Projection and two-slot attention", `Figure 3. Noise-subspace projection and two-slot FSAM. Top left: the projection removes from each temporal feature of the voxel embedding what can be regressed on the clip's own noise references. Heat maps are real: the 120 strongest of 784 features over 160 frames, in units of the standard deviation before projection. Top right: the embedding is factorized with a pulse basis and a noise basis, and only the pulse basis forms the attention. Traces are the real bases after the updates; on this clip they peak at ${pp.b_p.toFixed(0)} and ${pp.b_n.toFixed(0)} bpm. Bottom: the module in the network. Example clip: generated face, not in the training pool, heart rate ${D.hr} bpm, flicker ${D.nuis} bpm, seed 1600. One clip, shown for illustration. Over the evaluation sets the gain comes from fewer large mistakes, and lock-in does not fall; see the paper.`);
  // ---- 위 왼쪽: 투영
  frame(0.2, 0.14, 6.0, 3.56, { fill: K.oursT, noline: true, name: "panel_i" });
  txt("**Noise-subspace projection** (no parameters)", 0.3, 0.18, 5.8, 0.3, { fs: 14, align: "left", color: K.ours });
  group("F", () => { vox(0.32, 1.16, { front: "F4F6F9", top: "E6EAF0", side: "D2D8E2", edge: "8C94A3" }, { labels: ["$C_{1}$", "$C_{2}$", "$C_{16}$"] }); txt("$T$", 0.46, 0.8, 0.24, 0.22, { fs: 12 }); txt("$F$: 160×7×7×16", 0.24, 3.1, 1.84, 0.24, { fs: 12 }); txt("Voxel embeddings", 0.24, 3.34, 1.84, 0.24, { fs: 12, b: true }); });
  blockArrow(1.98, 1.42, 0.3, 0.3);
  group("heat", () => {
    img("X_before.png", 2.34, 1.0, 1.04, 1.14, { bc: K.line, alt: "Temporal features before projection" }); txt("$X$", 2.34, 0.7, 1.04, 0.28, { fs: 14 }); txt("features × time", 2.24, 2.16, 1.24, 0.2, { fs: 10.5, color: K.mute });
    txt("−", 3.4, 1.38, 0.26, 0.36, { fs: 20, b: true });
    img("X_removed.png", 3.68, 1.0, 1.04, 1.14, { bc: K.noise, blw: 1.25, alt: "Part of the features explained by the noise references" }); txt("fit on $R$", 3.68, 0.7, 1.04, 0.28, { fs: 13, color: K.noise, b: true });
    txt("=", 4.74, 1.38, 0.26, 0.36, { fs: 20, b: true });
    img("X_after.png", 5.02, 1.0, 1.04, 1.14, { bc: K.ours, blw: 1.25, alt: "Temporal features after projection" }); txt("$X′$", 5.02, 0.7, 1.04, 0.28, { fs: 14, color: K.ours });
  });
  group("Rin", () => { img("R_matrix.png", 2.42, 2.5, 0.88, 0.62, { bc: K.line, alt: "Reference matrix R" }); txt("$R$: side-band colour,\nmotion, + Hilbert", 2.12, 3.14, 1.48, 0.42, { fs: 10.5, color: K.mute }); poly([[3.3, 2.81], [4.2, 2.81], [4.2, 2.18]], { head: "triangle", color: K.noise, lw: 1.5, name: "ridgearrow" }); txt("$x̃R^{T}(RR^{T}+λTI)^{−1}R$", 3.62, 2.86, 2.5, 0.3, { fs: 12 }); txt("ridge fit, $λ$ = 0.01", 3.62, 3.16, 2.5, 0.24, { fs: 11, color: K.mute }); });
  // ---- 위 오른쪽: 두 슬롯 NMF
  frame(6.3, 0.14, 6.88, 3.56, { fill: K.warm, noline: true, name: "panel_ii" });
  txt("**Two-slot factorization**: attention from the pulse basis only", 6.4, 0.18, 6.7, 0.3, { fs: 14, align: "left" });
  group("M", () => { const x = 6.74, y = 1.0, w = 0.98, h = 1.44; frame(x, y, w, h, { fill: K.white, line: K.mute, name: "M" }); const g = []; for (let i = 1; i < 26; i++) g.push([[x + w * i / 26, y], [x + w * i / 26, y + h]]); poly(g, { color: K.gray, lw: 0.4, name: "stripes" }); line(x, y - 0.08, x + w, y - 0.08, { lw: 1 }); txt("$N$ = 8·7·7", x - 0.1, y - 0.33, w + 0.2, 0.22, { fs: 11 }); line(x - 0.08, y, x - 0.08, y + h, { lw: 1 }); txt("$T$ = 160", x - 0.36, y + 0.3, 0.24, 0.84, { fs: 11, rot: 270 }); txt("$M$ ≥ 0, rows index time", x - 0.42, y + h + 0.06, w + 0.84, 0.24, { fs: 11 }); });
  blockArrow(7.76, 1.58, 0.24, 0.3);
  group("nmf", () => box("Non-negative matrix\nfactorization, rank 2", 8.04, 0.86, 0.6, 1.74, { rot: 270, fill: K.white, line: K.ours, lw: 1.25, fs: 11, b: true, r: 0.08, margin: 0 }));
  blockArrow(8.68, 1.58, 0.24, 0.3);
  group("bases", () => {
    txt("Pulse basis $s_{p}$", 8.96, 0.54, 1.76, 0.24, { fs: 12, color: K.pulse, b: true }); wave(D.b_p, 8.98, 0.8, 1.72, 0.48, { color: K.pulse, lw: 1.25, name: "pulse_basis" }); txt(`starts at 1/√$T$\nhere peak ${pp.b_p.toFixed(0)} bpm`, 8.8, 1.29, 2.1, 0.38, { fs: 10.5, color: K.mute });
    txt("Noise basis $s_{n}$", 8.96, 1.68, 1.76, 0.24, { fs: 12, color: K.noise, b: true }); wave(D.b_n, 8.98, 1.94, 1.72, 0.48, { color: K.noise, lw: 1.25, name: "noise_basis" }); txt(`starts at the intensity trace\nhere peak ${pp.b_n.toFixed(0)} bpm`, 8.8, 2.43, 2.1, 0.38, { fs: 10.5, color: K.mute });
    txt("4 multiplicative updates (no gradient)\n+ 1 differentiable coefficient update", 6.5, 3.0, 4.3, 0.46, { fs: 11, color: K.mute });
  });
  txt("pulse\nonly", 10.72, 1.02, 0.44, 0.4, { fs: 10.5, color: K.pulse, b: true, name: "pulseonly" }); blockArrow(10.82, 1.46, 0.26, 0.3, { color: K.pulse });
  group("A", () => { vox(11.12, 1.06, { front: "FDF3F1", top: "F8E6E3", side: "F0D3CE", edge: "D39A92" }, { labels: ["$C_{1}$", "$C_{2}$", "$C_{16}$"] }); txt("**Attention**", 10.96, 2.9, 2.2, 0.24, { fs: 12 }); txt("$A$ = conv($s_{p}c_{p}^{T}$), 160×7×7×16", 10.86, 3.14, 2.3, 0.42, { fs: 10.5 }); });
  // ---- 아래: 파이프라인
  const yc = 5.42;
  frame(2.86, 4.18, 8.02, 3.16, { fill: K.oursT, noline: true, name: "module" });
  txt("**Noise-subspace projection and two-slot FSAM**", 2.92, 6.98, 7.9, 0.3, { fs: 14, color: K.ours });
  brace(3.02, 4.34, 3.82, 0.3, { tip: 3.68, color: K.ours }); brace(5.0, 8.86, 3.82, 0.3, { tip: 7.6, color: K.mute });
  group("in", () => { stack(["face_0.png", "face_1.png", "face_2.png"], 0.2, yc - 0.24, 0.84, 0.09); txt("**Input video\nframes**", 0.1, yc + 0.66, 1.24, 0.46, { fs: 12 }); txt("generated face", 0.1, yc + 1.1, 1.24, 0.2, { fs: 10, color: K.mute }); });
  line(1.25, yc + 0.1, 1.42, yc + 0.1);
  group("fe", () => { [[1.42, 0.78], [1.8, 0.6], [2.14, 0.44], [2.42, 0.32]].forEach((q) => cuboid(q[0], yc + 0.2 - q[1] / 2, 0.22, q[1], 0.2, 0.2)); txt("**Feature\nextractor**", 1.5, yc + 0.66, 1.2, 0.46, { fs: 12 }); });
  line(2.84, yc, 3.04, yc, { lw: 1.5 }); txt("$F$", 2.8, yc - 0.3, 0.24, 0.24, { fs: 13, name: "F" });
  group("proj", () => box("Noise-subspace\nprojection", 3.04, yc - 0.45, 1.28, 0.9, { fill: K.ours, line: K.ours, color: K.white, fs: 11.5, b: true, margin: 1 }));
  line(4.32, yc, 4.52, yc, { lw: 1.5 }); txt("$F′$", 4.3, yc - 0.3, 0.26, 0.24, { fs: 13, color: K.ours, name: "Fp" });
  ramp(4.52, yc - 0.17, 0.34); line(4.86, yc, 5.0, yc);
  box("Embeddings\nto $T$×$N$\nmatrix", 5.0, yc - 0.55, 1.16, 1.1, { fill: K.white, line: K.ours, lw: 1.25, fs: 12, margin: 1 }); line(6.16, yc, 6.3, yc);
  box("**NMF**, rank 2\npulse basis +\nnoise basis", 6.3, yc - 0.55, 1.2, 1.1, { fill: K.white, line: K.ours, lw: 1.25, fs: 12, margin: 1 }); line(7.5, yc, 7.64, yc);
  box("Pulse basis\nonly, back to\nembeddings", 7.64, yc - 0.55, 1.2, 1.1, { fill: K.white, line: K.ours, lw: 1.25, fs: 12, margin: 1 }); line(8.84, yc, 8.98, yc);
  ramp(8.98, yc - 0.17, 0.34);
  poly([[9.15, yc + 0.17], [9.15, 6.05]], { head: "triangle", name: "attdown" }); op("×", 9.15, 6.2, 0.3);
  poly([[4.42, yc], [4.42, 6.2], [9.0, 6.2]], { head: "triangle", name: "fprime_to_mul" });
  poly([[9.3, 6.2], [9.56, 6.2], [9.56, yc], [9.72, yc]], { head: "triangle", name: "mul_to_norm" });
  box("Instance\nNorm", 9.72, yc - 0.32, 0.86, 0.64, { fill: K.white, line: K.ours, lw: 1.25, fs: 12, margin: 1 });
  poly([[10.58, yc], [10.78, yc], [10.78, 5.73]], { head: "triangle", name: "norm_to_add" }); op("+", 10.78, 5.88, 0.3);
  poly([[4.42, 6.2], [4.42, 6.66], [10.78, 6.66], [10.78, 6.03]], { head: "triangle", name: "residual" }); txt("Residual connection", 6.2, 6.68, 2.6, 0.26, { fs: 12 });
  line(10.93, 5.88, 11.1, 5.88, { lw: 1.5 });
  group("head", () => { [[11.1, 0.6], [11.5, 0.42]].forEach((q) => cuboid(q[0], 5.88 - q[1] / 2 + 0.1, 0.22, q[1], 0.2, 0.2)); txt("**Network\nhead**", 11.0, 6.42, 1.0, 0.46, { fs: 12 }); });
  line(11.94, 5.88, 12.1, 5.88, { lw: 1.25 });
  group("out", () => { wave(D.out, 12.12, 5.56, 1.08, 0.6, { n: 80, name: "pulse" }); txt("**Estimated\nBVP signal**", 12.04, 6.26, 1.2, 0.46, { fs: 12 }); });
  poly([[0.7, yc + 1.14], [0.7, 6.72], [2.62, 6.72], [2.62, 6.3], [3.68, 6.3], [3.68, yc + 0.45]], { head: "triangle", color: K.noise, dash: true, lw: 1.5, name: "refs_path" }); txt("noise references $R$", 0.8, 6.76, 2.0, 0.24, { fs: 11.5, color: K.noise, b: true, align: "left" });
  poly([[0.62, yc - 0.34], [0.62, 4.34], [6.9, 4.34], [6.9, yc - 0.55]], { head: "triangle", color: K.noise, dash: true, lw: 1.5, name: "trace_path" }); txt("the clip's intensity trace starts the noise basis", 3.0, 4.38, 3.88, 0.24, { fs: 11.5, color: K.noise, align: "left" });
}

// ======================= 그림 4: PPG 주입 합성 클립과 스펙트럼 유도 학습 =======================
function figTrain() {
  const b = D.bpm, HR = D.hr;
  begin("Figure 4: PPG-injected data and spectrum-guided training", `Figure 4. Top: a PPG-injected synthetic clip. A generated identity is animated, a waveform from the real PPG bank is injected per skin pixel along the blood-volume-pulse direction, and every nuisance is generated, so the clip can be rendered twice with the same nuisance draw to give an exact nuisance trace. The change image is the injected change at the pulse peak, amplified 80 times (alpha 0.01). Bottom: training on the 366-clip hybrid pool with the spectrum-guided terms; weights in brackets. The spectrum is the real SpecPhys output of one chunk of the example clip (heart rate ${HR} bpm, flicker ${D.nuis} bpm, seed 1600) with the heart-rate band, the halving and doubling bands and the noise-reference mask M(f). The clip is not in the training pool, so no loss was computed on it; it only shows where each term acts.`);
  frame(0.12, 0.12, 13.08, 7.26, { dash: true, lw: 1.25, name: "outer" });
  txt("**PPG-injected synthetic clip**", 0.26, 0.17, 8, 0.32, { fs: 15, align: "left" });
  frame(0.26, 0.52, 9.68, 2.92, { lw: 1.5, name: "row1" });
  group("id", () => { img("anim_0.png", 0.42, 0.98, 1.08, 1.08, { bc: K.line, alt: "Generated identity" }); txt("Generated\nidentity", 0.32, 2.12, 1.28, 0.46, { fs: 12 }); });
  line(1.52, 1.52, 1.72, 1.52);
  group("anim", () => { stack(["anim_2.png", "anim_1.png", "anim_0.png"], 1.74, 1.14, 0.92, 0.1, { bc: K.line }); txt("Animated with\nreal driving\ntrajectories", 1.62, 2.12, 1.4, 0.68, { fs: 12 }); });
  line(2.9, 1.52, 3.1, 1.52);
  group("inject", () => { box("**Per-pixel PPG injection**\nalong $P_{bv}$ = [0.33, 0.77, 0.53]", 3.1, 0.68, 2.36, 0.68, { fill: K.pulseF, line: K.pulse, fs: 11.5, margin: 1 }); wave(D.ppg_wave, 3.26, 1.48, 2.04, 0.5, { n: 120, name: "ppg" }); txt("real PPG bank, 131 segments", 3.1, 2.02, 2.36, 0.24, { fs: 11, color: K.pulse }); txt("amplitude < 1 %, lossless", 3.1, 2.26, 2.36, 0.24, { fs: 11, color: K.mute }); });
  line(5.46, 1.52, 5.66, 1.52);
  group("nuis", () => box("**Nuisance generator**\ndrift, flicker, rigid motion,\nsensor noise", 5.66, 0.92, 2.0, 1.2, { fill: K.noiseF, line: K.noise, fs: 11.5, margin: 2 }));
  line(7.66, 1.52, 7.84, 1.52);
  group("twice", () => {
    box("Pulse on", 7.84, 0.72, 1.0, 0.42, { fill: K.pulseF, line: K.pulse, fs: 12, margin: 0 }); box("Pulse off", 7.84, 1.88, 1.0, 0.42, { fill: K.noiseF, line: K.noise, fs: 12, margin: 0 }); txt("same\nnuisance\ndraw", 7.84, 1.16, 1.0, 0.7, { fs: 11, color: K.mute });
    line(8.84, 0.93, 9.0, 0.93); txt("clip +\nPPG", 9.0, 0.72, 0.86, 0.42, { fs: 11.5, b: true });
    line(8.84, 2.09, 9.0, 2.09, { color: K.noise }); txt("$n(t)$", 9.0, 1.92, 0.86, 0.3, { fs: 13, color: K.noise, b: true }); wave(D.n_trace, 7.9, 2.42, 1.92, 0.42, { color: K.noise, lw: 1.25, name: "n_trace" }); txt("exact nuisance trace, 48 clips", 7.66, 2.92, 2.26, 0.24, { fs: 11, color: K.noise });
  });
  txt("Stored losslessly as FFV1. The dual-render trace is a nuisance label that real video cannot give.", 0.42, 3.1, 7.1, 0.28, { fs: 11.5, color: K.mute, align: "left", name: "note1" });
  frame(10.06, 0.52, 3.02, 2.92, { fill: K.pulseF, line: K.pulse, lw: 1, name: "side" });
  txt("**Injected change**", 10.12, 0.56, 2.9, 0.28, { fs: 13 });
  group("diff", () => { img("inject_on.png", 10.2, 0.92, 1.32, 1.32, { bc: K.line, alt: "Frame with injected pulse" }); img("inject_diff.png", 11.62, 0.92, 1.32, 1.32, { bc: K.line, alt: "Injected change amplified 80 times" }); txt("frame", 10.2, 2.26, 1.32, 0.22, { fs: 11 }); txt("change × 80", 11.62, 2.26, 1.32, 0.22, { fs: 11 }); });
  txt("$I_{c}$ ← $I_{c}$ (1 + $α$ $w(t)$ $P_{bv,c}$ $S(x,y)$)", 10.12, 2.54, 2.9, 0.34, { fs: 12 }); txt("$S$: skin mask, $α$ = 0.01", 10.12, 2.92, 2.9, 0.3, { fs: 11, color: K.mute });
  // ---- 아래: 학습
  txt("**Spectrum-guided training on the hybrid pool**", 0.26, 3.5, 8, 0.32, { fs: 15, align: "left" });
  frame(0.26, 3.85, 12.82, 3.42, { lw: 1.5, name: "row2" });
  group("pool", () => { txt("**366-clip pool**", 0.36, 3.92, 2.1, 0.26, { fs: 12.5 }); [["UBFC-rPPG, PURE, MCD-rPPG, MMPD 1–6", 237, K.gray2], ["real + synthetic noise", 29, K.noiseF], ["PPG-injected synthetic", 80, K.pulseF], ["lossy-codec copies", 20, K.gray3]].forEach((r, i) => { const y = 4.24 + i * 0.56; box(`**${r[1]}**`, 0.38, y, 0.5, 0.46, { fill: r[2], line: K.edge, fs: 12, margin: 0 }); txt(r[0], 0.94, y, 1.56, 0.46, { fs: 10.5, align: "left" }); }); txt("+ double-speed replay,\n$p$ = 0.25", 0.34, 6.48, 2.2, 0.46, { fs: 11, color: K.mute }); });
  line(2.52, 5.3, 2.72, 5.3, { lw: 1.5 });
  group("model", () => { txt("**SpecPhys**", 2.74, 3.92, 2.8, 0.26, { fs: 12.5, color: K.ours }); box("Skin weighting", 2.74, 4.22, 2.8, 0.34, { fill: K.oursF, line: K.ours, fs: 11, margin: 0, color: K.ours }); line(3.3, 4.56, 3.3, 4.72, { lw: 1 }); [[2.9, 0.62], [3.34, 0.48], [3.74, 0.36]].forEach((q) => cuboid(q[0], 5.4 - q[1], 0.26, q[1], 0.22, 0.22)); txt("3D-CNN\ntrunk", 4.36, 4.82, 1.1, 0.46, { fs: 11, color: K.mute, align: "left" }); line(3.3, 5.4, 3.3, 5.54, { lw: 1 }); box("Noise-subspace projection", 2.74, 5.54, 2.8, 0.34, { fill: K.ours, line: K.ours, fs: 11, b: true, color: K.white, margin: 0 }); line(3.3, 5.88, 3.3, 6.02, { lw: 1 }); box("Two-slot FSAM", 2.74, 6.02, 1.5, 0.34, { fill: K.ours, line: K.ours, fs: 11, b: true, color: K.white, margin: 0 }); line(4.24, 6.19, 4.38, 6.19, { lw: 1 }); box("Head", 4.38, 6.02, 1.16, 0.34, { fill: K.front, line: K.edge, fs: 11, margin: 0 }); txt("52,169 parameters", 2.74, 6.48, 2.8, 0.26, { fs: 11, color: K.mute }); });
  line(5.54, 5.3, 5.74, 5.3, { lw: 1.5 });
  group("yhat", () => { txt("$ŷ(t)$", 5.74, 4.56, 1.14, 0.3, { fs: 14 }); wave(D.out, 5.8, 4.98, 1.02, 0.56, { n: 80, name: "yhat" }); txt("+ slot traces\n$s_{p}$, $s_{n}$", 5.74, 5.68, 1.14, 0.46, { fs: 11, color: K.mute }); });
  line(6.88, 5.3, 7.06, 5.3, { lw: 1.5 });
  group("spectrum", () => {
    txt("Where each term acts on $P̂(f)$", 6.98, 3.92, 2.68, 0.26, { fs: 12.5, b: true });
    const w5 = 5; const px = 7.14, pw = 2.36, py = 4.46, ph = 1.72; spectrumPlot(px, py, pw, ph, b, [{ y: D.mask, color: K.noise, fill: K.noiseF, closed: true, noline: true }, { y: D.p_out.map((v) => v / Math.max(...D.p_out)), color: K.ours, lw: 1.75 }], { ymax: 1, bands: [{ a: HR - w5, b: HR + w5, color: K.pulse, alpha: 72 }, { a: HR / 2 - w5, b: HR / 2 + w5, color: K.mute, alpha: 72 }, { a: 2 * HR - w5, b: 2 * HR + w5, color: K.mute, alpha: 72 }], xlabel: "bpm" });
    const X = (v) => px + pw * (v - 30) / 170; txt("HR", X(HR) - 0.2, py - 0.21, 0.4, 0.2, { fs: 10.5, color: K.pulse, b: true }); txt("HR/2", X(HR / 2) - 0.2, py - 0.21, 0.42, 0.2, { fs: 10.5, color: K.mute }); txt("2·HR", X(2 * HR) - 0.25, py - 0.21, 0.5, 0.2, { fs: 10.5, color: K.mute }); txt("$M(f)$", X(D.nuis) + 0.06, py + 0.02, 0.56, 0.22, { fs: 11.5, color: K.noise, b: true, align: "left" });
    txt("One chunk of the example clip;\nnot a training clip", 7.0, 6.62, 2.62, 0.48, { fs: 10.5, color: K.mute, i: true });
  });
  group("losses", () => {
    txt("**Loss terms** (weight)", 9.74, 3.92, 3.3, 0.26, { fs: 12.5, align: "left" });
    [["$L_{wave}$", "Neg-Pearson with the PPG (1.0)", K.ink], ["$L_{spec}$", "heart-rate band likelihood (1.0)", K.pulse], ["$L_{half}$, $L_{dbl}$", "power near HR/2, 2·HR (0.5 each)", K.mute], ["$L_{excl}$", "power under $M(f)$ (1.0)", K.noise], ["$L_{ps}$", "same likelihood on $s_{p}$ (0.5)", K.pulse], ["$L_{nuis}$", "$s_{n}$ spectrum follows $n(t)$ (1.0)", K.noise], ["$L_{decor}$", "corr²($s_{p}$, $s_{n}$) (0.25)", K.mute]].forEach((r, i) => { const y = 4.2 + i * 0.38; txt(r[0], 9.74, y, 0.98, 0.36, { fs: 12.5, color: r[2], b: true, align: "left" }); txt(r[1], 10.72, y, 2.34, 0.36, { fs: 10.5, align: "left" }); });
    txt("Chunks without $n(t)$ use the masked reference spectrum", 9.74, 6.88, 3.32, 0.34, { fs: 10, color: K.mute, align: "left" });
  });
}

// ======================= 그림 5: 여러 합성 얼굴 × PPG 뱅크 × 잡음, 심박 스펙트럼과 잡음 스펙트럼의 분리 =======================
function figSeparation() {
  const SY = JSON.parse(fs.readFileSync(`${A}/syn_data.json`)), SE = JSON.parse(fs.readFileSync(`${A}/sep_data.json`)), SL = JSON.parse(fs.readFileSync(`${A}/sep_data_noproj.json`)); const HO = SE.held_out, HL = SL.held_out; const nS = SE.seeds.length;
  const off = [1, 2, 3, 4].map((i) => HO[`STRESS_OFFBAND6/${i}`]);
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length; const gaps = [...new Set(Object.values(HO).filter((r) => r.kind === "flicker").map((r) => Math.round(r.gap)))].sort((a, b) => a - b);
  const byGap = (H, sel) => gaps.map((g) => mean(Object.values(H).filter((r) => r.kind === "flicker" && Math.round(r.gap) === g).map(sel))); const nGap = gaps.map((g) => Object.values(HO).filter((r) => r.kind === "flicker" && Math.round(r.gap) === g).length);
  const hS = byGap(HO, (r) => r.hit_hr.out), hL = byGap(HL, (r) => r.hit_hr.out), hP = byGap(HO, (r) => r.hit_hr.pub); const motOf = (H, sel) => mean(Object.values(H).filter((r) => r.kind === "motion").map(sel)); const pc = (v) => (v * 100).toFixed(0);
    begin("Figure 5: Synthetic data and spectrum separation", `Figure 5. (a) All five generated identities of the training pool, two real clips each, and the four identities used only in test clips. Frames come from the clips themselves, chosen as the first frame after a fixed time where a face is detected. (b) Twelve of the ${SY.bank.n} real PPG segments, evenly spaced in rate rank. (c) Nuisance traces n(t) of three training clips. (d) The 80 PPG-injected clips of the 366-clip pool; 48 are rendered twice. (e) Five of the 48 dual-render clips, covering all five identities, one 160-frame training chunk each, spectra from the loss code (1 bpm grid): the injected PPG gives the heart-rate label and its band (shaded, plus or minus 3 bpm), the pulse-off render gives the noise label, whereas in the video trace the two are mixed. All training chunks have the heart-rate label, including the real clips, so only the noise label needs the dual render; it exists for these 48 clips (288 of 3,665 chunks per epoch), and other chunks use the clip's own traces. Chunk labels follow the injected waveform, so they can differ from the clip's target rate. (f) How the labels guide training; the output exclusion uses a mask from the clip's own noise references, not the dual-render label. Measured on the final model (five paired seeds): turning off the dual-render noise label changes no result, and removing the synthetic clips lowers the MMPD error (9.07 to 8.13 bpm) while the VIPL-HR gain from them (6.22 to 5.98) does not survive correction; on the synthetic test clips of (g) and (h) the synthetic training clips do help (readouts within 6 bpm at a 30 bpm gap: 91 % with, 55 % without). The figure describes how the labels are used, not that they improve real-video accuracy. (g) All four clips of one test set outside the pool, flicker 30 bpm from the heart rate; mean spectra over all chunks and ${nS} seeds. (h) All 16 flicker test clips, mean per gap (clips per gap: ${nGap.join(", ")}): share of chunk and seed readouts within 6 bpm of the heart rate. Within about 5 bpm the output falls on about twice the heart rate, with or without the projection. The public weights reach 100 % at gap 0 only because they read a flicker that sits on the heart rate. Without the projection the same training already separates the two at wide gaps (${pc(hL[hL.length - 1])} % vs. ${pc(hS[hS.length - 1])} % at gap 30), and the projection lowers the score at small gaps, so the separation shown here is mostly the training. Periodic-motion clips at gap 6 bpm, not plotted: SpecPhys ${pc(motOf(HO, (r) => r.hit_hr.out))} %, no projection ${pc(motOf(HL, (r) => r.hit_hr.out))} %, public ${pc(motOf(HO, (r) => r.hit_hr.pub))} %. These are controlled synthetic clips; they do not show that lock-in falls on real evaluation sets, where it does not.`);
  // ---- (a) 얼굴
  frame(0.14, 0.12, 4.16, 3.2, { fill: K.gray3, noline: true, name: "panel_a" });
  txt("**(a) Generated faces**", 0.22, 0.15, 4, 0.28, { fs: 13.5, align: "left" });
  group("faces_pool", () => { txt("Training pool: 5 identities", 0.24, 0.43, 3.9, 0.22, { fs: 11, align: "left", color: K.mute }); const s = 0.74, g = 0.05, x0 = 0.26, y0 = 0.67; SY.pool_tiles.forEach((t, i) => { const col = Math.floor(i / 2), row = i % 2; img(t.file, x0 + col * (s + g), y0 + row * (s + g), s, s, { alt: "Generated face from a training clip" }); }); });
  group("faces_test", () => { txt("Test clips only: 4 identities", 0.24, 2.27, 2.1, 0.22, { fs: 11, align: "left", color: K.mute }); const s = 0.54, g = 0.05, x0 = 0.26, y0 = 2.5; SY.test_tiles.forEach((t, i) => img(t.file, x0 + i * (s + g), y0, s, s, { alt: "Generated face from a test clip" })); txt("SFHQ and SDXL faces,\nnot real people, animated\nby LivePortrait", 2.64, 2.46, 1.66, 0.62, { fs: 10, color: K.mute }); });
  txt("×", 4.3, 1.5, 0.3, 0.4, { fs: 22, b: true, name: "times1" });
  // ---- (b) PPG 뱅크
  frame(4.6, 0.12, 3.14, 3.2, { fill: K.pulseF, noline: true, name: "panel_b" });
  txt("**(b) Real PPG bank**", 4.68, 0.15, 3, 0.28, { fs: 13.5, align: "left" });
  txt(`${SY.bank.n} segments: PURE ${SY.bank.src_counts.PURE}, UBFC-rPPG ${SY.bank.src_counts.UBFC}`, 4.7, 0.43, 3, 0.22, { fs: 11, align: "left", color: K.mute });
  group("bank", () => { const w = 0.93, h = 0.58, gx = 0.05, gy = 0.06, x0 = 4.7, y0 = 0.7; SY.bank.tiles.forEach((t, i) => { const c = i % 3, r = Math.floor(i / 3), x = x0 + c * (w + gx), y = y0 + r * (h + gy); frame(x, y, w, h, { fill: K.white, line: K.pulse, lw: 0.5, name: "tile" }); wave(t.wave, x + 0.04, y + 0.05, w - 0.08, h - 0.25, { lw: 1, name: "ppg" }); txt(`${t.bpm.toFixed(0)} bpm`, x, y + h - 0.2, w, 0.18, { fs: 9.5, color: K.mute, name: "rate" }); }); });
  txt("×", 7.72, 1.5, 0.3, 0.4, { fs: 22, b: true, name: "times2" });
  // ---- (c) 잡음
  frame(8.02, 0.12, 2.5, 3.2, { fill: K.noiseF, noline: true, name: "panel_c" });
  txt("**(c) Generated noise**", 8.1, 0.15, 2.4, 0.28, { fs: 13.5, align: "left" });
  txt("Traces $n(t)$ of three clips", 8.12, 0.43, 2.4, 0.22, { fs: 11, align: "left", color: K.mute });
  group("noise", () => { const L = [["296", "flicker"], ["276", "periodic motion"], ["265", "flicker + real head path"]]; L.forEach((q, i) => { const c = SY.clips[q[0]], y = 0.72 + i * 0.68; wave(c.n, 8.14, y, 2.26, 0.4, { color: K.noise, lw: 1, name: "n_trace" }); txt(q[1], 8.1, y + 0.4, 2.34, 0.2, { fs: 10, color: K.mute, name: "ntype" }); }); txt("+ drift, rotation, sensor noise", 8.1, 2.8, 2.34, 0.24, { fs: 11, color: K.ink }); });
  blockArrow(10.54, 1.5, 0.34, 0.36);
  // ---- (d) 클립
  frame(10.9, 0.12, 2.3, 3.2, { line: K.ours, lw: 1, name: "panel_d" });
  txt("**(d) Synthetic clips**", 10.98, 0.15, 2.2, 0.28, { fs: 13.5, align: "left" });
  group("cards", () => { const fl = ["tile_sfhq_01_0.png", "tile_sfhq_05_0.png", "tile_sfhq_06_0.png", "tile_sfhq_03_0.png", "tile_sfhq_02_0.png"]; for (let i = 4; i >= 0; i--) img(fl[i], 11.04 + i * 0.25, 0.5 + i * 0.13, 0.92, 0.92, { alt: "Frame of a PPG-injected clip", bc: K.line }); });
  txt("**80** clips in the pool", 10.96, 2.02, 2.2, 0.26, { fs: 12.5 });
  txt("each with a known PPG;\n**48** rendered twice, so the\nnoise trace is known too", 10.96, 2.3, 2.2, 0.72, { fs: 11.5 });
  // ---- (e) 클립마다 분리된 두 스펙트럼 라벨
  const By = 3.44, eW = 4.56;
  frame(0.14, By, eW, 3.92, { fill: K.white, line: K.ours, lw: 1.5, name: "panel_e" });
  txt("**(e) Each dual-render clip gives two labels**", 0.22, By + 0.03, eW - 0.1, 0.28, { fs: 13.5, align: "left", color: K.ours });
  txt("Injected PPG and generated noise are known, chunk by chunk", 0.22, By + 0.31, eW - 0.12, 0.2, { fs: 10.5, align: "left", color: K.mute });
  group("label_table", () => {
    const col = { tile: 0.22, ppg: 0.74, mix: 1.56, hr: 2.56, noise: 3.62 }, wP = 0.76, wM = 0.86, wL = 0.98, y0 = By + 1.02, pitch = 0.53, h = 0.44, nn = (a) => { const m = Math.max(...a); return a.map((v) => v / m); };
    txt("Clip", col.tile, By + 0.78, 0.48, 0.2, { fs: 10, b: true }); txt("Injected\nPPG", col.ppg, By + 0.62, wP, 0.36, { fs: 10, b: true, color: K.pulse }); txt("Video trace\n(mixed)", col.mix, By + 0.62, wM, 0.36, { fs: 10, b: true });
    box("**Training labels**", col.hr, By + 0.54, col.noise + wL - col.hr, 0.22, { fill: K.ours, line: K.ours, color: K.white, fs: 10.5, margin: 0, r: 0.05 });
    txt("Heart rate", col.hr, By + 0.78, wL, 0.2, { fs: 10.5, b: true, color: K.pulse }); txt("Noise", col.noise, By + 0.78, wL, 0.2, { fs: 10.5, b: true, color: K.noise });
    SY.label_rows.forEach((r, i) => { const y = y0 + i * pitch, last = i === SY.label_rows.length - 1, f = r.f; const X = (x, w, v) => x + w * (v - 30) / 170;
      img(r.file, col.tile, y, 0.46, 0.46, { alt: "Frame of a dual-render synthetic clip", bc: K.line, blw: 0.5 });
      wave(r.ppg, col.ppg, y + 0.04, wP, h - 0.08, { n: 120, lw: 0.9, name: "ppg_label" });
      spectrumPlot(col.mix, y, wM, h, f, [{ y: nn(r.mix), color: K.arrow, lw: 1 }], { ymax: 1, ticks: last ? [60, 180] : [], tfs: 9 });
      line(col.mix + wM + 0.02, y + h / 2, col.hr - 0.02, y + h / 2, { lw: 1, name: "split" });
      spectrumPlot(col.hr, y, wL, h, f, [{ y: nn(r.hr), color: K.pulse, lw: 1.25, closed: true, fill: K.pulseF }], { ymax: 1, ticks: last ? [60, 180] : [], tfs: 9, bands: [{ a: r.hr_label - 3, b: r.hr_label + 3, color: K.pulse, alpha: 55 }] });
      spectrumPlot(col.noise, y, wL, h, f, [{ y: nn(r.noise), color: K.noise, lw: 1.25, closed: true, fill: K.noiseF }], { ymax: 1, ticks: last ? [60, 180] : [], tfs: 9 });
      const lab = (x, w, v, str, c) => { const right = X(x, w, v) < x + w * 0.56; txt(str, right ? x + w - 0.5 : x, y - 0.02, 0.5, str.includes("\n") ? 0.3 : 0.16, { fs: 9, b: true, color: c, align: right ? "right" : "left", name: "value" }); };
      lab(col.hr, wL, r.hr_label, `${r.hr_label.toFixed(0)}`, K.pulse); lab(col.noise, wL, r.noise_defined, `${r.kind}\n${r.noise_defined.toFixed(0)}`, K.noise); });
    txt("bpm", col.mix, y0 + 5 * pitch - 0.0, col.noise + wL - col.mix, 0.16, { fs: 9, color: K.mute, name: "bpm" });
  });
  // ---- 라벨에서 안내로
  const fX = 0.14 + eW + 0.2, fW = 2.3;
  blockArrow(0.14 + eW + 0.01, By + 0.86, 0.19, 0.3, { color: K.pulse }); blockArrow(0.14 + eW + 0.01, By + 1.94, 0.19, 0.3, { color: K.noise });
  // ---- (f) 안내
  frame(fX, By, fW, 3.92, { fill: K.oursT, noline: true, name: "panel_f" });
  txt("**(f) Spectrum guidance**", fX + 0.07, By + 0.03, fW - 0.1, 0.28, { fs: 13.5, align: "left" });
  group("guide", () => {
    const x = fX + 0.08, w = fW - 0.16, cx = x + w / 2;
    box("**Heart-rate label**\noutput and pulse basis must peak in its band\n$L_{spec}$, $L_{ps}$, $L_{half}$, $L_{dbl}$", x, By + 0.34, w, 1.08, { fill: K.pulseF, line: K.pulse, fs: 10.5, margin: 2 });
    box("**Noise label**\nnoise basis must follow it\n$L_{nuis}$\nOutput stays off the clip's noise references: $L_{excl}$", x, By + 1.48, w, 1.28, { fill: K.noiseF, line: K.noise, fs: 10.5, margin: 2 });
    line(cx, By + 2.76, cx, By + 2.88, { lw: 1.25 });
    box("Noise-subspace projection", x, By + 2.88, w, 0.34, { fill: K.ours, line: K.ours, color: K.white, fs: 10.5, b: true, margin: 0 }); line(cx, By + 3.22, cx, By + 3.32, { lw: 1 });
    box("Pulse basis $s_{p}$", x, By + 3.32, w / 2 - 0.03, 0.34, { fill: K.pulseF, line: K.pulse, fs: 10.5, margin: 0 }); box("Noise basis $s_{n}$", x + w / 2 + 0.03, By + 3.32, w / 2 - 0.03, 0.34, { fill: K.noiseF, line: K.noise, fs: 10.5, margin: 0 });
    txt("attention from the pulse basis only", fX + 0.02, By + 3.68, fW - 0.04, 0.2, { fs: 9.5, color: K.mute });
  });
  const gX = fX + fW + 0.2, gW = 3.62; blockArrow(fX + fW + 0.01, By + 1.74, 0.19, 0.32, { color: K.ours });
  // ---- (g) 분리 결과
  frame(gX, By, gW, 3.92, { fill: K.white, line: K.gray, lw: 1, name: "panel_g" });
  txt("**(g) Separated on test clips**", gX + 0.07, By + 0.03, gW - 0.1, 0.28, { fs: 13.5, align: "left" });
  group("legend_g", () => { const y = By + 0.34, x = gX + 0.08; seg(x, y + 0.09, x + 0.2, y + 0.09, { color: K.arrow, lw: 1.5, name: "key" }); txt("before projection", x + 0.22, y, 1.06, 0.18, { fs: 9.5, align: "left" }); seg(x + 1.3, y + 0.09, x + 1.5, y + 0.09, { color: K.pulse, lw: 2, name: "key" }); txt("pulse basis", x + 1.52, y, 0.74, 0.18, { fs: 9.5, align: "left", color: K.pulse, b: true }); seg(x + 2.28, y + 0.09, x + 2.48, y + 0.09, { color: K.noise, lw: 2, name: "key" }); txt("noise basis", x + 2.5, y, 0.8, 0.18, { fs: 9.5, align: "left", color: K.noise, b: true }); });
  const tileBy = { sfhq04_frames: "tile_test_1.png", diff_female_01_frames: "tile_test_2.png", diff_male_01_frames: "tile_test_3.png", diff_female_02_frames: "tile_test_4.png" };
  off.forEach((r, i) => group(`clip_${i}`, () => {
    const y = By + 0.58 + i * 0.8, h = 0.56, x = gX + 0.66, w = 1.86, f = r.f; const nn = (a) => { const m = Math.max(...a); return a.map((v) => v / m); };
    img(tileBy[r.face], gX + 0.07, y, 0.54, 0.54, { alt: "Generated face from a test clip" });
    spectrumPlot(x, y, w, h, f, [{ y: nn(r.S.before), color: K.gray, lw: 1 }, { y: nn(r.S.bp), color: K.pulse, lw: 1.5 }, { y: nn(r.S.bn), color: K.noise, lw: 1.5 }], { ymax: 1, marks: [{ v: r.hr, color: K.pulse }, { v: r.nuis, color: K.noise }], ticks: i === 3 ? [60, 120, 180] : [], xlabel: i === 3 ? "bpm" : undefined, tfs: 9.5 });
    txt(`HR **${r.hr.toFixed(0)}** bpm\nflicker **${r.nuis.toFixed(0)}**\noutput **${r.peak_mean.out.toFixed(0)}**\npublic **${r.peak_mean.pub.toFixed(0)}**`, x + w + 0.05, y - 0.06, gW - (x - gX) - w - 0.07, 0.72, { fs: 9.5, align: "left", name: "peaks" });
  }));
  // ---- (h) 필요한 간격
  const hX = gX + gW + 0.08, hW = 13.2 - hX;
  frame(hX, By, hW, 3.92, { fill: K.white, line: K.gray, lw: 1, name: "panel_h" });
  txt("**(h) Where it stops**", hX + 0.07, By + 0.03, hW - 0.1, 0.28, { fs: 13.5, align: "left" });
  group("gap", () => {
    const x0 = hX + 0.5, y0 = By + 0.4, w = hW - 0.6, h = 1.62, gmax = 32; const X = (g) => x0 + 0.08 + (w - 0.1) * g / gmax, Y = (v) => y0 + h - h * v;
    [0, 0.5, 1].forEach((v) => { seg(x0, Y(v), x0 + w, Y(v), { color: K.gray2, lw: 0.75, name: "grid" }); txt(`${(v * 100).toFixed(0)}`, x0 - 0.3, Y(v) - 0.09, 0.27, 0.18, { fs: 9.5, color: K.mute, align: "right" }); });
    txt("within 6 bpm of HR (%)", hX + 0.04, y0 - 0.04, 0.18, h + 0.08, { fs: 9.5, color: K.mute, rot: 270 });
    seg(x0, y0 + h, x0 + w, y0 + h, { lw: 1, name: "axis" }); [0, 10, 20, 30].forEach((g) => { seg(X(g), y0 + h, X(g), y0 + h + 0.04, { lw: 0.75 }); txt(`${g}`, X(g) - 0.2, y0 + h + 0.05, 0.4, 0.17, { fs: 9.5, color: K.mute }); }); txt("|HR − flicker| (bpm)", x0 - 0.2, y0 + h + 0.21, w + 0.4, 0.18, { fs: 9.5, color: K.mute });
    const dot = (cx, cy, c, d = 0.09, hollow) => CUR.addShape(S.OVAL, { x: cx - d / 2, y: cy - d / 2, w: d, h: d, fill: hollow ? { color: K.white } : { color: c }, line: { color: c, width: 1 }, objectName: nm("dot") });
    const curve = (vals, c, lw, hollow, name) => { poly(gaps.map((g, i) => [X(g), Y(vals[i])]), { color: c, lw, name }); gaps.forEach((g, i) => dot(X(g), Y(vals[i]), c, 0.09, hollow)); };
    curve(hP, K.arrow, 1, true, "public"); curve(hL, "7F9BD1", 1.25, false, "no_projection"); curve(hS, K.ours, 1.75, false, "specphys");
    const ly = By + 2.46; [["SpecPhys", K.ours, false], ["without projection", "7F9BD1", false], ["public FactorizePhys", K.arrow, true]].forEach((q, i) => { dot(hX + 0.16, ly + 0.09 + i * 0.18, q[1], 0.09, q[2]); txt(q[0], hX + 0.24, ly + i * 0.18, hW - 0.26, 0.18, { fs: 9.5, align: "left" }); });
    txt("Under 5 bpm the output falls on 2×HR. Public hits 100 % at 0 only by reading the flicker.", hX + 0.05, By + 3.04, hW - 0.1, 0.84, { fs: 10.5, color: K.pulse });
  });
}

figCompare(); figNetwork(); figModule(); figTrain(); figSeparation();
pres.writeFile({ fileName: OUTF }).then(async (f) => { await applyTheme(f, THEME); console.log("wrote", f, "shapes", N); });
