// [56 문서] fig_spec.json(scripts/paper_figs_nf.py 가 생성) → 편집 가능한 PPT 도형 그림 (pptxgenjs). 한 그림 = 한 슬라이드(13.33×7.5 in).
//   node scripts/paper_figs_nf.js  → _workspace/slides/paper_figs_specfp.pptx
// 모든 좌표·색·문구는 사양 JSON 에서만 온다(matplotlib 미리보기와 동일 기하).
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const SPEC = JSON.parse(fs.readFileSync("/SPECPHYS_ROOT/_workspace/data/results/figs_nf/fig_spec.json", "utf8"));
const C = SPEC.colors; const FONT = SPEC.font || "Arial";
const col = (k) => (C[k] || k);
const pres = new pptxgen(); pres.layout = "LAYOUT_WIDE"; pres.title = "SpecPhys paper figures"; pres.author = "Virtual-Face rPPG";

// 텍스트 마크업: a_{b} → 아래 첨자 run
function runs(s, base) {
  const out = []; let i = 0, buf = "";
  const flush = () => { if (buf) { out.push({ text: buf, options: Object.assign({}, base) }); buf = ""; } };
  while (i < s.length) {
    if (s[i] === "_" && s[i + 1] === "{") { const j = s.indexOf("}", i); flush(); out.push({ text: s.slice(i + 2, j), options: Object.assign({}, base, { subscript: true }) }); i = j + 1; }
    else if (s[i] === "\n") { flush(); out.push({ text: "", options: Object.assign({}, base, { breakLine: true }) }); i++; }
    else { buf += s[i]; i++; }
  }
  flush(); return out;
}

for (const fig of SPEC.figures) {
  const slide = pres.addSlide(); slide.background = { color: "FFFFFF" };
  for (const o of fig.items) {
    switch (o.kind) {
      case "panel": {
        slide.addShape(pres.ShapeType.roundRect, { x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.12, fill: { color: col(o.fill) }, line: { color: o.dash ? col("ink") : col("gray"), width: 1, dashType: o.dash ? "dash" : "solid" } });
        slide.addText(o.label, { x: o.x + 0.12, y: o.y + o.h - 0.42, w: o.w - 0.3, h: 0.36, fontFace: FONT, fontSize: 11.5, bold: true, color: col("ink"), margin: 0, isTextBox: true, valign: "bottom" });
        break; }
      case "rect": slide.addShape(pres.ShapeType.rect, { x: o.x, y: o.y, w: o.w, h: o.h, fill: { color: col(o.fill) }, line: { color: col(o.border || "gray"), width: 0.8 } }); break;
      case "box": {
        slide.addShape(pres.ShapeType.roundRect, { x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.08, fill: { color: col(o.fill) }, line: { color: col(o.border || "gray"), width: 0.8 } });
        slide.addText(runs(o.text, { fontFace: FONT, fontSize: o.fs * 0.85, bold: !!o.bold, color: col("ink") }), { x: o.x, y: o.y, w: o.w, h: o.h, align: "center", valign: "middle", margin: 2, isTextBox: true });
        break; }
      case "text": slide.addText(runs(o.text, { fontFace: FONT, fontSize: o.fs * 0.85, bold: !!o.bold, italic: !!o.italic, color: col(o.color) }), { x: o.x, y: o.y, w: o.w, h: o.h, align: o.align, valign: "top", margin: 2, isTextBox: true }); break;
      case "circle": slide.addShape(pres.ShapeType.ellipse, { x: o.x, y: o.y, w: o.d, h: o.d, fill: { color: col(o.fill) }, line: { color: col("ink"), width: 0.6 } }); break;
      case "arrow": case "line": case "dblarrow": {
        const lo = { color: col(o.color), width: o.width || 1.2, dashType: o.dash ? "dash" : "solid" };
        if (o.kind === "arrow") lo.endArrowType = "triangle"; if (o.kind === "dblarrow") { lo.endArrowType = "triangle"; lo.beginArrowType = "triangle"; }
        const x = Math.min(o.x1, o.x2), y = Math.min(o.y1, o.y2), w = Math.abs(o.x2 - o.x1), h = Math.abs(o.y2 - o.y1);
        const opts = { x, y, w: Math.max(w, 0.001), h: Math.max(h, 0.001), line: lo };
        if ((o.x2 < o.x1) !== (o.y2 < o.y1)) opts.flipV = true;   // 대각선 방향 보정
        if (o.x2 < o.x1 && o.y2 === o.y1) { opts.flipH = true; }
        if (o.y2 < o.y1 && o.x2 === o.x1) { opts.flipV = true; }
        slide.addShape(pres.ShapeType.line, opts);
        if (o.label) slide.addText(o.label, { x: Math.min(o.x1, o.x2), y: o.y1 - 0.3, w: Math.abs(o.x2 - o.x1), h: 0.28, fontFace: FONT, fontSize: 8, color: col(o.color), align: "center", margin: 0, isTextBox: true });
        break; }
      case "axis": {   // 스펙트럼 스케치: 축 배경 + HR 띠 + 꺾은선(선분)
        slide.addShape(pres.ShapeType.rect, { x: o.x, y: o.y, w: o.w, h: o.h, fill: { color: "FFFFFF" }, line: { color: col("gray"), width: 0.6 } });
        const f0 = 30, f1 = 200, fx = (f) => o.x + 0.15 + (f - f0) / (f1 - f0) * (o.w - 0.3);
        slide.addShape(pres.ShapeType.rect, { x: fx(o.hr - o.band), y: o.y + 0.15, w: fx(o.hr + o.band) - fx(o.hr - o.band), h: o.h - 0.45, fill: { color: col("tokY"), transparency: 40 }, line: { color: col("tokY"), width: 0 } });
        const N = 60; const pts = []; let mx = 0;
        for (let i = 0; i <= N; i++) { const f = f0 + (f1 - f0) * i / N; let v = 0.05; for (const [pf, pa] of o.peaks) v += pa * Math.exp(-0.5 * ((f - pf) / 4) ** 2); pts.push([f, v]); if (v > mx) mx = v; }
        for (let i = 0; i < N; i++) { const [fa, va] = pts[i], [fb, vb] = pts[i + 1]; const ya = o.y + o.h - 0.3 - va / mx * (o.h - 0.5), yb = o.y + o.h - 0.3 - vb / mx * (o.h - 0.5);
          slide.addShape(pres.ShapeType.line, { x: fx(fa), y: Math.min(ya, yb), w: Math.max(fx(fb) - fx(fa), 0.001), h: Math.max(Math.abs(yb - ya), 0.001), flipV: yb < ya, line: { color: col("accent"), width: 1.2 } }); }
        slide.addShape(pres.ShapeType.line, { x: o.x + 0.15, y: o.y + o.h - 0.3, w: o.w - 0.3, h: 0.001, line: { color: col("ink"), width: 0.8 } });
        for (const fq of [30, 60, 120, 200]) slide.addText(String(fq), { x: fx(fq) - 0.2, y: o.y + o.h - 0.3, w: 0.4, h: 0.22, fontFace: FONT, fontSize: 6.5, color: col("muted"), align: "center", margin: 0, isTextBox: true });
        slide.addText("HR", { x: fx(o.hr) - 0.2, y: o.y + 0.0, w: 0.4, h: 0.2, fontFace: FONT, fontSize: 7, color: col("ink"), align: "center", margin: 0, isTextBox: true });
        slide.addText(runs(o.label, { fontFace: FONT, fontSize: 7.5, color: col("muted") }), { x: o.x, y: o.y + o.h + 0.02, w: o.w, h: 0.28, align: "center", margin: 0, isTextBox: true });
        break; }
      case "grid": {
        const n = o.n, cw = o.w / n;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
          let v; if (o.kind2 === "skin") v = (Math.abs(i - n / 2 + 0.5) < n * 0.33 && Math.abs(j - n / 2 + 0.5) < n * 0.4) ? 1 : 0.15;
          else v = Math.exp(-0.5 * (((i - n / 2 + 0.5) / (n * 0.28)) ** 2 + ((j - n / 2 + 0.2) / (n * 0.3)) ** 2));
          const rgb = o.kind2 === "skin" ? [255 - 128 * v, 255 - 64 * v, 255 - 26 * v] : [242 - 153 * v, 242 - 64 * v, 153 + 51 * (1 - v)];
          const hex = rgb.map((c) => Math.max(0, Math.min(255, Math.round(c))).toString(16).padStart(2, "0")).join("");
          slide.addShape(pres.ShapeType.rect, { x: o.x + j * cw, y: o.y + i * cw, w: cw, h: cw, fill: { color: hex }, line: { color: "FFFFFF", width: 0.5 } });
        }
        slide.addText(runs(o.label, { fontFace: FONT, fontSize: 7.5, color: col("muted") }), { x: o.x - 0.3, y: o.y + o.h + 0.03, w: o.w + 0.6, h: 0.45, align: "center", margin: 0, isTextBox: true });
        break; }
      case "table": {
        const nc = o.header.length; const cw0 = o.w * 0.26; const cw = (o.w - cw0) / (nc - 1); const colW = [cw0].concat(Array(nc - 1).fill(cw));
        const hdr = o.header.map((h) => ({ text: h, options: { bold: true, fill: { color: col("lav") }, color: col("ink"), align: "center", valign: "middle", fontSize: o.fs, fontFace: FONT } }));
        const body = o.rows.map((r, ri) => r.map((v, ci) => ({ text: String(v), options: { bold: ci > 0 && o.best[ci - 1] === ri, color: col("ink"), align: ci === 0 ? "left" : "center", valign: "middle", fontSize: o.fs, fontFace: FONT, fill: { color: ri % 2 ? "F6F8FC" : "FFFFFF" } } })));
        slide.addTable([hdr].concat(body), { x: o.x, y: o.y, w: o.w, h: o.h, colW, border: { type: "solid", pt: 0.4, color: col("gray") }, margin: 0.03 });
        break; }
      case "chart": {
        const data = o.series.map(([nm, vals]) => ({ name: nm, labels: o.cats, values: vals.map((v) => (v == null ? 0 : Math.round(v * 100) / 100)) }));
        slide.addChart(pres.ChartType.bar, data, { x: o.x, y: o.y, w: o.w, h: o.h, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
          chartColors: o.series.map(([, , c]) => col(c)), showLegend: true, legendPos: "t", legendFontSize: 8, legendFontFace: FONT,
          showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 6.5, dataLabelFontFace: FONT, dataLabelFormatCode: "0.0", dataLabelColor: col("ink"),
          catAxisLabelFontSize: 8, catAxisLabelFontFace: FONT, catAxisLabelColor: col("ink"), valAxisLabelFontSize: 8, valAxisLabelFontFace: FONT, valAxisLabelColor: col("muted"),
          valAxisTitle: o.ylabel, showValAxisTitle: true, valAxisTitleFontSize: 8, valAxisTitleFontFace: FONT, valAxisTitleColor: col("muted"),
          valGridLine: { color: col("gray"), size: 0.4 }, catGridLine: { style: "none" }, plotArea: { fill: { color: "FFFFFF" } }, chartArea: { fill: { color: "FFFFFF" }, roundedCorners: false } });
        break; }
    }
  }
  slide.addNotes(fig.title);
}
const OUT = "/SPECPHYS_ROOT/_workspace/slides/paper_figs_specfp.pptx";
pres.writeFile({ fileName: OUT }).then(() => console.log("-> " + OUT));
