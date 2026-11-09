#!/usr/bin/env python3
"""[63 문서] pptxgenjs 는 도형 묶기가 없으므로, objectName 이 "묶음|이름" 인 연속 도형을 <p:grpSp> 로 감싼다(도형은 그대로, 좌표 변환 없음).
  python scripts/paper_arch_figs_group.py deck.pptx   (제자리 수정)"""
import re, sys, zipfile, shutil, os
EL = re.compile(r"<p:(sp|pic|cxnSp)>.*?</p:\1>", re.S)
def group_slide(x):
    a = x.index("</p:grpSpPr>") + len("</p:grpSpPr>"); b = x.rindex("</p:spTree>"); head, body, tail = x[:a], x[a:b], x[b:]
    els = [m.group(0) for m in EL.finditer(body)]; assert "".join(els) == body.strip() or len("".join(els)) == len(body.replace("\n", "")), "unexpected content in spTree"
    ids = [int(i) for i in re.findall(r'<p:cNvPr id="(\d+)"', x)]; nid = max(ids) + 1000; out = []; i = 0; made = 0
    def pref(e):
        m = re.search(r'<p:cNvPr id="\d+" name="([^"]*)"', e); return m.group(1).split("|")[0] if m and "|" in m.group(1) else None
    while i < len(els):
        p = pref(els[i])
        if p is None: out.append(els[i]); i += 1; continue
        j = i
        while j + 1 < len(els) and pref(els[j + 1]) == p: j += 1
        kids = [re.sub(r'(<p:cNvPr id="\d+" name=")[^"|]*\|', r"\1", e, count=1) for e in els[i:j + 1]]
        if len(kids) == 1: out.append(kids[0]); i = j + 1; continue
        X0 = Y0 = 10 ** 12; X1 = Y1 = -10 ** 12
        for e in kids:
            m = re.search(r'<a:off x="(-?\d+)" y="(-?\d+)"\s*/>\s*<a:ext cx="(\d+)" cy="(\d+)"', e); xx, yy, cx, cy = map(int, m.groups()); X0 = min(X0, xx); Y0 = min(Y0, yy); X1 = max(X1, xx + cx); Y1 = max(Y1, yy + cy)
        out.append(f'<p:grpSp><p:nvGrpSpPr><p:cNvPr id="{nid}" name="{p}"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="{X0}" y="{Y0}"/><a:ext cx="{X1 - X0}" cy="{Y1 - Y0}"/><a:chOff x="{X0}" y="{Y0}"/><a:chExt cx="{X1 - X0}" cy="{Y1 - Y0}"/></a:xfrm></p:grpSpPr>' + "".join(kids) + "</p:grpSp>"); nid += 1; made += 1; i = j + 1
    return head + "".join(out) + tail, made
f = sys.argv[1]; t = f + ".tmp"; tot = 0
with zipfile.ZipFile(f) as zi, zipfile.ZipFile(t, "w", zipfile.ZIP_DEFLATED) as zo:
    for it in zi.infolist():
        d = zi.read(it.filename)
        if re.fullmatch(r"ppt/slides/slide\d+\.xml", it.filename): s, n = group_slide(d.decode("utf8")); d = s.encode("utf8"); tot += n
        zo.writestr(it, d)
shutil.move(t, f); print("grouped", tot)
