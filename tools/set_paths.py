#!/usr/bin/env python3
"""Replace the path placeholders of this repository with local paths (edits files in place; docs/ is left unchanged).

  python tools/set_paths.py --root /abs/SpecPhys --raw /abs/datasets --cache /abs/cache --conda /abs/miniforge3 --tmp /abs/tmp [--home ~]

Placeholders: /SPECPHYS_ROOT (this repository), /SPECPHYS_RAW (raw datasets), /SPECPHYS_CACHE (large disk for frame caches),
/SPECPHYS_CONDA (conda root with the vface-rppg and vface-gen environments), /SPECPHYS_TMP (temporary files), /SPECPHYS_HOME (home directory).
Placeholders that are not given stay in place, so the tool can be run again later."""
import argparse
import os
import re

ap = argparse.ArgumentParser()
for k in ("root", "raw", "cache", "conda", "tmp", "home"):
    ap.add_argument(f"--{k}")
a = ap.parse_args()
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
given = {"/SPECPHYS_ROOT": a.root, "/SPECPHYS_RAW": a.raw, "/SPECPHYS_CACHE": a.cache, "/SPECPHYS_CONDA": a.conda, "/SPECPHYS_TMP": a.tmp, "/SPECPHYS_HOME": a.home}
M = {k: os.path.abspath(os.path.expanduser(v)).rstrip("/") for k, v in given.items() if v}
if not M:
    ap.error("give at least one path")
pat = re.compile("|".join(re.escape(k) for k in sorted(M, key=len, reverse=True)) + r"(?=[/\"'\s:;,()`]|$)")
EXT = (".py", ".sh", ".js", ".yaml", ".yml", ".json", ".txt", ".patch", ".cfg", ".parked")
SELF = os.path.abspath(__file__)
changed = 0
for dp, dn, fn in os.walk(here):
    parts = os.path.relpath(dp, here).split(os.sep)
    if ".git" in parts or parts[0] == "docs":
        continue
    for f in fn:
        if not f.endswith(EXT):
            continue
        p = os.path.join(dp, f)
        if os.path.abspath(p) == SELF:
            continue
        t = open(p, encoding="utf-8", errors="ignore").read()
        u = pat.sub(lambda m: M[m.group(0)], t)
        if u != t:
            open(p, "w", encoding="utf-8").write(u)
            changed += 1
print(f"set {', '.join(f'{k} -> {v}' for k, v in M.items())}; {changed} files changed")
