#!/usr/bin/env python3
"""[체인 68, 49 문서] 플리커 개정판 코퍼스 COTRAIN_ALLF_TWFR — 공식 코퍼스 203 과 클립 단위로 일치하되, 플리커가 있는 49 클립의 플리커만
주파수 분산(0.8/1.0/1.3/1.6/1.9/2.2/2.5 Hz 순환, 클립 인덱스로 결정) + 진폭 0.02 로 바꾼다(드리프트·움직임·센서·시드 동일; make_noflk_pool.py 와 같은 구조).
마커 /tmp/fr_pool.done"""
import glob, json, os, shutil, subprocess, sys
import numpy as np, cv2
ROOT = "/SPECPHYS_ROOT"; PFA = f"{ROOT}/_workspace/data/physformer"
sys.path.insert(0, f"{ROOT}/scripts")
from make_noisy_real import corrupt  # noqa
from make_noisy_real_train import VARIANTS  # noqa
PY = sys.executable; BUILD = f"{ROOT}/generator/build_synth_dataset.py"
HZ = [0.8, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5]; AMP = 0.02
def revise(nz, k):
    nz = dict(nz)
    if nz.get("flicker_amp", 0) > 0: nz["flicker_hz"] = HZ[k % len(HZ)]; nz["flicker_amp"] = AMP
    return nz
def frames_equal(a, b, n=60):
    ca, cb = cv2.VideoCapture(a), cv2.VideoCapture(b); d = []
    for _ in range(n):
        oa, fa = ca.read(); ob, fb = cb.read()
        if not (oa and ob): break
        d.append(np.abs(fa.astype(np.float32) - fb.astype(np.float32)).mean())
    ca.release(); cb.release(); return float(np.mean(d)) if d else 1e9
def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst): os.remove(dst) if os.path.islink(dst) else shutil.rmtree(dst)
    os.symlink(os.path.realpath(src), dst)
log = []
# 1. 실 잡음 사본
OUT = f"{PFA}/COTRAIN_NOISY_REAL_FR"; os.makedirs(OUT, exist_ok=True)
subs = [s for s in sorted(glob.glob(f"{PFA}/COTRAIN/subject*")) if int(os.path.basename(s).replace("subject", "")) < 200]; assert len(subs) == 34
checked = False
for i, s in enumerate(subs):
    name = os.path.basename(s); d = f"{OUT}/{name}"; nz0 = dict(VARIANTS[i % len(VARIANTS)]); has_fl = nz0.get("flicker_amp", 0) > 0; nz = revise(nz0, i)
    if os.path.exists(f"{d}/vid.avi") or (os.path.islink(d) and os.path.exists(d)): continue
    if not has_fl and checked: link(f"{PFA}/COTRAIN_NOISY_REAL/{name}", d); continue
    os.makedirs(d, exist_ok=True); n = corrupt(os.path.realpath(f"{s}/vid.avi"), f"{d}/vid.avi", nz, seed=800 + i); shutil.copy(os.path.realpath(f"{s}/ground_truth.txt"), f"{d}/ground_truth.txt")
    if not has_fl:
        md = frames_equal(f"{d}/vid.avi", f"{PFA}/COTRAIN_NOISY_REAL/{name}/vid.avi"); print(f"[real] mapping check {name}: mean|diff|={md:.4f}"); assert md < 0.5; checked = True
    if has_fl: log.append(("real", name, nz["flicker_hz"], nz["flicker_amp"]))
    print(f"[real] {name}: {n} frames, variant {i % 5}, flicker={nz.get('flicker_hz') if has_fl else None}", flush=True)
# 2. 가상 세트
for cfg, src, dst in [("train20h_tw.json", "TRAIN20H_TW", "TRAIN20H_TWFR"), ("train6hi_tw.json", "TRAIN6HI_TW", "TRAIN6HI_TWFR"), ("train6lo_tw.json", "TRAIN6LO_TW", "TRAIN6LO_TWFR")]:
    c = json.load(open(f"{PFA}/configs/{cfg}")); os.makedirs(f"{PFA}/{dst}", exist_ok=True); fl = []
    for sub in c["subjects"]:
        nz = sub.get("noise") or {}
        if nz.get("flicker_amp", 0) > 0:
            if os.path.exists(f"{PFA}/{dst}/subject{sub['id']}/vid.avi"): continue
            sub = json.loads(json.dumps(sub)); sub["noise"] = revise(sub["noise"], int(sub["id"]) + 100); fl.append(sub); log.append((dst, sub["id"], sub["noise"]["flicker_hz"], sub["noise"]["flicker_amp"]))
        else: link(f"{PFA}/{src}/subject{sub['id']}", f"{PFA}/{dst}/subject{sub['id']}")
    if fl:
        c2 = dict(c); c2["subjects"] = fl; c2["out"] = f"_workspace/data/physformer/{dst}"; tmp = f"{PFA}/configs/{cfg.replace('.json', 'fr.json')}"; json.dump(c2, open(tmp, "w"), indent=1)
        print(f"[syn] build {dst}: {len(fl)} flicker clips", flush=True); subprocess.run([PY, BUILD, "--config", tmp], cwd=ROOT, check=True)
# 3. 압축 사본
D = f"{PFA}/CMP_SYN_TWFR"; os.makedirs(D, exist_ok=True)
for i, k in enumerate([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 1):
    d = f"{D}/subject{600 + i}"; srcdir = f"{PFA}/TRAIN20H_TWFR/subject{k}"
    if os.path.islink(srcdir): link(f"{PFA}/CMP_SYN_TW/subject{600 + i}", d); continue
    if os.path.exists(f"{d}/vid.avi"): continue
    os.makedirs(d, exist_ok=True); subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{srcdir}/vid.avi", "-c:v", "mpeg4", "-q:v", "6", f"{d}/vid.avi"], check=True); shutil.copy(f"{srcdir}/ground_truth.txt", f"{d}/ground_truth.txt")
# 4. 풀
P = f"{PFA}/COTRAIN_ALLF_TWFR"; shutil.rmtree(P, ignore_errors=True); os.makedirs(P)
rep = {"COTRAIN_NOISY_REAL": "COTRAIN_NOISY_REAL_FR", "TRAIN20H_TW": "TRAIN20H_TWFR", "TRAIN6HI_TW": "TRAIN6HI_TWFR", "TRAIN6LO_TW": "TRAIN6LO_TWFR", "CMP_SYN_TW": "CMP_SYN_TWFR"}; nrep = 0
for s in sorted(os.listdir(f"{PFA}/COTRAIN_ALLF_TW")):
    t = os.path.realpath(f"{PFA}/COTRAIN_ALLF_TW/{s}"); dn = os.path.basename(os.path.dirname(t)); tgt = t
    if dn in rep: tgt = f"{PFA}/{rep[dn]}/{os.path.basename(t)}"; nrep += 1
    assert os.path.exists(f"{tgt}/vid.avi"), tgt; os.symlink(os.path.realpath(tgt), f"{P}/{s}")
json.dump(log, open(f"{PFA}/COTRAIN_ALLF_TWFR_flicker_log.json", "w"), indent=0)
print(f"POOL ALLF_TWFR n={len(os.listdir(P))} replaced={nrep} flicker_clips={len(log)} hz_hist={ {h: sum(1 for l in log if l[2]==h) for h in HZ} }")
open("/tmp/fr_pool.done", "w").write("ok\n"); print("[fr] done")
