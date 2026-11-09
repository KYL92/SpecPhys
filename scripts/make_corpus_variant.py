#!/usr/bin/env python3
"""[체인 69, 50 문서] 코퍼스 교란 설정 변형 생성기 — 공식 코퍼스 203 과 클립 단위로 일치하는 짝 풀 COTRAIN_ALLF_TW<NAME> 을 만든다.
  python scripts/make_corpus_variant.py NAME --flicker keep|broad|mix --amp keep|0.03 --motion 1.0 --drift 1.0
    --flicker broad : 플리커 클립 49 의 주파수를 0.8/1.0/1.3/1.6/1.9/2.2/2.5 Hz 순환(클립 인덱스)으로
    --flicker mix   : 실 잡음 사본(27)은 공식 그대로(1.1 Hz 근방·진폭 0.025–0.04), 가상·압축(22)만 broad + --amp
    --amp A         : 플리커 진폭(플리커 클립만; mix 에서는 가상·압축만)
    --motion S      : 모든 잡음 사본(실 34·가상 32)의 motion_px·rot_deg 배율
    --drift S       : illum_drift 배율
  make_noflk_pool.py / make_fr_pool.py 와 같은 구조(같은 시드 → 바뀌지 않은 항은 동일; 플리커·배율 항은 난수를 소비하지 않음).
  배율 1·플리커 keep 인 클립은 심볼릭 링크로 재사용. 매핑 검증(배율 1 일 때 움직임 전용 변형 재생성본 == 원본). 마커 /tmp/pool_<NAME>.done"""
import argparse, glob, json, os, shutil, subprocess, sys
import numpy as np, cv2
ROOT = "/SPECPHYS_ROOT"; PFA = f"{ROOT}/_workspace/data/physformer"
sys.path.insert(0, f"{ROOT}/scripts")
from make_noisy_real import corrupt  # noqa
from make_noisy_real_train import VARIANTS  # noqa
PY = sys.executable; BUILD = f"{ROOT}/generator/build_synth_dataset.py"
HZ = [0.8, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5]

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--flicker", default="keep", choices=["keep", "broad", "mix"])
ap.add_argument("--amp", default="keep"); ap.add_argument("--motion", type=float, default=1.0); ap.add_argument("--drift", type=float, default=1.0)
A = ap.parse_args(); NAME = A.name; AMP = None if A.amp == "keep" else float(A.amp)


def revise(nz, k, virtual):
    nz = dict(nz); changed = False
    if nz.get("flicker_amp", 0) > 0:
        if A.flicker == "broad" or (A.flicker == "mix" and virtual): nz["flicker_hz"] = HZ[k % len(HZ)]; changed = True
        if AMP is not None and (A.flicker != "mix" or virtual): nz["flicker_amp"] = AMP; changed = True
    if A.motion != 1.0:
        for key in ("motion_px", "rot_deg"):
            if nz.get(key, 0): nz[key] = nz[key] * A.motion; changed = True
    if A.drift != 1.0 and nz.get("illum_drift", 0): nz["illum_drift"] = nz["illum_drift"] * A.drift; changed = True
    return nz, changed


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
OUT = f"{PFA}/COTRAIN_NOISY_REAL_{NAME}"; os.makedirs(OUT, exist_ok=True)
subs = [s for s in sorted(glob.glob(f"{PFA}/COTRAIN/subject*")) if int(os.path.basename(s).replace("subject", "")) < 200]; assert len(subs) == 34
checked = A.motion != 1.0 or A.drift != 1.0   # 배율이 있으면 검증 불가(모든 사본이 바뀜)
for i, s in enumerate(subs):
    name = os.path.basename(s); d = f"{OUT}/{name}"; nz, changed = revise(VARIANTS[i % len(VARIANTS)], i, virtual=False)
    if os.path.exists(f"{d}/vid.avi") or (os.path.islink(d) and os.path.exists(d)): continue
    if not changed and checked: link(f"{PFA}/COTRAIN_NOISY_REAL/{name}", d); continue
    os.makedirs(d, exist_ok=True); n = corrupt(os.path.realpath(f"{s}/vid.avi"), f"{d}/vid.avi", nz, seed=800 + i); shutil.copy(os.path.realpath(f"{s}/ground_truth.txt"), f"{d}/ground_truth.txt")
    if not changed:
        md = frames_equal(f"{d}/vid.avi", f"{PFA}/COTRAIN_NOISY_REAL/{name}/vid.avi"); print(f"[real] mapping check {name}: mean|diff|={md:.4f}", flush=True); assert md < 0.5; checked = True
    log.append(("real", name, nz)); print(f"[real] {name}: {n} frames, variant {i % 5}, nz={nz}", flush=True)
# 2. 가상 세트
for cfg, src, dst in [("train20h_tw.json", "TRAIN20H_TW", f"TRAIN20H_TW{NAME}"), ("train6hi_tw.json", "TRAIN6HI_TW", f"TRAIN6HI_TW{NAME}"), ("train6lo_tw.json", "TRAIN6LO_TW", f"TRAIN6LO_TW{NAME}")]:
    c = json.load(open(f"{PFA}/configs/{cfg}")); os.makedirs(f"{PFA}/{dst}", exist_ok=True); todo = []
    for sub in c["subjects"]:
        nz, changed = revise(sub.get("noise") or {}, int(sub["id"]) + 100, virtual=True)
        if changed:
            if os.path.exists(f"{PFA}/{dst}/subject{sub['id']}/vid.avi"): continue
            sub = json.loads(json.dumps(sub)); sub["noise"] = nz; todo.append(sub); log.append((dst, sub["id"], nz))
        else: link(f"{PFA}/{src}/subject{sub['id']}", f"{PFA}/{dst}/subject{sub['id']}")
    if todo:
        c2 = dict(c); c2["subjects"] = todo; c2["out"] = f"_workspace/data/physformer/{dst}"; tmp = f"{PFA}/configs/{cfg.replace('.json', NAME.lower() + '.json')}"; json.dump(c2, open(tmp, "w"), indent=1)
        print(f"[syn] build {dst}: {len(todo)} clips", flush=True); subprocess.run([PY, BUILD, "--config", tmp], cwd=ROOT, check=True)
# 3. 압축 사본 (TRAIN20H 홀수 10 → 601..610, mpeg4 q6)
D = f"{PFA}/CMP_SYN_TW{NAME}"; os.makedirs(D, exist_ok=True)
for i, k in enumerate([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 1):
    d = f"{D}/subject{600 + i}"; srcdir = f"{PFA}/TRAIN20H_TW{NAME}/subject{k}"
    if os.path.islink(srcdir): link(f"{PFA}/CMP_SYN_TW/subject{600 + i}", d); continue
    if os.path.exists(f"{d}/vid.avi"): continue
    os.makedirs(d, exist_ok=True); subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{srcdir}/vid.avi", "-c:v", "mpeg4", "-q:v", "6", f"{d}/vid.avi"], check=True); shutil.copy(f"{srcdir}/ground_truth.txt", f"{d}/ground_truth.txt")
# 4. 풀
P = f"{PFA}/COTRAIN_ALLF_TW{NAME}"; shutil.rmtree(P, ignore_errors=True); os.makedirs(P)
rep = {"COTRAIN_NOISY_REAL": f"COTRAIN_NOISY_REAL_{NAME}", "TRAIN20H_TW": f"TRAIN20H_TW{NAME}", "TRAIN6HI_TW": f"TRAIN6HI_TW{NAME}", "TRAIN6LO_TW": f"TRAIN6LO_TW{NAME}", "CMP_SYN_TW": f"CMP_SYN_TW{NAME}"}; nrep = 0
for s in sorted(os.listdir(f"{PFA}/COTRAIN_ALLF_TW")):
    t = os.path.realpath(f"{PFA}/COTRAIN_ALLF_TW/{s}"); dn = os.path.basename(os.path.dirname(t)); tgt = t
    if dn in rep: tgt = f"{PFA}/{rep[dn]}/{os.path.basename(t)}"; nrep += 1
    assert os.path.exists(f"{tgt}/vid.avi"), tgt; os.symlink(os.path.realpath(tgt), f"{P}/{s}")
json.dump({"args": vars(A), "clips": log}, open(f"{PFA}/COTRAIN_ALLF_TW{NAME}_log.json", "w"), indent=0, default=str)
print(f"POOL ALLF_TW{NAME} n={len(os.listdir(P))} replaced={nrep} regenerated={len(log)} args={vars(A)}", flush=True)
open(f"/tmp/pool_{NAME}.done", "w").write("ok\n"); print(f"[{NAME}] done")
