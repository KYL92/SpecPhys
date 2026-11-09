#!/usr/bin/env python3
"""[체인 65, 48 문서] 플리커 제거 대조 풀 COTRAIN_ALLF_TWNF — 공식 코퍼스 COTRAIN_ALLF_TW(203) 과 클립 단위로 일치하되
플리커 항(flicker_amp)만 0 으로 다시 만든 사본으로 교체한다(같은 시드·같은 드리프트/움직임/센서 잡음: 두 생성기 모두 플리커가 난수를 소비하지 않음).
  실 잡음 사본 COTRAIN_NOISY_REAL(34, 변형 5종 순환 중 4종 플리커) → COTRAIN_NOISY_REAL_NF
  가상 TRAIN20H_TW(12/20)·TRAIN6HI_TW(2/6)·TRAIN6LO_TW(2/6) → *_TWNF (플리커 없는 클립은 심볼릭 링크)
  압축 사본 CMP_SYN_TW(TRAIN20H_TW 홀수 10, mpeg4 q6) → CMP_SYN_TWNF
검증: 움직임 전용 변형(i%5==1)을 재생성해 원본과 프레임 동일한지 확인(매핑 검증). 마커 /tmp/noflk_pool.done"""
import glob, json, os, shutil, subprocess, sys
import numpy as np, cv2
ROOT = "/SPECPHYS_ROOT"; PFA = f"{ROOT}/_workspace/data/physformer"
sys.path.insert(0, f"{ROOT}/scripts")
from make_noisy_real import corrupt  # noqa
from make_noisy_real_train import VARIANTS  # noqa
PY = sys.executable; BUILD = f"{ROOT}/generator/build_synth_dataset.py"


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


# 1. 실 잡음 사본
OUT = f"{PFA}/COTRAIN_NOISY_REAL_NF"; os.makedirs(OUT, exist_ok=True)
subs = [s for s in sorted(glob.glob(f"{PFA}/COTRAIN/subject*")) if int(os.path.basename(s).replace("subject", "")) < 200]
assert len(subs) == 34, len(subs)
checked = False
for i, s in enumerate(subs):
    name = os.path.basename(s); d = f"{OUT}/{name}"; nz = dict(VARIANTS[i % len(VARIANTS)])
    has_fl = nz.get("flicker_amp", 0) > 0
    for k in ("flicker_amp", "flicker_hz"): nz.pop(k, None)
    if os.path.exists(f"{d}/vid.avi") or (os.path.islink(d) and os.path.exists(d)): print(f"[real] skip {name}"); continue
    if not has_fl and checked:  # 움직임 전용 변형은 원본과 동일 → 링크
        link(f"{PFA}/COTRAIN_NOISY_REAL/{name}", d); print(f"[real] link {name} (variant {i % 5})"); continue
    os.makedirs(d, exist_ok=True)
    n = corrupt(os.path.realpath(f"{s}/vid.avi"), f"{d}/vid.avi", nz, seed=800 + i)
    shutil.copy(os.path.realpath(f"{s}/ground_truth.txt"), f"{d}/ground_truth.txt")
    if not has_fl:  # 매핑 검증: 재생성 == 기존 사본
        md = frames_equal(f"{d}/vid.avi", f"{PFA}/COTRAIN_NOISY_REAL/{name}/vid.avi")
        print(f"[real] mapping check {name}: mean|diff|={md:.4f}"); assert md < 0.5, "variant mapping mismatch"; checked = True
    print(f"[real] {name}: {n} frames, variant {i % 5}, flicker removed={has_fl}", flush=True)

# 2. 가상 세트
for cfg, src, dst in [("train20h_tw.json", "TRAIN20H_TW", "TRAIN20H_TWNF"), ("train6hi_tw.json", "TRAIN6HI_TW", "TRAIN6HI_TWNF"), ("train6lo_tw.json", "TRAIN6LO_TW", "TRAIN6LO_TWNF")]:
    c = json.load(open(f"{PFA}/configs/{cfg}")); os.makedirs(f"{PFA}/{dst}", exist_ok=True)
    fl = []
    for sub in c["subjects"]:
        nz = sub.get("noise") or {}
        if nz.get("flicker_amp", 0) > 0:
            if os.path.exists(f"{PFA}/{dst}/subject{sub['id']}/vid.avi"): continue
            sub = json.loads(json.dumps(sub)); sub["noise"].pop("flicker_amp"); sub["noise"].pop("flicker_hz", None); fl.append(sub)
        else:
            link(f"{PFA}/{src}/subject{sub['id']}", f"{PFA}/{dst}/subject{sub['id']}")
    if fl:
        c2 = dict(c); c2["subjects"] = fl; c2["out"] = f"_workspace/data/physformer/{dst}"
        tmp = f"{PFA}/configs/{cfg.replace('.json', 'nf.json')}"; json.dump(c2, open(tmp, "w"), indent=1)
        print(f"[syn] build {dst}: {len(fl)} flicker clips", flush=True)
        subprocess.run([PY, BUILD, "--config", tmp], cwd=ROOT, check=True)

# 3. 압축 사본 (mpeg4 q6, TRAIN20H_TW 홀수 → 601..610)
D = f"{PFA}/CMP_SYN_TWNF"; os.makedirs(D, exist_ok=True)
for i, k in enumerate([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 1):
    d = f"{D}/subject{600 + i}"; srcdir = f"{PFA}/TRAIN20H_TWNF/subject{k}"
    if os.path.islink(srcdir): link(f"{PFA}/CMP_SYN_TW/subject{600 + i}", d); continue
    if os.path.exists(f"{d}/vid.avi"): continue
    os.makedirs(d, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{srcdir}/vid.avi", "-c:v", "mpeg4", "-q:v", "6", f"{d}/vid.avi"], check=True)
    shutil.copy(f"{srcdir}/ground_truth.txt", f"{d}/ground_truth.txt"); print(f"[cmp] subject{600 + i} from TRAIN20H_TWNF/subject{k}", flush=True)

# 4. 풀
P = f"{PFA}/COTRAIN_ALLF_TWNF"; shutil.rmtree(P, ignore_errors=True); os.makedirs(P)
rep = {"COTRAIN_NOISY_REAL": "COTRAIN_NOISY_REAL_NF", "TRAIN20H_TW": "TRAIN20H_TWNF", "TRAIN6HI_TW": "TRAIN6HI_TWNF", "TRAIN6LO_TW": "TRAIN6LO_TWNF", "CMP_SYN_TW": "CMP_SYN_TWNF"}
nrep = 0
for s in sorted(os.listdir(f"{PFA}/COTRAIN_ALLF_TW")):
    t = os.path.realpath(f"{PFA}/COTRAIN_ALLF_TW/{s}"); d = os.path.basename(os.path.dirname(t)); tgt = t
    if d in rep: tgt = f"{PFA}/{rep[d]}/{os.path.basename(t)}"; nrep += 1
    assert os.path.exists(f"{tgt}/vid.avi"), tgt
    os.symlink(os.path.realpath(tgt), f"{P}/{s}")
print(f"POOL ALLF_TWNF n={len(os.listdir(P))} replaced={nrep}")
open("/tmp/noflk_pool.done", "w").write("ok\n"); print("[noflk] done")
