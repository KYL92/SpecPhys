#!/usr/bin/env python3
"""[65 문서, 체인 124] 이중 렌더 잡음 영상(vid_noise.avi, 둘째 벌)을 학습 캐시의 입력 청크와 같은 크롭·크기·청크로 저장: {cache}/{name}_noisevid{i}.npy (u8, 161×128×128×3).
얼굴 상자는 vid.avi 첫 프레임에서 툴박스와 같은 Haar 검출(큰 상자 1.5)로 한 번 구하고 두 영상에 툴박스 crop_face_resize 로 똑같이 적용한다.
vid.avi 로 만든 청크가 캐시의 _input 청크와 비트 단위로 같을 때만 저장한다(재렌더 결정성 + 정렬 검증). 불일치 클립은 MISMATCH 로 알리고 건너뛴다.
  python scripts/noisevid_sidecar.py SRC_DIR CACHE_DIR"""
import glob, os, sys
import numpy as np
ARGS = [os.path.abspath(x) for x in sys.argv[1:3]]   # 상대 경로는 chdir 전에 절대 경로로
TB = "/SPECPHYS_ROOT/repos/rPPG-Toolbox"; sys.path.insert(0, TB); os.chdir(TB)
from dataset.data_loader.BaseLoader import BaseLoader
from dataset.data_loader.UBFCrPPGLoader import UBFCrPPGLoader
src, cache = ARGS[0].rstrip("/"), ARGS[1].rstrip("/"); L = 160


def crop(frames, box):
    o = object.__new__(UBFCrPPGLoader); o.face_detection = lambda *a, **k: box   # 정적 검출: 첫 프레임 상자 하나를 모든 프레임에
    return np.clip(np.rint(o.crop_face_resize(frames, True, "HC", True, 1.5, False, 30, False, 128, 128)), 0, 255).astype(np.uint8)


ok = bad = 0
for nv in sorted(glob.glob(f"{src}/subject*/vid_noise.avi")):
    d = os.path.dirname(nv); name = os.path.basename(d)
    v = UBFCrPPGLoader.read_video(f"{d}/vid.avi"); n_ = UBFCrPPGLoader.read_video(nv)
    assert v.shape == n_.shape, (name, v.shape, n_.shape)
    box = BaseLoader.face_detection(None, v[0], "HC", True, 1.5)
    cv, cn = crop(v, box), crop(n_, box); nc = cv.shape[0] // L
    good = nc > 0 and all(os.path.exists(f"{cache}/{name}_input{i}.npy") and np.array_equal(np.load(f"{cache}/{name}_input{i}.npy"), cv[i * L:min(i * L + L + 1, len(cv))]) for i in range(nc))
    if not good:
        bad += 1; print("MISMATCH", name, flush=True); continue
    for i in range(nc): np.save(f"{cache}/{name}_noisevid{i}.npy", cn[i * L:min(i * L + L + 1, len(cn))])
    dpx = float(np.abs(cv.astype(np.int16) - cn.astype(np.int16)).mean())
    ok += 1; print(f"ok {name} chunks={nc} box={list(map(int, box))} mean|input-noise|={dpx:.3f} gray", flush=True)
print(f"NOISEVID_DONE {src} ok={ok} mismatch={bad}", flush=True)
