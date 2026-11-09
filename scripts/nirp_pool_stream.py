#!/usr/bin/env python3
"""UBFC 레이아웃 풀 → uint8 캐시 스트리밍 전처리 (툴박스 BaseLoader u8 모드와 동일 산출, 상주 메모리 < 1 GB/프로세스).
툴박스 UBFCrPPGLoader 는 영상 전체를 float 로 들고 있어 7~12k 프레임 영상에서 워커당 10+ GB → 10-14 MR-NIRP 풀 전처리 사고. 본 스크립트는
프레임을 순차 복호해 Haar 정적 박스(첫 프레임, LARGE_BOX_COEF 1.5) 크롭 → 128×128 INTER_AREA → uint8 보관, diff 표준편차는 누적합으로 계산.
사용: python scripts/nirp_pool_stream.py POOL_DIR CACHE_DIR [--workers N] [--only subjectA,subjectB]
검증: scripts/nirp_pool_stream.py 는 기존 캐시 피험자(subject30141)에서 입력 청크 bit-동일·라벨 동일·scale 상대오차 <1e-5 확인 후 사용(§5 로그)."""
import os, sys, glob, time, multiprocessing as mp
import numpy as np, cv2
HAAR = "/SPECPHYS_ROOT/repos/rPPG-Toolbox/dataset/haarcascade_frontalface_default.xml"
L, W, H, COEF = 160, 128, 128, 1.5


def face_box(frame):
    det = cv2.CascadeClassifier(HAAR); z = det.detectMultiScale(frame[:, :, :3].astype(np.uint8))
    if len(z) < 1: print("ERROR: No Face Detected", flush=True); box = [0, 0, frame.shape[0], frame.shape[1]]
    elif len(z) >= 2: box = list(z[np.argmax(z[:, 2])])
    else: box = list(z[0])
    box = [float(b) for b in box]
    box[0] = max(0, box[0] - (COEF - 1.0) / 2 * box[2]); box[1] = max(0, box[1] - (COEF - 1.0) / 2 * box[3])
    box[2] = COEF * box[2]; box[3] = COEF * box[3]
    return np.asarray(box, dtype="int")


def diff_normalize_label(label):
    d = np.diff(label, axis=0); d = d / np.std(d); d = np.append(d, np.zeros(1), axis=0); d[np.isnan(d)] = 0
    return d


def process(args):
    sdir, cache = args; name = os.path.basename(sdir.rstrip("/")); t0 = time.time()
    cap = cv2.VideoCapture(os.path.join(sdir, "vid.avi")); cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    ok, fr = cap.read(); assert ok, name
    frames = []; box = None; prev = None; s1 = 0.0; s2 = 0.0; cnt = 0
    while ok:
        fr = cv2.cvtColor(np.array(fr), cv2.COLOR_BGR2RGB)
        if box is None: box = face_box(fr)
        c = fr[max(box[1], 0):min(box[1] + box[3], fr.shape[0]), max(box[0], 0):min(box[0] + box[2], fr.shape[1])]
        r = cv2.resize(c, (W, H), interpolation=cv2.INTER_AREA)  # uint8 (툴박스는 float64 배열에 넣은 뒤 rint/clip → 동일)
        frames.append(r); f32 = r.astype(np.float32)
        if prev is not None:
            dd = (f32 - prev) / (f32 + prev + 1e-7); s1 += float(dd.sum(dtype=np.float64)); s2 += float((dd.astype(np.float64) ** 2).sum()); cnt += dd.size
        prev = f32; ok, fr = cap.read()
    cap.release(); frames = np.stack(frames); n = len(frames)
    mean = s1 / cnt; sc = float(np.sqrt(max(s2 / cnt - mean * mean, 0.0)))
    bvp = np.asarray([float(x) for x in open(os.path.join(sdir, "ground_truth.txt")).read().split("\n")[0].split()])
    assert len(bvp) == n, (name, len(bvp), n)
    lab = diff_normalize_label(bvp); k = n // L
    os.makedirs(cache, exist_ok=True)
    for i in range(k):
        np.save(f"{cache}/{name}_input{i}.npy", frames[i * L:min(i * L + L + 1, n)]); np.save(f"{cache}/{name}_label{i}.npy", lab[i * L:(i + 1) * L])
    np.save(f"{cache}/{name}_u8scale.npy", np.float32(sc))
    return f"ok {name} n={n} chunks={k} scale={sc:.6f} box={box.tolist()} sec={time.time()-t0:.0f}"


if __name__ == "__main__":
    pool, cache = sys.argv[1], sys.argv[2]; a = sys.argv[3:]
    nw = int(a[a.index("--workers") + 1]) if "--workers" in a else 2
    only = a[a.index("--only") + 1].split(",") if "--only" in a else None
    subs = sorted(d for d in glob.glob(f"{pool}/subject*") if os.path.exists(f"{d}/ground_truth.txt") and (only is None or os.path.basename(d) in only))
    print(f"STREAM_PREPROCESS start n={len(subs)} workers={nw} cache={cache}", flush=True)
    with mp.get_context("spawn").Pool(nw) as P:
        for r in P.imap_unordered(process, [(s, cache) for s in subs]): print(r, flush=True)
    print("STREAM_PREPROCESS_DONE", flush=True)
