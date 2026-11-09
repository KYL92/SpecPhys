#!/usr/bin/env python3
"""[63 문서 §2] 합성 데이터 그림 재료: 풀 정체성 5 + 시험 전용 정체성 4 의 실제 클립 프레임, PPG 뱅크 파형, 이중 렌더 클립의 혼합·심박·잡음 스펙트럼(학습 없음).
  python scripts/paper_arch_syn_data.py → results/figs_arch/{tile_*.png, syn_data.json}"""
import json, os, sys
import cv2, numpy as np
ROOT = "/SPECPHYS_ROOT"; PFA = f"{ROOT}/_workspace/data/physformer"; TB = f"{ROOT}/repos/rPPG-Toolbox"; OUT = f"{ROOT}/_workspace/data/results/figs_arch"
fc = cv2.CascadeClassifier(f"{TB}/dataset/haarcascade_frontalface_default.xml")
def frame(path, k):
    v = cv2.VideoCapture(path); v.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = v.read(); assert ok, path; return f
def box_of(path):  # 그림용 가운데 자르기(옆모습은 Haar 가 놓친다, 합성 얼굴의 늘어난 가장자리도 가린다)
    f = frame(path, 0); h, w = f.shape[:2]; m = int(0.11 * w); return m, m, w - 2 * m, h - 2 * m
J = {"pool_tiles": [], "test_tiles": []}
# 풀 5 정체성: 실제 합성 클립 두 개(이중 렌더 1 + 20클립 집합 1). 프레임은 시점 200 이후 Haar 로 얼굴이 잡히는 첫 프레임(일부 클립은 큰 궤적으로 얼굴이 화면 밖으로 나간다, 63 문서 §2).
cfg = {s["id"]: s for n in ("train24dn", "train24dn2") for s in json.load(open(f"{PFA}/configs/{n}.json"))["subjects"]}; cfg20 = {s["id"]: s for s in json.load(open(f"{PFA}/configs/train20h_tw.json"))["subjects"]}
def first_face(path, start):
    v = cv2.VideoCapture(path); n = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    for k in range(start, n, 10):
        v.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = v.read()
        if ok and len(fc.detectMultiScale(f, minSize=(60, 60), minNeighbors=5)) > 0: return f, k
    raise RuntimeError(path)
faces = {}; f20 = {}
for i, s in sorted(cfg.items()): faces.setdefault(s["frames_dir"].split("/")[-1], []).append(i)
for i, s in sorted(cfg20.items()): f20.setdefault(s["frames_dir"].split("/")[-1], []).append(i)
for fd in ("sfhq_01_frames", "sfhq_02_frames", "sfhq_03_frames", "sfhq_05_frames", "sfhq_06_frames"):
    for r, (ds, sid, t0) in enumerate((("TRAIN20H_TW", f20[fd][0], 200), ("TRAIN24DN" if faces[fd][-1] <= 284 else "TRAIN24DN2", faces[fd][-1], 640))):
        p = f"{PFA}/{ds}/subject{sid}/vid.avi"; x, y, w, h = box_of(p); fr_, k = first_face(p, t0); name = f"tile_{fd[:7]}_{r}.png"; cv2.imwrite(f"{OUT}/{name}", cv2.resize(fr_[y:y + h, x:x + w], (200, 200), interpolation=cv2.INTER_AREA)); J["pool_tiles"].append(dict(file=name, face=fd, set=ds, id=sid, frame=k))
# (e) 패널 클립: 머리 궤적 없는 주기 움직임 클립(심박 60, 잡음 96 bpm)
p276 = f"{PFA}/TRAIN24DN/subject276/vid.avi"; x, y, w, h = box_of(p276); fr_, k = first_face(p276, 200); cv2.imwrite(f"{OUT}/tile_e.png", cv2.resize(fr_[y:y + h, x:x + w], (200, 200), interpolation=cv2.INTER_AREA))
# 시험 전용 4 정체성(풀에 없음)
for sid, fd in ((1, "sfhq04"), (2, "diff_female_01"), (3, "diff_male_01"), (4, "diff_female_02")):
    p = f"{PFA}/STRESS_OFFBAND6/subject{sid}/vid.avi"; x, y, w, h = box_of(p); f = frame(p, 200)[y:y + h, x:x + w]; name = f"tile_test_{sid}.png"; cv2.imwrite(f"{OUT}/{name}", cv2.resize(f, (200, 200), interpolation=cv2.INTER_AREA)); J["test_tiles"].append(dict(file=name, face=fd))
# PPG 뱅크: 기본 주파수 순서로 고르게 12개(고른 규칙: 순위 등간격)
bk = np.load(f"{PFA}/ppg_bank.npz", allow_pickle=True); seg, f0, src = bk["segments"], bk["f0"], bk["src"]; order = np.argsort(f0); pick = order[np.linspace(0, len(order) - 1, 12).round().astype(int)]
J["bank"] = dict(n=int(len(seg)), src_counts={"PURE": int(sum(str(k).startswith("9") for k in src)), "UBFC": int(sum(str(k).startswith("ubfc") for k in src))}, f0_bpm=[float(f0.min() * 60), float(np.median(f0) * 60), float(f0.max() * 60)], tiles=[dict(bpm=float(f0[i] * 60), src="UBFC-rPPG" if str(src[i]).startswith("ubfc") else "PURE", wave=seg[i][:150].tolist()) for i in pick])
print("bank", J["bank"]["n"], J["bank"]["src_counts"], J["bank"]["f0_bpm"], [round(t["bpm"]) for t in J["bank"]["tiles"]], [t["src"] for t in J["bank"]["tiles"]])
# 이중 렌더 클립: 혼합(영상 피부 중앙 G), 심박(PPG 라벨), 잡음(n(t)) 스펙트럼
def spec(a, fs=30.0, lo=30, hi=200):
    a = np.asarray(a, float); a = a - a.mean(); N = len(a) * 2; A = np.abs(np.fft.rfft(a * np.hanning(len(a)), n=N)) ** 2; f = np.fft.rfftfreq(N, 1 / fs) * 60; s = (f >= lo) & (f <= hi); return f[s], A[s]
def hp(x, k=31): return x - np.convolve(x, np.ones(k) / k, mode="same")
J["clips"] = {}
for sid, s in sorted(cfg.items()):
    ds = "TRAIN24DN" if sid <= 284 else "TRAIN24DN2"; d = f"{PFA}/{ds}/subject{sid}"; L = open(f"{d}/ground_truth.txt").read().strip().split("\n"); ppg = np.array(L[0].split(), float); gt = float(np.median(np.array(L[1].split(), float)))
    cap = cv2.VideoCapture(f"{d}/vid.avi"); G = []
    while True:
        ok, f = cap.read()
        if not ok: break
        h, w = f.shape[:2]; G.append(f[h // 4:3 * h // 4, w // 4:3 * w // 4, 1].mean())
    G = np.array(G); nt = np.load(f"{PFA}/noise_gt_hyb2/subject{sid}.npy"); T = min(len(G), len(ppg), len(nt)); dG = np.diff(G[:T]); ff, Sm = spec(hp(dG)); _, Sh = spec(ppg[:T]); _, Sn = spec(hp(nt[:T]))
    pk = lambda a: float(ff[np.argmax(a)]); n_ = s["noise"]; J["clips"][str(sid)] = dict(face=s["frames_dir"].split("/")[-1], bpm=s["bpm"], gt=gt, flicker=n_.get("flicker_hz") if n_.get("flicker_amp", 0) else None, motion=n_.get("periodic_motion_hz"), peak_mix=pk(Sm), peak_hr=pk(Sh), peak_noise=pk(Sn), f=ff.tolist(), mix=Sm.tolist(), hr=Sh.tolist(), noise=Sn.tolist(), ppg=ppg[100:260].tolist(), n=nt[100:260].tolist())
    print(sid, J["clips"][str(sid)]["face"][:7], "bpm", s["bpm"], "flicker", J["clips"][str(sid)]["flicker"], "motion", J["clips"][str(sid)]["motion"], "| peaks mix %.0f hr %.0f noise %.0f" % (pk(Sm), pk(Sh), pk(Sn)))
# (e) 패널 라벨 표: 이중 렌더 클립 5개(정체성 5개 모두, 심박 54~160 bpm, 머리 화면 이탈 클립 제외, 심박과 잡음이 11 bpm 이상 떨어진 클립), 학습과 같은 청크 단위(160프레임, 두 번째 청크)와 같은 스펙트럼 계산(NFLoss.power: 1 bpm 격자)
def power(x, fs=30.0, lo=30, hi=200):
    x = np.asarray(x, float); t = np.arange(len(x)) / fs; bpm = np.arange(lo, hi, 1.0); ang = 2 * np.pi * (bpm / 60.0)[:, None] * t[None, :]; xz = x - x.mean(); P = (np.cos(ang) @ xz) ** 2 + (np.sin(ang) @ xz) ** 2; return bpm, P / P.sum()
def label_hr(x, fs=30.0, lo=0.5, hi=3.3):
    x = np.asarray(x, float); x = x - x.mean(); nfft = 8 * len(x); P = np.abs(np.fft.rfft(x, n=nfft)) ** 2; f = np.fft.rfftfreq(nfft, 1 / fs); P[(f < lo) | (f > hi)] = 0; return float(f[np.argmax(P)] * 60)
J["label_rows"] = []; C0, L0 = 1, 160
for sid in (299, 276, 288, 302, 296):
    s_ = cfg[sid]; ds = "TRAIN24DN" if sid <= 284 else "TRAIN24DN2"; d = f"{PFA}/{ds}/subject{sid}"; ppg = np.array(open(f"{d}/ground_truth.txt").read().strip().split("\n")[0].split(), float); nt = np.load(f"{PFA}/noise_gt_hyb2/subject{sid}.npy"); a0, a1 = C0 * L0, (C0 + 1) * L0
    cap = cv2.VideoCapture(f"{d}/vid.avi"); cap.set(cv2.CAP_PROP_POS_FRAMES, a0); G = []
    for _ in range(L0 + 1):
        ok, f = cap.read(); h, w = f.shape[:2]; G.append(f[h // 4:3 * h // 4, w // 4:3 * w // 4, 1].mean())
    bpm, Pm = power(np.diff(np.array(G))); _, Ph = power(ppg[a0:a1]); _, Pn = power(nt[a0:a1]); hrl = label_hr(ppg[a0:a1]); n_ = s_["noise"]; kind = "flicker" if n_.get("flicker_amp", 0) else "motion"; nb = float(n_.get("flicker_hz") if kind == "flicker" else n_.get("periodic_motion_hz")) * 60
    cap = cv2.VideoCapture(f"{d}/vid.avi"); cap.set(cv2.CAP_PROP_POS_FRAMES, a0 + 80); ok, f = cap.read(); x, y, w, h = box_of(f"{d}/vid.avi"); name = f"tile_row_{sid}.png"; cv2.imwrite(f"{OUT}/{name}", cv2.resize(f[y:y + h, x:x + w], (200, 200), interpolation=cv2.INTER_AREA))
    J["label_rows"].append(dict(id=sid, file=name, face=s_["frames_dir"].split("/")[-1], bpm=float(s_["bpm"]), hr_label=hrl, noise_defined=nb, kind=kind, peak_hr=float(bpm[np.argmax(Ph)]), peak_noise=float(bpm[np.argmax(Pn)]), peak_mix=float(bpm[np.argmax(Pm)]), f=bpm.tolist(), mix=Pm.tolist(), hr=Ph.tolist(), noise=Pn.tolist(), ppg=ppg[a0:a1].tolist(), band_idx=int(s_.get("wave_bank_idx", -1))))
    print("row", sid, s_["frames_dir"].split("/")[-1][:7], "defined HR", s_["bpm"], "label_hr %.1f" % hrl, "noise", kind, nb, "| chunk peaks mix %.0f hr %.0f noise %.0f" % (J["label_rows"][-1]["peak_mix"], J["label_rows"][-1]["peak_hr"], J["label_rows"][-1]["peak_noise"]))
json.dump(J, open(f"{OUT}/syn_data.json", "w"))
