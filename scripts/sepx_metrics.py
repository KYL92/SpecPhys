#!/usr/bin/env python3
"""[65 문서] 풀 밖 합성 분리 시험(SEPEVAL: 새 STRESS_SEPX24 24클립 + 기존 스트레스 20클립) 모델별 지표. pro6000 컨테이너에서 실행(클립 meta 와 예측 pickle 이 거기 있다).
청크(160프레임)마다: r_ppg = 출력과 정답 PPG 의 Pearson, hr_ok = 출력 주기도(30~200 bpm, 8배 영 채움) 정점이 정답 심박 ±6 bpm 안, flick = 잡음 주파수 정현파(동상·직교)와의 결맞음 √(r_sin²+r_cos²).
클립 값 = 청크 평균. 출력: results/sepx_{MODEL}.json  사용: python scripts/sepx_metrics.py MODEL [MODEL...]"""
import glob, json, os, pickle, sys
import numpy as np
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; RES = "/SPECPHYS_ROOT/_workspace/data/results"
CLEAN = os.environ.get("SEPX_SET", "") == "C"   # [66 §7] 깨끗한 배경 판(폴더 끝 _CLEAN, 평가 폴더 ev_{M}_SEPEVALC, 출력 sepxc_{M}.json)
SETS = {1000: ("STRESS_SEPX24", "stress_sepx24"), 1100: ("STRESS_OFFBAND6", "stress_offband6"), 1200: ("STRESS_PMOT6", "stress_pmot6"), 1300: ("STRESS_INBAND6", "stress_inband6"), 1400: ("STRESS_NEAR6", "stress_near6"), 1500: ("TEST_C3", "test_C3")}
META = {}
for base, (dname, cname) in SETS.items():
    for sb in json.load(open(f"{PFA}/configs/{cname}.json"))["subjects"]:
        sid = base + int(sb["id"]); m = json.load(open(f"{PFA}/{dname}{'_CLEAN' if CLEAN else ''}/subject{sb['id']}/meta.json")); nz = sb.get("noise") or {}
        fn = nz.get("flicker_hz") if nz.get("flicker_amp", 0) > 0 else (nz.get("local_flicker") or {}).get("hz") or nz.get("periodic_motion_hz")
        META[f"subject{sid}"] = dict(set=dname, id=int(sb["id"]), face=sb["frames_dir"].split("/")[-1], hr=float(m["achieved_bpm"]), noise_hz=float(fn) if fn else None,
                                     kind=sb.get("sepx_type") or ("G" if nz.get("flicker_amp", 0) > 0 else "L" if nz.get("local_flicker") else "M" if nz.get("periodic_motion_amp") else "?"),
                                     gap=(float(fn) * 60 - float(m["achieved_bpm"])) if fn else None)
def z(a): a = np.asarray(a, np.float64); return (a - a.mean()) / (a.std() + 1e-12)
def peak(a, fs=30.0, lo=30, hi=200, pad=8):
    a = z(a); N = len(a) * pad; P = np.abs(np.fft.rfft(a, n=N)) ** 2; f = np.fft.rfftfreq(N, 1 / fs) * 60; s = (f >= lo) & (f <= hi); return float(f[s][np.argmax(P[s])])
for M in sys.argv[1:]:
    pk = glob.glob(f"{PFA}/runs/ev_{M}_SEPEVAL{'C' if CLEAN else ''}/*/saved_test_outputs/*.pickle")
    if not pk: print("no pickle", M); continue
    d = pickle.load(open(pk[0], "rb")); out = {}
    for s, ch in d["predictions"].items():
        mt = META.get(s)
        if mt is None: continue
        rr, ok, fl = [], [], []
        for c in sorted(ch):
            p = ch[c].cpu().numpy().ravel() if hasattr(ch[c], "cpu") else np.asarray(ch[c]).ravel(); y = d["labels"][s][c]; y = y.cpu().numpy().ravel() if hasattr(y, "cpu") else np.asarray(y).ravel()
            rr.append(float(np.mean(z(p) * z(y)))); ok.append(float(abs(peak(p) - mt["hr"]) <= 6.0))
            if mt["noise_hz"]:
                t = (int(c) * len(p) + np.arange(len(p))) / 30.0; zs, zc = z(np.sin(2 * np.pi * mt["noise_hz"] * t)), z(np.cos(2 * np.pi * mt["noise_hz"] * t)); zp = z(p)
                fl.append(float(np.hypot(np.mean(zp * zs), np.mean(zp * zc))))
        out[s] = dict(mt, chunks=len(rr), r_ppg=float(np.mean(rr)), hr_ok=float(np.mean(ok)), flick=float(np.mean(fl)) if fl else None)
    json.dump(dict(model=M, clips=out), open(f"{RES}/sepx{'c' if CLEAN else ''}_{M}.json", "w"), indent=1); print("sepx", "clean" if CLEAN else "", M, len(out), "clips")
