#!/usr/bin/env python3
"""20단계(체인 37): 학습된 열화 diffusion — 깨끗한 실 PPG 창을 조건으로 backbone 출력풍 열화 창을 생성,
가상얼굴 주입 코퍼스에 저SNR 클립을 공급한다 (설계 _workspace/22_degradation_diffusion_design.md).

방향이 핵심: 실패한 refinement(열화→깨끗, 추론 시 적용)와 달리 이건 순방향(깨끗→열화) '생성'이고,
산출물은 검증된 채널(픽셀 주입 + 깨끗한 GT 라벨)로 들어가 backbone이 열화→HR 복원을 학습 분포 안에서 배운다.

subcommands:
  pairs    — (clean GT 창 → backbone 출력 창) 쌍: TRAINPOOL 203 + MMPD 1~6(×4 oversample) → refiner_deg/pairs.npz
             (Xtr=cond=clean, Ytr=target=degraded — diffusion_refiner.cmd_train 재사용을 위한 배치)
  train    — V1 ε-U-Net CFG 재사용 (dr.RD 교체)
  check    — 검증 창에서 생성 열화의 사실성: spec-SNR 분포·|ΔHR| 분포를 실측과 대조 (DSYN 실패 지표로 사전 게이트)
  genclips — 12개 신규 클립용 파형 생성: bank v2 nearest→clean, 창별 diffusion 열화→overlap-add,
             mean spec-SNR ∈ [-9, 0] dB 거부 샘플링(이중 열화 방지) → degwaves/dg_<id>.npz {inject, gt}
"""
import argparse, glob, hashlib, importlib.util, json, os, pickle
import numpy as np
import scipy.signal as ss

ROOT = "/SPECPHYS_ROOT"
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
dr = _load("dr", f"{ROOT}/scripts/diffusion_refiner.py")
lp = _load("lp", f"{ROOT}/scripts/lowsnr_param.py")
PF, FS, L, STRIDE = dr.PF, dr.FS, dr.L, dr.STRIDE
RD4 = f"{PF}/refiner_deg"
DW = f"{PF}/degwaves"

def band_znorm(x):
    b, a = ss.butter(1, [0.6 / FS * 2, 3.3 / FS * 2], btype="bandpass")
    return ss.filtfilt(b, a, np.asarray(x, float) - np.mean(x))

# ---------- pairs ----------
def cmd_pairs(a):
    Xtr, Ytr, Xva, Yva = [], [], [], []
    def add(vid, pred, lab, w_mult=1):
        p = dr.pulse_space(pred); l = dr.pulse_space(lab)
        n = min(len(p), len(l))
        if n < L: return
        val = int(hashlib.md5(str(vid).encode()).hexdigest()[:8], 16) % 10 == 0
        for st in range(0, n - L + 1, STRIDE):
            cw = l[st:st + L]  # cond = clean(GT)
            if cw.std() < 1e-6 or not dr.fundamental_ok(cw): continue
            tw = dr.znorm(p[st:st + L]); cw = dr.znorm(cw)  # target = degraded(backbone 출력)
            for _ in range(1 if val else w_mult):
                (Xva if val else Xtr).append(cw.astype(np.float32))
                (Yva if val else Ytr).append(tw.astype(np.float32))
    pk = glob.glob(f"{PF}/runs/ev_CoTAF_TW_L_TRAINPOOL/*/saved_test_outputs/*.pickle")[0]
    d = pickle.load(open(pk, "rb"))
    for vid in d["predictions"]:
        add(vid, dr.cat_chunks(d["predictions"][vid]), dr.cat_chunks(d["labels"][vid]), 1)
    import re
    for k in range(1, 7):
        po = glob.glob(f"{PF}/runs/ev_CoTAF_TW_L_MMPD_B{k}/*/saved_test_outputs/*.pickle")
        if not po: continue
        d = pickle.load(open(po[0], "rb"))
        for vid in d["predictions"]:
            m = re.match(r"subject(\d+)_", str(vid))
            if not m or int(m.group(1)) >= 7: continue
            add(vid, dr.cat_chunks(d["predictions"][vid]), dr.cat_chunks(d["labels"][vid]), 4)
    os.makedirs(RD4, exist_ok=True)
    np.savez(f"{RD4}/pairs.npz", Xtr=np.array(Xtr), Ytr=np.array(Ytr), Xva=np.array(Xva), Yva=np.array(Yva))
    print(f"PAIRS_DEG train={len(Xtr)} (MMPD x4 oversample) val={len(Xva)}")

def cmd_train(a):
    dr.RD = RD4
    dr.cmd_train(a)

# ---------- check ----------
def _gen_windows(model, conds, w, seed0, device, bs=256):
    import torch
    outs = []
    for i in range(0, len(conds), bs):
        c = torch.from_numpy(conds[i:i + bs]).unsqueeze(1).to(device)
        r = dr.ddim_sample(model, c, w=w, steps=50, seed=seed0 + i, device=device)
        outs.append(r.squeeze(1).cpu().numpy())
    return np.concatenate(outs)

def cmd_check(a):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(f"{RD4}/meta.json")); w = float(meta["w"])
    model = dr.UNet1D().to(device)
    model.load_state_dict(torch.load(f"{RD4}/ckpt.pt", map_location=device)); model.eval()
    z = np.load(f"{RD4}/pairs.npz")
    Xva, Yva = z["Xva"][:400], z["Yva"][:400]
    gen = _gen_windows(model, Xva, w, 777, device)
    def stats(ws):
        snr = [lp.win_specsnr(x) for x in ws]
        return np.percentile(snr, [10, 50, 90]).round(1)
    dhr_gen = [abs(dr.win_hr(g) - dr.win_hr(c)) for g, c in zip(gen, Xva)]
    dhr_real = [abs(dr.win_hr(y) - dr.win_hr(c)) for y, c in zip(Yva, Xva)]
    print(f"DEGCHECK n={len(Xva)} specSNR gen p10/50/90={stats(gen)} real={stats(Yva)} clean={stats(Xva)} | "
          f"|dHR| gen p50/90={np.percentile(dhr_gen, [50, 90]).round(1)} real={np.percentile(dhr_real, [50, 90]).round(1)}")
    print("CHECK_DONE")

# ---------- genclips ----------
def degrade_full(model, clean, w, seed, device, lo=-9.0, hi=0.0, tries=5):
    import torch
    x = band_znorm(clean)
    pad = max(0, L - len(x))
    if pad: x = np.pad(x, (0, pad), mode="reflect")
    starts = list(range(0, len(x) - L + 1, STRIDE))
    if starts[-1] != len(x) - L: starts.append(len(x) - L)
    conds = np.stack([dr.znorm(x[s:s + L]) for s in starts]).astype(np.float32)
    hann = np.hanning(L) + 1e-3
    for t in range(tries):
        r = _gen_windows(model, conds, w, seed + 1000 * t, device)
        msnr = float(np.mean([lp.win_specsnr(g) for g in r]))
        if lo <= msnr <= hi or t == tries - 1:
            acc = np.zeros(len(x)); den = np.zeros(len(x))
            for s, rw in zip(starts, r):
                acc[s:s + L] += rw * hann; den[s:s + L] += hann
            return (acc / den)[:len(clean)], msnr, t
    raise RuntimeError("unreachable")

def cmd_genclips(a):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(f"{RD4}/meta.json")); w = float(meta["w"])
    model = dr.UNet1D().to(device)
    model.load_state_dict(torch.load(f"{RD4}/ckpt.pt", map_location=device)); model.eval()
    bank = np.load(f"{PF}/ppg_bank_v2.npz")
    f0s = np.asarray(bank["f0"], float)
    n = a.frames
    os.makedirs(DW, exist_ok=True)
    bpms = [45, 55, 65, 75, 85, 95, 105, 120, 135, 150, 165, 180]
    for i, bpm in enumerate(bpms):
        cid = a.id0 + i
        order = np.argsort(np.abs(np.log(f0s / (bpm / 60.0))))
        k = int(order[i % 20])
        seg = bank["segments"][k]; r = (bpm / 60.0) / float(f0s[k])
        res = ss.resample(seg, max(int(round(len(seg) / r)), 8))
        clean = np.tile(res, int(np.ceil(n / len(res)) + 1))[:n].astype(np.float64)
        clean = band_znorm(clean)
        deg, msnr, t = degrade_full(model, clean, w, seed=9000 + cid, device=device)
        np.savez(f"{DW}/dg_{cid}.npz", inject=deg.astype(np.float32), gt=clean.astype(np.float32))
        print(f"GENCLIP id={cid} bpm={bpm} seg_f0={f0s[k]*60:.0f} ratio={r:.2f} meanSNR={msnr:.1f}dB draws={t+1}", flush=True)
    print("GENCLIPS_DONE")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pairs"); sub.add_parser("check")
    tp = sub.add_parser("train"); tp.add_argument("--steps", type=int, default=20000)
    gp = sub.add_parser("genclips"); gp.add_argument("--frames", type=int, default=1008); gp.add_argument("--id0", type=int, default=241)
    a = ap.parse_args()
    {"pairs": cmd_pairs, "train": cmd_train, "check": cmd_check, "genclips": cmd_genclips}[a.cmd](a)
