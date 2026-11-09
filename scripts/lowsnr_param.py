#!/usr/bin/env python3
"""18단계 체인 34: 저SNR 쌍 재학습(DLS) + 파라메트릭 harmonic template(PARM) — 설계서 §14.

사용자 제안 2건의 구현:
 (A) "노이즈 많은 데이터로 학습" — MMPD 지정 학습분할(피험자 1~6) backbone출력-GT 쌍을 refiner 학습에 추가
 (B) "정답 심박을 함수형으로 파라미터 고정" — 실 PPG 뱅크에서 harmonic amplitude ratio(r2,r3)를 고정하고
     창별 f0만 자유 파라미터로 두는 harmonic-template 재구성 (학습 불필요)

subcommands:
  pairs     — TRAINPOOL 쌍 + MMPD 1~6 쌍 → refiner_ls/pairs.npz
  train     — V1 아키텍처 재학습 (diffusion_refiner.cmd_train 재사용, RD만 refiner_ls로 교체)
  apply     — DLS refiner 전 축 적용 → CoTAF_TW_DLS_L{S}
  bank      — ppg_bank 131세그에서 r2,r3(진폭비)·r2p,r3p(파워비) 중앙값 → refiner_ls/bank_harm.json
  parm      — 창별 harmonic-template matching f0 + 형태 고정 재구성 → CoTAF_TW_PARM_L{S}
  gate      — 임의 모델의 MMPD 1~6 τ 보정 (calibrate_m 일반화) → refiner_ls/<out>
  applygate — 도메인 조건부 게이트: --domain 축에서만 발화, 그 외 축 원본 유지 → <name>{S}

모든 산출 모델은 'MMPD train-split calibrated (CoTAM 계열)' 표기.
"""
import argparse, glob, hashlib, importlib.util, json, os, pickle, re
import numpy as np
import scipy.signal as ss

ROOT = "/SPECPHYS_ROOT"
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
dr = _load("dr", f"{ROOT}/scripts/diffusion_refiner.py")
srm = _load("srm", f"{ROOT}/scripts/selective_refine.py")
agg2 = _load("agg2", f"{ROOT}/scripts/agg_model2.py")
PF, FS, L, STRIDE, AXES = dr.PF, dr.FS, dr.L, dr.STRIDE, dr.AXES
RD2 = f"{PF}/refiner_ls"
RD3 = f"{PF}/refiner_syn"

def pulse_wide(diffsig):
    """분석용 광대역 pulse space(0.5–8 Hz) — 고조파(2f0,3f0)가 협대역(0.6–3.3)에 잘리지 않게."""
    x = dr.fast_detrend(np.cumsum(np.asarray(diffsig, float)))
    b, a = ss.butter(1, [0.5 / FS * 2, 8.0 / FS * 2], btype="bandpass")
    return ss.filtfilt(b, a, np.double(x))

# ---------- pairs ----------
def cmd_pairs(a):
    Xtr, Ytr, Xva, Yva = [], [], [], []
    def add(vid, pred, lab):
        p = dr.pulse_space(pred); l = dr.pulse_space(lab)
        n = min(len(p), len(l))
        if n < L: return 0
        val = int(hashlib.md5(str(vid).encode()).hexdigest()[:8], 16) % 10 == 0
        c = 0
        for st in range(0, n - L + 1, STRIDE):
            tw = l[st:st + L]
            if tw.std() < 1e-6 or not dr.fundamental_ok(tw): continue
            cw = dr.znorm(p[st:st + L]); tw = dr.znorm(tw)
            (Xva if val else Xtr).append(cw.astype(np.float32))
            (Yva if val else Ytr).append(tw.astype(np.float32)); c += 1
        return c
    w0 = 0
    pk = glob.glob(f"{PF}/runs/ev_CoTAF_TW_L_TRAINPOOL/*/saved_test_outputs/*.pickle")[0]
    d = pickle.load(open(pk, "rb"))
    ntp = 0
    for vid in d["predictions"]:
        w0 += add(vid, dr.cat_chunks(d["predictions"][vid]), dr.cat_chunks(d["labels"][vid])); ntp += 1
    nmm = wmm = 0
    for k in range(1, 7):
        po = glob.glob(f"{PF}/runs/ev_CoTAF_TW_L_MMPD_B{k}/*/saved_test_outputs/*.pickle")
        if not po: continue
        d = pickle.load(open(po[0], "rb"))
        for vid in d["predictions"]:
            m = re.match(r"subject(\d+)_", str(vid))
            if not m or int(m.group(1)) >= 7: continue
            wmm += add(vid, dr.cat_chunks(d["predictions"][vid]), dr.cat_chunks(d["labels"][vid])); nmm += 1
    os.makedirs(RD2, exist_ok=True)
    np.savez(f"{RD2}/pairs.npz", Xtr=np.array(Xtr), Ytr=np.array(Ytr), Xva=np.array(Xva), Yva=np.array(Yva))
    print(f"PAIRS_LS trainpool_vids={ntp} wins={w0} | mmpd16_vids={nmm} wins={wmm} | train={len(Xtr)} val={len(Xva)}")

# ---------- train / apply (dr 재사용, RD 교체) ----------
def cmd_train(a):
    dr.RD = RD2
    dr.cmd_train(a)

def cmd_apply(a):
    dr.RD = RD2
    ns = argparse.Namespace(src=f"CoTAF_TW_L{a.suffix}", dst=f"CoTAF_TW_DLS_L{a.suffix}")
    dr.cmd_apply(ns)

# ---------- 대규모 저SNR 합성 쌍 (체인 35, 설계서 §16) ----------
def _band(x, lo, hi):
    b, a = ss.butter(1, [lo / FS * 2, hi / FS * 2], btype="bandpass")
    return ss.filtfilt(b, a, x)

def synth_noise(rng, f0):
    """rPPG 실패 모드를 본뜬 합성 잡음(단위 분산): 대역 잡음 + distractor 발진 + 2f/0.5f 교란 + 모션 버스트."""
    t = np.arange(L) / FS
    comps = [_band(rng.standard_normal(L), 0.5, 8.0)]
    if rng.random() < 0.5:  # distractor 발진(모션/flicker 주파수, AM 변조)
        fd = rng.uniform(0.3, 4.0)
        am = 1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t + rng.uniform(0, 6.28))
        comps.append(am * np.cos(2 * np.pi * fd * t + rng.uniform(0, 6.28)))
    if rng.random() < 0.4:  # harmonic doubling 방향 교란
        comps.append(np.cos(2 * np.pi * 2 * f0 * t + rng.uniform(0, 6.28)))
    if rng.random() < 0.2:  # subharmonic
        comps.append(np.cos(2 * np.pi * 0.5 * f0 * t + rng.uniform(0, 6.28)))
    if rng.random() < 0.3:  # 모션 버스트(국소 가우시안 창 잡음)
        c = rng.integers(0, L); wdt = rng.uniform(0.5, 2.0) * FS
        g = np.exp(-0.5 * ((np.arange(L) - c) / wdt) ** 2)
        comps.append(3 * g * _band(rng.standard_normal(L), 0.5, 8.0))
    n = np.zeros(L)
    for w, c in zip(rng.dirichlet(np.ones(len(comps))), comps):
        n += w * c / (c.std() + 1e-8)
    return n / (n.std() + 1e-8)

def win_specsnr(w):
    """창 단위 스펙트럼 SNR(dB) — selective_refine.spec_snr과 동일 정의(입력이 이미 pulse-space 창)."""
    f, p = ss.periodogram(w, FS)
    band = (f >= 0.6) & (f <= 3.3)
    if not band.any() or p[band].max() <= 0: return -20.0
    f0 = f[band][p[band].argmax()]
    sigm = band & ((np.abs(f - f0) <= 0.1) | (np.abs(f - 2 * f0) <= 0.1))
    s = p[sigm].sum(); n = p[band & ~sigm].sum()
    return float(10 * np.log10(s / max(n, 1e-12)))

def cmd_pairs_syn(a):
    z = np.load(f"{RD2}/pairs.npz")  # 체인 34 실 쌍(TRAINPOOL+MMPD 1~6) 재사용
    gm = json.load(open(f"{PF}/refiner/gate_m.json"))
    snr_pool = np.array([r["snr"] for r in gm["rows"]])  # MMPD 1~6 실측 SNR 분포
    rng = np.random.default_rng(100)
    # 깨끗한 타깃: 실 쌍의 GT 창 + ppg_bank 세그 창
    bw = []
    for seg in np.load(f"{PF}/ppg_bank.npz")["segments"]:
        xx = _band(dr.fast_detrend(np.asarray(seg, float)), 0.6, 3.3)
        for st in range(0, len(xx) - L + 1, STRIDE):
            w = dr.znorm(xx[st:st + L])
            if dr.fundamental_ok(w): bw.append(w.astype(np.float32))
    clean = np.concatenate([z["Ytr"], np.array(bw)], 0)
    Xs, Ys = [], []
    for w in clean:
        f0 = dr.win_hr(w) / 60.0
        for _ in range(a.k):
            snr = float(rng.choice(snr_pool)) + rng.normal(0, 1.0)
            # -3 dB 보정: znorm 후 재측정 시 합성 SNR이 목표보다 ~3 dB 높게 나오는 편향 상쇄
            amp = float(np.clip(10 ** (-(snr - 3.0) / 20), 0.05, 20))
            cond = dr.znorm(w + amp * synth_noise(rng, f0))
            Xs.append(cond.astype(np.float32)); Ys.append(np.asarray(w, np.float32))
    Xtr = np.concatenate([z["Xtr"], np.array(Xs)], 0)
    Ytr = np.concatenate([z["Ytr"], np.array(Ys)], 0)
    os.makedirs(RD3, exist_ok=True)
    np.savez(f"{RD3}/pairs.npz", Xtr=Xtr, Ytr=Ytr, Xva=z["Xva"], Yva=z["Yva"])
    samp = [win_specsnr(Xs[i]) for i in rng.choice(len(Xs), 500, replace=False)]
    print(f"PAIRS_SYN clean={len(clean)}(real {len(z['Ytr'])}+bank {len(bw)}) K={a.k} syn={len(Xs)} "
          f"train={len(Xtr)} val={len(z['Xva'])} | syn-SNR p10/50/90={np.percentile(samp, [10, 50, 90]).round(1)} "
          f"real-SNR p10/50/90={np.percentile(snr_pool, [10, 50, 90]).round(1)}")

def cmd_train_syn(a):
    dr.RD = RD3
    dr.cmd_train(a)

def cmd_apply_syn(a):
    dr.RD = RD3
    ns = argparse.Namespace(src=f"CoTAF_TW_L{a.suffix}", dst=f"CoTAF_TW_DSYN_L{a.suffix}")
    dr.cmd_apply(ns)

# ---------- bank ----------
def cmd_bank(a):
    z = np.load(f"{PF}/ppg_bank.npz")
    r2a, r3a, r2p, r3p = [], [], [], []
    f0s = np.asarray(z["f0"], float)
    if np.median(f0s) > 10: f0s = f0s / 60.0  # BPM → Hz
    for seg, f0 in zip(z["segments"], f0s):
        x = np.asarray(seg, float); x = (x - x.mean()) / (x.std() + 1e-8)
        t = np.arange(len(x)) / FS
        amps = []
        for k in (1, 2, 3):
            if k * f0 > FS / 2 - 0.5: amps.append(0.0); continue
            zc = (x * np.exp(-2j * np.pi * k * f0 * t)).mean()
            amps.append(2 * abs(zc))
        if amps[0] <= 1e-6: continue
        r2a.append(amps[1] / amps[0]); r3a.append(amps[2] / amps[0])
        r2p.append((amps[1] / amps[0]) ** 2); r3p.append((amps[2] / amps[0]) ** 2)
    out = dict(r2a=float(np.median(r2a)), r3a=float(np.median(r3a)),
               r2p=float(np.median(r2p)), r3p=float(np.median(r3p)),
               n=len(r2a), f0_hz_med=float(np.median(f0s)))
    os.makedirs(RD2, exist_ok=True)
    json.dump(out, open(f"{RD2}/bank_harm.json", "w"), indent=1)
    print(f"BANK_HARM n={out['n']} r2a={out['r2a']:.3f} r3a={out['r3a']:.3f} "
          f"r2p={out['r2p']:.3f} r3p={out['r3p']:.3f} f0med={out['f0_hz_med']:.2f}Hz")

# ---------- parm ----------
def parm_full(sig, hb):
    n = len(sig)
    if n < 40: return np.asarray(sig, float).copy()
    x = pulse_wide(sig)
    pad = max(0, L - len(x))
    if pad: x = np.pad(x, (0, pad), mode="reflect")
    starts = list(range(0, len(x) - L + 1, STRIDE))
    if starts[-1] != len(x) - L: starts.append(len(x) - L)
    hann = np.hanning(L) + 1e-3
    acc = np.zeros(len(x)); den = np.zeros(len(x))
    t = np.arange(L) / FS
    fr = np.fft.rfftfreq(L * 8, 1 / FS)
    band = (fr >= 0.6) & (fr <= 3.3)
    cand = fr[band]
    r2a, r3a, r2p, r3p = hb["r2a"], hb["r3a"], hb["r2p"], hb["r3p"]
    for s in starts:
        w = dr.znorm(x[s:s + L])
        P = np.abs(np.fft.rfft(w - w.mean(), n=L * 8)) ** 2
        S = (np.interp(cand, fr, P) + r2p * np.interp(2 * cand, fr, P, right=0)
             + r3p * np.interp(3 * cand, fr, P, right=0)) / (1 + r2p + r3p)
        f0 = float(cand[S.argmax()])
        rec = np.zeros(L); a1 = 1.0
        for k, rk in ((1, 1.0), (2, r2a), (3, r3a)):
            if k * f0 >= FS / 2: continue
            zc = (w * np.exp(-2j * np.pi * k * f0 * t)).mean()
            if k == 1: a1 = 2 * abs(zc)
            rec += a1 * rk * np.cos(2 * np.pi * k * f0 * t + np.angle(zc))
        rec = dr.znorm(rec)
        acc[s:s + L] += rec * hann; den[s:s + L] += hann
    out = (acc / den)[:n]
    return np.diff(out, prepend=out[:1])

def cmd_parm(a):
    import torch
    hb = json.load(open(f"{RD2}/bank_harm.json"))
    S = a.suffix; done = 0
    for ax in AXES:
        pks = glob.glob(f"{PF}/runs/ev_CoTAF_TW_L{S}_{ax}/*/saved_test_outputs/*.pickle")
        if not pks:
            print(f"skip {ax}"); continue
        d = pickle.load(open(pks[0], "rb"))
        newp = {}
        for vid in d["predictions"]:
            ch = d["predictions"][vid]
            ref = parm_full(dr.cat_chunks(ch), hb)
            nd = {}; off = 0
            for k in sorted(ch):
                ln = ch[k].numel()
                nd[k] = torch.from_numpy(ref[off:off + ln].astype(np.float32)); off += ln
            newp[vid] = nd
        od = f"{PF}/runs/ev_CoTAF_TW_PARM_L{S}_{ax}/parm/saved_test_outputs"
        os.makedirs(od, exist_ok=True)
        pickle.dump(dict(predictions=newp, labels=d["labels"], label_type=d.get("label_type"), fs=d.get("fs", FS)),
                    open(f"{od}/parm_outputs.pickle", "wb"))
        done += 1
        print(f"PARM{S or '_s100'} {ax} vids={len(newp)}", flush=True)
    print(f"PARM_DONE CoTAF_TW_PARM_L{S} axes={done}")

# ---------- gate / applygate ----------
def cmd_gate(a):
    rows = []
    for k in range(1, 7):
        po = glob.glob(f"{PF}/runs/ev_CoTAF_TW_L_MMPD_B{k}/*/saved_test_outputs/*.pickle")
        pr = glob.glob(f"{PF}/runs/ev_{a.model}_MMPD_B{k}/*/saved_test_outputs/*.pickle")
        if not po or not pr: continue
        do = pickle.load(open(po[0], "rb")); dref = pickle.load(open(pr[0], "rb"))
        for vid in do["predictions"]:
            m = re.match(r"subject(\d+)_", str(vid))
            if not m or int(m.group(1)) >= 7: continue
            if vid not in dref["predictions"]: continue
            pred = dr.cat_chunks(do["predictions"][vid]); lab = dr.cat_chunks(do["labels"][vid])
            ref = dr.cat_chunks(dref["predictions"][vid])
            hr_gt = agg2.toolbox_hr(lab)
            rows.append(dict(vid=str(vid), snr=srm.spec_snr(pred),
                             e_o=abs(agg2.toolbox_hr(pred) - hr_gt), e_r=abs(agg2.toolbox_hr(ref) - hr_gt)))
    snrs = np.array([r["snr"] for r in rows]); eo = np.array([r["e_o"] for r in rows]); er = np.array([r["e_r"] for r in rows])
    taus = np.arange(-10, 15.01, 0.25)
    maes = np.array([np.where(snrs < t, er, eo).mean() for t in taus])
    i = int(maes.argmin()); tau = float(taus[i])
    print(f"GATE[{a.model}] n={len(rows)} orig={eo.mean():.3f} all-ref={er.mean():.3f} "
          f"gated(τ*={tau:+.2f}dB)={maes[i]:.3f} frac={float((snrs < tau).mean()):.2f}")
    os.makedirs(RD2, exist_ok=True)
    json.dump(dict(tau=tau, model=a.model, mae_orig=float(eo.mean()), mae_all_ref=float(er.mean()),
                   mae_gated=float(maes[i]), frac=float((snrs < tau).mean()), n=len(rows)),
              open(f"{RD2}/{a.out}", "w"), indent=1)
    print("GATE_DONE")

def cmd_applygate(a):
    tau = json.load(open(f"{RD2}/{a.gate}"))["tau"]
    S = a.suffix; tot = ref = 0
    for ax in AXES:
        po = glob.glob(f"{PF}/runs/ev_CoTAF_TW_L{S}_{ax}/*/saved_test_outputs/*.pickle")
        if not po: print(f"skip {ax}"); continue
        do = pickle.load(open(po[0], "rb"))
        dref = None
        if ax.startswith(a.domain):
            pr = glob.glob(f"{PF}/runs/ev_{a.model}{S}_{ax}/*/saved_test_outputs/*.pickle")
            if pr: dref = pickle.load(open(pr[0], "rb"))
        newp = {}
        for vid in do["predictions"]:
            use = False
            if dref is not None and vid in dref["predictions"]:
                use = srm.spec_snr(dr.cat_chunks(do["predictions"][vid])) < tau
            newp[vid] = dref["predictions"][vid] if use else do["predictions"][vid]
            tot += 1; ref += int(use)
        od = f"{PF}/runs/ev_{a.name}{S}_{ax}/gated/saved_test_outputs"
        os.makedirs(od, exist_ok=True)
        pickle.dump(dict(predictions=newp, labels=do["labels"], label_type=do.get("label_type"), fs=do.get("fs", FS)),
                    open(f"{od}/gated_outputs.pickle", "wb"))
    print(f"APPLYGATE_DONE {a.name}{S} refined {ref}/{tot} (τ={tau:+.2f}dB, domain={a.domain})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pairs"); sub.add_parser("bank")
    ps = sub.add_parser("pairs_syn"); ps.add_argument("--k", type=int, default=12)
    ts = sub.add_parser("train_syn"); ts.add_argument("--steps", type=int, default=40000)
    aps = sub.add_parser("apply_syn"); aps.add_argument("--suffix", default="")
    tp = sub.add_parser("train"); tp.add_argument("--steps", type=int, default=20000)
    app = sub.add_parser("apply"); app.add_argument("--suffix", default="")
    pp = sub.add_parser("parm"); pp.add_argument("--suffix", default="")
    gp = sub.add_parser("gate"); gp.add_argument("--model", required=True); gp.add_argument("--out", required=True)
    ag = sub.add_parser("applygate")
    ag.add_argument("--suffix", default=""); ag.add_argument("--model", required=True)
    ag.add_argument("--gate", required=True); ag.add_argument("--name", required=True)
    ag.add_argument("--domain", default="MMPD")
    a = ap.parse_args()
    {"pairs": cmd_pairs, "train": cmd_train, "apply": cmd_apply, "bank": cmd_bank,
     "pairs_syn": cmd_pairs_syn, "train_syn": cmd_train_syn, "apply_syn": cmd_apply_syn,
     "parm": cmd_parm, "gate": cmd_gate, "applygate": cmd_applygate}[a.cmd](a)
