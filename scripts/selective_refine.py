#!/usr/bin/env python3
"""18단계 체인 33: confidence-gated selective refinement (설계서 §10-3 후속).

calibrate — TRAINPOOL 덤프(학습 분할 203클립)에 V1 head를 적용해 per-video (SNR, err_orig, err_ref)를 만들고,
            스펙트럼 SNR 임계 τ*를 학습 분할 MAE 최소화로 선택(평가 축 불사용 — 누수 차단).
apply     — τ* 게이트: SNR<τ*이면 refined(CoTAF_TW_DIFF_L*), 아니면 원본(CoTAF_TW_L*) → CoTAF_TW_GATE_L*
"""
import argparse, glob, hashlib, importlib.util, json, os, pickle
import numpy as np
import torch
import scipy.signal as ss

ROOT = "/SPECPHYS_ROOT"
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
dr = _load("dr", f"{ROOT}/scripts/diffusion_refiner.py")
agg2 = _load("agg2", f"{ROOT}/scripts/agg_model2.py")
PF, RD, AXES, FS = dr.PF, dr.RD, dr.AXES, dr.FS

def spec_snr(diffsig):
    """de Haan식 스펙트럼 SNR(dB): f0±0.1Hz + 2f0±0.1Hz 파워 vs 나머지(0.6–3.3Hz)."""
    x = dr.pulse_space(diffsig)
    f, p = ss.periodogram(x, FS)
    band = (f >= 0.6) & (f <= 3.3)
    if not band.any() or p[band].max() <= 0: return -20.0
    f0 = f[band][p[band].argmax()]
    sigm = band & ((np.abs(f - f0) <= 0.1) | (np.abs(f - 2 * f0) <= 0.1))
    s = p[sigm].sum(); n = p[band & ~sigm].sum()
    return float(10 * np.log10(s / max(n, 1e-12)))

def cmd_calibrate(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(f"{RD}/meta.json")); w = float(meta["w"])
    model = dr.UNet1D().to(device)
    model.load_state_dict(torch.load(f"{RD}/ckpt.pt", map_location=device)); model.eval()
    pk = glob.glob(f"{PF}/runs/ev_CoTAF_TW_L_TRAINPOOL/*/saved_test_outputs/*.pickle")[0]
    d = pickle.load(open(pk, "rb"))
    rows = []
    for vid in d["predictions"]:
        pred = dr.cat_chunks(d["predictions"][vid]); lab = dr.cat_chunks(d["labels"][vid])
        if len(pred) < 160: continue
        hr_gt = agg2.toolbox_hr(lab)
        e_o = abs(agg2.toolbox_hr(pred) - hr_gt)
        seed = int(hashlib.md5(f"{vid}".encode()).hexdigest()[:8], 16) % (2 ** 31)
        ref = dr.refine_full(model, pred, w, seed, device)
        e_r = abs(agg2.toolbox_hr(ref) - hr_gt)
        rows.append(dict(vid=str(vid), snr=spec_snr(pred), e_o=e_o, e_r=e_r))
        if len(rows) % 50 == 0: print(f"calib {len(rows)}", flush=True)
    snrs = np.array([r["snr"] for r in rows]); eo = np.array([r["e_o"] for r in rows]); er = np.array([r["e_r"] for r in rows])
    taus = np.arange(-10, 15.01, 0.25)
    maes = np.array([np.where(snrs < t, er, eo).mean() for t in taus])
    i = int(maes.argmin()); tau = float(taus[i])
    print(f"CALIB n={len(rows)} MAE orig={eo.mean():.3f} all-refined={er.mean():.3f} "
          f"gated(τ*={tau:+.2f}dB)={maes[i]:.3f} refine-frac={float((snrs < tau).mean()):.2f}")
    json.dump(dict(tau=tau, mae_orig=float(eo.mean()), mae_all_ref=float(er.mean()),
                   mae_gated=float(maes[i]), frac=float((snrs < tau).mean()), n=len(rows), rows=rows),
              open(f"{RD}/gate.json", "w"), indent=1)
    print("CALIB_DONE")

def cmd_calibrate_m(a):
    """MMPD 지정 학습 분할(피험자 1~6, 120클립)에서 τ 보정 — 결과 모델은 'MMPD train-split calibrated'로 별도 표기.
    GPU 불필요: 원본(ev_CoTAF_TW_L_MMPD_B*)·refined(ev_CoTAF_TW_DIFF_L_MMPD_B*) pickle 재사용."""
    import re
    rows = []
    for k in range(1, 7):
        po = glob.glob(f"{PF}/runs/ev_CoTAF_TW_L_MMPD_B{k}/*/saved_test_outputs/*.pickle")
        pr = glob.glob(f"{PF}/runs/ev_CoTAF_TW_DIFF_L_MMPD_B{k}/*/saved_test_outputs/*.pickle")
        if not po or not pr: continue
        do = pickle.load(open(po[0], "rb")); dref = pickle.load(open(pr[0], "rb"))
        for vid in do["predictions"]:
            m = re.match(r"subject(\d+)_", str(vid))
            if not m or int(m.group(1)) >= 7: continue  # 학습 분할 1~6만
            pred = dr.cat_chunks(do["predictions"][vid]); lab = dr.cat_chunks(do["labels"][vid])
            ref = dr.cat_chunks(dref["predictions"][vid])
            hr_gt = agg2.toolbox_hr(lab)
            rows.append(dict(vid=str(vid), snr=spec_snr(pred),
                             e_o=abs(agg2.toolbox_hr(pred) - hr_gt), e_r=abs(agg2.toolbox_hr(ref) - hr_gt)))
    snrs = np.array([r["snr"] for r in rows]); eo = np.array([r["e_o"] for r in rows]); er = np.array([r["e_r"] for r in rows])
    taus = np.arange(-10, 15.01, 0.25)
    maes = np.array([np.where(snrs < t, er, eo).mean() for t in taus])
    i = int(maes.argmin()); tau = float(taus[i])
    print(f"CALIB_M n={len(rows)} MAE orig={eo.mean():.3f} all-refined={er.mean():.3f} "
          f"gated(τ*={tau:+.2f}dB)={maes[i]:.3f} refine-frac={float((snrs < tau).mean()):.2f} "
          f"snr[p10,p50,p90]={np.percentile(snrs,[10,50,90]).round(2)}")
    json.dump(dict(tau=tau, mae_orig=float(eo.mean()), mae_all_ref=float(er.mean()),
                   mae_gated=float(maes[i]), frac=float((snrs < tau).mean()), n=len(rows), rows=rows),
              open(f"{RD}/gate_m.json", "w"), indent=1)
    print("CALIB_M_DONE")

def cmd_apply(a):
    tau = json.load(open(f"{RD}/{a.gate}"))["tau"]
    S = a.suffix
    tot = ref = 0
    for ax in AXES:
        po = glob.glob(f"{PF}/runs/ev_CoTAF_TW_L{S}_{ax}/*/saved_test_outputs/*.pickle")
        pr = glob.glob(f"{PF}/runs/ev_CoTAF_TW_DIFF_L{S}_{ax}/*/saved_test_outputs/*.pickle")
        if not po or not pr: print(f"skip {ax}"); continue
        do = pickle.load(open(po[0], "rb")); dref = pickle.load(open(pr[0], "rb"))
        newp = {}
        for vid in do["predictions"]:
            sig = dr.cat_chunks(do["predictions"][vid])
            use_ref = spec_snr(sig) < tau and vid in dref["predictions"]
            newp[vid] = dref["predictions"][vid] if use_ref else do["predictions"][vid]
            tot += 1; ref += int(use_ref)
        od = f"{PF}/runs/ev_{a.name}{S}_{ax}/gated/saved_test_outputs"
        os.makedirs(od, exist_ok=True)
        pickle.dump(dict(predictions=newp, labels=do["labels"], label_type=do.get("label_type"), fs=do.get("fs", FS)),
                    open(f"{od}/gated_outputs.pickle", "wb"))
        print(f"GATE{S or '_s100'} {ax} vids={len(newp)}", flush=True)
    print(f"APPLY_DONE {a.name}{S} refined {ref}/{tot} (τ={tau:+.2f}dB)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate"); sub.add_parser("calibrate_m")
    app = sub.add_parser("apply"); app.add_argument("--suffix", default=""); app.add_argument("--gate", default="gate.json"); app.add_argument("--name", default="CoTAF_TW_GATE_L")
    a = ap.parse_args()
    {"calibrate": cmd_calibrate, "calibrate_m": cmd_calibrate_m, "apply": cmd_apply}[a.cmd](a)
