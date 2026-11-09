#!/usr/bin/env python3
"""[52 문서 Phase A] NF-Phys 슬롯 분리 진단 — 학습된 체크포인트를 통제 교란 축(RTNI: 드리프트+1.1 Hz 플리커, RTNC: +움직임+센서)과 청정 축(PURE)의 캐시 청크에 돌려
  맥파 슬롯 s_p / 교란 슬롯 s_n 이 각각 정답 맥파(PPG)·알려진 교란 궤적(gain = 1 + drift + flicker, make_noisy_real 과 같은 시드로 재생성)과 얼마나 상관되는지 측정.
  python scripts/nf_slot_probe.py → 표 + results/nf_slot_probe.json
지표(청크별 → 평균): corr(s_p, dPPG), corr(s_n, dGain)[교란 포착], corr(s_p, dGain)[누설], corr(g_init, dGain)[초기값 자체], 출력 파형 SNR@HR, 청크 HR 오차."""
import glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/SPECPHYS_ROOT/repos/rPPG-Toolbox"); sys.path.insert(0, "/SPECPHYS_ROOT/scripts"); sys.path.insert(0, "/SPECPHYS_ROOT/generator")
from add_noise import smooth_walk  # noqa
from neural_methods.loss import NFLoss as NFL  # noqa
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; R = f"{PFA}/../results"; FPS = 30.0; L = 160
CONDS = {"RTNI": (0, dict(illum_drift=0.15, flicker_amp=0.03, flicker_hz=1.1)), "RTNC": (2, dict(illum_drift=0.15, flicker_amp=0.03, flicker_hz=1.1, motion_px=10, rot_deg=2.0, sensor_std=2))}
SUBS = [42, 43, 44, 45, 46, 47, 48, 49]
MD = {"FRAME_NUM": 160, "MD_TYPE": "NMF", "MD_FSAM": True, "MD_TRANSFORM": "T_KAB", "MD_S": 1, "MD_R": 1, "MD_STEPS": 4, "MD_INFERENCE": True, "MD_RESIDUAL": True}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gain_trace(cond, si, n):
    ci, nz = CONDS[cond]; rng = np.random.default_rng(700 + ci * 10 + si); t = np.arange(n) / FPS
    drift = smooth_walk(n, nz["illum_drift"] / max(np.sqrt(n), 1), int(FPS), rng); drift = drift - drift.mean()
    flick = nz["flicker_amp"] * np.sin(2 * np.pi * nz["flicker_hz"] * t)
    return np.clip(1 + drift + flick, 0.3, 1.7)


def nframes(cond, s):
    import cv2
    cap = cv2.VideoCapture(f"{PFA}/{cond}/subject{s}/vid.avi"); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release(); return n


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def load_model(name, rank1):
    os.environ["NF_RANK1"] = "1" if rank1 else "0"; os.environ["RPPG_NF"] = "1"
    import importlib, neural_methods.model.FactorizePhys.FactorizePhysNF as M; importlib.reload(M)
    m = M.FactorizePhysNF(frames=160, md_config=dict(MD), in_channels=3, dropout=0.1, device=dev).to(dev)
    sd = torch.load(f"{PFA}/runs/finetune_{name}/PreTrainedModels/SYNTH_{name}_factorizephys_Epoch9.pth", map_location=dev, weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}; m.load_state_dict(sd, strict=False); m.eval(); return m


def hr_peak(x, fs=FPS):
    P, bpm = NFL.power(x, fs, 30, 200); return bpm[P.argmax(-1)]


res = {}
for name, rank1 in (("NF_HYB_L", False), ("NF_HYB_NOSUP_L", False), ("NF_HYB_R1_L", True)):
    m = load_model(name, rank1); res[name] = {}
    for cond in ("RTNI", "RTNC", "PURE"):
        cache = f"{PFA}/cache/fp_real_{cond}"; acc = {k: [] for k in ("sp_dppg", "sp_ppg", "sn_dgain", "sn_dppg", "sp_dgain", "g_dgain", "snr", "hr_err")}
        files = sorted(glob.glob(f"{cache}/*_input*.npy"))
        for f in files:
            base = os.path.basename(f); subj = base.split("_input")[0]; cid = int(base.split("_input")[1].split(".")[0])
            x = np.load(f).astype(np.float32); y = np.load(f.replace("_input", "_label")).astype(np.float32)
            if x.shape[0] != L: continue
            xt = torch.tensor(x).permute(3, 0, 1, 2).unsqueeze(0).to(dev)                 # (1,3,T,H,W)
            xt = torch.cat([xt, xt[:, :, -1:]], 2)                                            # trainer 와 같이 마지막 프레임 추가
            with torch.no_grad(): out, _, _, _ = m(xt)
            aux = m.nf_aux; sp = NFL.hp(aux["s_p"]).cpu().numpy()[0]; g = aux["g_diff"].cpu().numpy()[0]
            sn = NFL.hp(aux["s_n"]).cpu().numpy()[0] if aux["s_n"] is not None else None
            dppg = np.diff(np.append(y, y[-1])); ppg = y
            hr_l = float(hr_peak(torch.tensor(y)[None])[0]); o = out[0]
            P, bpm = NFL.power(o[None], FPS, 30, 200); P = P[0]; w = (bpm - hr_l).abs() <= 3
            acc["snr"].append(float(P[w].sum() / (P.sum() + 1e-8))); acc["hr_err"].append(abs(float(bpm[P.argmax()]) - hr_l))
            acc["sp_dppg"].append(abs(corr(sp, dppg))); acc["sp_ppg"].append(abs(corr(sp, ppg)))
            if cond in CONDS:
                s_idx = SUBS.index(int(subj.replace("subject", ""))); n = nframes(cond, int(subj.replace("subject", "")))
                gt = gain_trace(cond, s_idx, n); seg = gt[cid * L: cid * L + L + 1]
                if len(seg) < L + 1: seg = np.append(seg, [seg[-1]] * (L + 1 - len(seg)))
                dg = np.diff(seg)
                acc["sp_dgain"].append(abs(corr(sp, dg))); acc["g_dgain"].append(abs(corr(g, dg)))
                if sn is not None: acc["sn_dgain"].append(abs(corr(sn, dg)))
            if sn is not None: acc["sn_dppg"].append(abs(corr(sn, dppg)))
        res[name][cond] = {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}; res[name][cond]["n_chunks"] = len(files)
json.dump(res, open(f"{R}/nf_slot_probe.json", "w"), indent=1)
print(f"{'모델':16s} {'축':5s} {'n':>4s} | corr(s_p,dPPG) corr(s_p,PPG) | corr(s_n,dGain) corr(g_init,dGain) | 누설 corr(s_p,dGain) | corr(s_n,dPPG) | SNR@HR  HR오차")
for name in res:
    for cond, v in res[name].items():
        f = lambda k: f"{v[k]:.3f}" if v.get(k) is not None else "  –  "
        print(f"{name:16s} {cond:5s} {v['n_chunks']:4d} | {f('sp_dppg'):>13s} {f('sp_ppg'):>12s} | {f('sn_dgain'):>14s} {f('g_dgain'):>17s} | {f('sp_dgain'):>19s} | {f('sn_dppg'):>13s} | {f('snr'):>6s} {f('hr_err'):>7s}")
