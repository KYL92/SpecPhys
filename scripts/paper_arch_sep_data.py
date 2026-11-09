#!/usr/bin/env python3
"""[63 문서 §2] 합성 데이터의 '심박 스펙트럼'과 '잡음 스펙트럼' 분리 그림 재료(학습 없음, CPU).
 ① 풀 안 이중 렌더 클립: PPG 라벨 스펙트럼과 n(t) 스펙트럼(정답). ② 풀 밖 합성 클립 전부: 최종 체크포인트(시드별)에서 투영 전후 특징, 펄스·잡음 기저, 출력 스펙트럼.
 평가는 고른 클립이 아니라 집합 전체를 돈다(STRESS_OFFBAND6, STRESS_PMOT6, STRESS_INBAND6, STRESS_NEAR6, TEST_C3). 판정: 정점이 심박 또는 잡음 주파수 ±6 bpm 안(bin 간격 11.25 bpm 의 절반 근처).
  python scripts/paper_arch_sep_data.py → results/figs_arch/sep_data.json"""
import glob, json, os, sys
CELL = os.environ.get("SEP_CELL", "NF_NSP2_L"); ARCH = "" if CELL == "NF_SKREF_REP_L" else "nsp"; OUTJ = os.environ.get("SEP_OUT") or ("sep_data.json" if ARCH else "sep_data_noproj.json")   # [64 문서] SEP_OUT·SEP_SEEDS 로 다른 셀·시드 묶음   # SEP_CELL=NF_SKREF_REP_L: 투영 없는 같은 레시피로 같은 클립을 재서 층의 몫을 본다
os.environ.update(RPPG_NF="1", NF_ARCH=ARCH, NF_NSP_LAM="0.01", CUDA_VISIBLE_DEVICES=""); import cv2, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(16)
ROOT = "/SPECPHYS_ROOT"; PFA = f"{ROOT}/_workspace/data/physformer"; TB = f"{ROOT}/repos/rPPG-Toolbox"; OUT = f"{ROOT}/_workspace/data/results/figs_arch"
sys.path.insert(0, TB); os.chdir(TB)
import neural_methods.model.FactorizePhys.FactorizePhys as FP
from neural_methods.model.FactorizePhys.FactorizePhysNF import FactorizePhysNF, noise_refs, nsp_component
from neural_methods.loss import NFLoss as NFL
from transformers import SegformerForSemanticSegmentation
PM = SegformerForSemanticSegmentation.from_pretrained(f"{ROOT}/_workspace/models/face_parsing").eval()
def spec(a, fs=30.0, lo=30, hi=200, pad=8):
    a = np.asarray(a, dtype=np.float64); a = a - a.mean(-1, keepdims=True); N = a.shape[-1] * pad; A = np.abs(np.fft.rfft(a, n=N, axis=-1)) ** 2; f = np.fft.rfftfreq(N, 1 / fs) * 60; s = (f >= lo) & (f <= hi); return f[s], A[..., s]
def read(path):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f[..., ::-1])
    return np.stack(fr)
fc = cv2.CascadeClassifier(f"{TB}/dataset/haarcascade_frontalface_default.xml")
def crop_all(fr):
    bx = fc.detectMultiScale(fr[0][..., ::-1].astype(np.uint8), minSize=(30, 30), minNeighbors=3)
    if len(bx) == 0: x0, y0, w0, h0 = 0, 0, fr.shape[2], fr.shape[1]; c = 1.0
    else: x0, y0, w0, h0 = max(bx, key=lambda b: b[2] * b[3]); c = 1.5
    x1, y1 = max(0, int(x0 - (c - 1) / 2 * w0)), max(0, int(y0 - (c - 1) / 2 * h0)); w1, h1 = int(c * w0), int(c * h0); return fr[:, y1:y1 + h1, x1:x1 + w1]
def parse(seg):
    L = len(seg); idx = np.linspace(0, L - 1, 8).round().astype(int); xp = torch.from_numpy(np.stack([cv2.resize(seg[i], (128, 128), interpolation=cv2.INTER_AREA) for i in idx])).permute(0, 3, 1, 2).float() / 255
    xp = F.interpolate(xp, size=(384, 384), mode="bilinear", align_corners=False); xp = (xp - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
    with torch.no_grad(): lab = PM(pixel_values=xp).logits.argmax(1)
    lab = F.interpolate(lab[:, None].float(), size=(64, 64), mode="nearest")[:, 0].long(); return sum((lab == i).float() for i in (1, 2)).mean(0).numpy()
cfgM = {**FP.model_config, "MD_FSAM": True, "MD_INFERENCE": True, "MD_RESIDUAL": True}
SEEDS = [s for s in tuple(int(x) for x in os.environ.get("SEP_SEEDS", "1600,1700,1800,1900,2000,2100,2200,2300,2400,2500").split(",")) if glob.glob(f"{PFA}/runs/finetune_{CELL}_s{s}/PreTrainedModels/*Epoch9.pth")]; print("seeds", SEEDS)
def load(ck, cls=FactorizePhysNF):
    m = cls(frames=160, md_config=cfgM, in_channels=3, dropout=0.1, device=torch.device("cpu")); sd = torch.load(ck, map_location="cpu"); m.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False); m.eval(); return m
MS = {s: load(glob.glob(f"{PFA}/runs/finetune_{CELL}_s{s}/PreTrainedModels/*Epoch9.pth")[0]) for s in SEEDS}
pub = load(f"{TB}/final_model_release/UBFC-rPPG_FactorizePhys_FSAM_Res.pth", FP.FactorizePhys)
hold = {}
for s, m in MS.items(): m.rppg_head.conv_block.register_forward_hook(lambda mm, i, o: hold.__setitem__("v", o.detach()))
SETS = {"STRESS_OFFBAND6": "stress_offband6", "STRESS_PMOT6": "stress_pmot6", "STRESS_INBAND6": "stress_inband6", "STRESS_NEAR6": "stress_near6", "TEST_C3": "test_C3"}
def nz(x, f):  # 30~200 bpm 정규화
    return x / (x.sum(-1, keepdims=True) + 1e-12)
R = {"held_out": {}, "seeds": SEEDS}
for st, cf in SETS.items():
    subs = json.load(open(f"{PFA}/configs/{cf}.json"))["subjects"]
    for sb in subs:
        vid = f"{PFA}/{st}/subject{sb['id']}/vid.avi"
        if not os.path.exists(vid): continue
        fr = crop_all(read(vid)); n_ = sb.get("noise", {}); nf = n_.get("flicker_hz") or n_.get("periodic_motion_hz"); kind = "flicker" if n_.get("flicker_hz") else "motion"; nb = float(nf) * 60; hr = float(sb["bpm"])
        acc = {k: [] for k in ("before", "after", "out", "bp", "bn", "pub")}; peaks = {k: [] for k in acc}
        for T0 in range(0, len(fr) - 161, 160):
            seg = fr[T0:T0 + 161]; small = np.stack([cv2.resize(f_, (72, 72), interpolation=cv2.INTER_AREA) for f_ in seg]).astype(np.float32); ps = cv2.resize(parse(seg), (72, 72), interpolation=cv2.INTER_LINEAR); xw = small * (0.2 + 0.8 * ps)[None, :, :, None]; X = torch.from_numpy(xw).permute(3, 0, 1, 2)[None].float()
            with torch.no_grad(): op = pub(torch.from_numpy(small).permute(3, 0, 1, 2)[None].float())[0][0].numpy()
            f, S = spec(op); acc["pub"].append(nz(S, f)); peaks["pub"].append(float(f[np.argmax(S)]))
            for s, m in MS.items():
                with torch.no_grad(): o = m(X)[0][0].numpy()
                v = hold["v"]; comp = nsp_component(v, noise_refs(X)) if ARCH else torch.zeros_like(v); Xb = v[0].permute(0, 2, 3, 1).reshape(-1, 160).numpy(); Xa = Xb - comp[0].permute(0, 2, 3, 1).reshape(-1, 160).numpy(); ok = Xb.std(-1) > 1e-6
                _, Pb = spec(Xb[ok]); _, Pa = spec(Xa[ok]); tot = Pb.sum(-1, keepdims=True); Sb = (Pb / tot).mean(0); Sa = (Pa / tot).mean(0)
                B_ = m.nf_aux["bases"][0]; bp = NFL.hp(B_[:, 0][None].float())[0].numpy(); bn = NFL.hp(B_[:, 1][None].float())[0].numpy(); _, Sbp = spec(bp); _, Sbn = spec(bn); _, So = spec(o)
                for k, a in (("before", Sb), ("after", Sa), ("out", So), ("bp", Sbp), ("bn", Sbn)): acc[k].append(nz(a, f) if k not in ("before", "after") else a); peaks[k].append(float(f[np.argmax(a)]))
        M = {k: np.mean(v, 0) for k, v in acc.items()}; share = lambda S, c, w=6: float(S[(f > c - w) & (f < c + w)].sum() / S.sum())
        hit = lambda p, c: float(np.mean(np.abs(np.array(p) - c) <= 6))
        row = dict(set=st, id=sb["id"], face=sb["frames_dir"].split("/")[-1], hr=hr, nuis=nb, kind=kind, gap=abs(hr - nb), chunks=int(len(acc["pub"])), peak_mean={k: float(f[np.argmax(M[k])]) for k in M}, hit_hr={k: hit(peaks[k], hr) for k in peaks}, hit_nuis={k: hit(peaks[k], nb) for k in peaks}, share_before=dict(hr=share(M["before"], hr), nuis=share(M["before"], nb)), share_after=dict(hr=share(M["after"], hr), nuis=share(M["after"], nb)), keep=float(M["after"].sum() / M["before"].sum()))
        R["held_out"][f"{st}/{sb['id']}"] = dict(row, f=f.tolist(), S={k: M[k].tolist() for k in M}); print(st, sb["id"], row["face"][:14], "HR", hr, "N", nb, kind, "| mean peaks", {k: round(v) for k, v in row["peak_mean"].items()}, "| hit_hr out %.2f bp %.2f pub %.2f | hit_nuis bn %.2f before %.2f pub %.2f | share nuis %.2f→%.2f hr %.2f→%.2f" % (row["hit_hr"]["out"], row["hit_hr"]["bp"], row["hit_hr"]["pub"], row["hit_nuis"]["bn"], row["hit_nuis"]["before"], row["hit_nuis"]["pub"], row["share_before"]["nuis"], row["share_after"]["nuis"], row["share_before"]["hr"], row["share_after"]["hr"]), flush=True)
json.dump(R, open(f"{OUT}/{OUTJ}", "w"))
