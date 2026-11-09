#!/usr/bin/env python3
"""18단계: conditional 1-D diffusion waveform refinement head (설계서: _workspace/18_diffusion_refiner_design.md)

subcommands:
  pairs  — TRAINPOOL 덤프 pickle → pulse-space 창 쌍 npz (fundamental-dominant 게이트)
  train  — conditional DDPM 학습(CFG) + 내부 검증으로 guidance w 선택
  apply  — 저장된 평가 pickle → refined pickle (모델명 CoTAF_TW_DIFF_L*)

pulse space: cumsum → detrend(λ=100, sparse) → butter1 0.6–3.3 → 창 z-score. 결정적(시드 고정).
"""
import argparse, glob, hashlib, json, math, os, pickle, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal as ss
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve

FS = 30; L = 300; STRIDE = 150; TSTEPS = 1000
ROOT = "/SPECPHYS_ROOT"
PF = f"{ROOT}/_workspace/data/physformer"
RD = f"{PF}/refiner"
AXES = ["C3", "C6", "E3", "DLCN", "MCD_A", "MCD_B", "MMPD_B1", "MMPD_B2", "MMPD_B3",
        "MMPD_B4", "MMPD_B5", "MMPD_B6", "NIRP_LARGE", "NIRP_SMALL", "NIRP_STILL",
        "PURE", "REALTEST", "RTNC", "RTNI"]

# ---------- 신호 공간 ----------
def fast_detrend(x, lam=100):
    n = len(x); ones = np.ones(n)
    D = sp.spdiags([ones, -2 * ones, ones], [0, 1, 2], n - 2, n)
    A = sp.identity(n) + lam ** 2 * (D.T @ D)
    return x - spsolve(sp.csc_matrix(A), x)

def pulse_space(diffsig):
    x = fast_detrend(np.cumsum(np.asarray(diffsig, float)))
    b, a = ss.butter(1, [0.6 / FS * 2, 3.3 / FS * 2], btype="bandpass")
    return ss.filtfilt(b, a, np.double(x))

def znorm(w):
    return (w - w.mean()) / (w.std() + 1e-8)

def win_hr(w):
    fr = np.fft.rfftfreq(len(w) * 8, 1 / FS)
    pw = np.abs(np.fft.rfft(w - w.mean(), n=len(w) * 8)) ** 2
    band = (fr >= 0.6) & (fr <= 3.3)
    return float(fr[band][pw[band].argmax()] * 60)

def fundamental_ok(w):
    f, p = ss.periodogram(w, FS)
    band = (f >= 0.6) & (f <= 4.0)
    if not band.any() or p[band].max() <= 0:
        return False
    f0 = f[band][p[band].argmax()]
    if not (0.75 <= f0 <= 3.5):
        return False
    p2 = p[np.argmin(np.abs(f - 2 * f0))]
    return p2 <= 0.9 * p[band].max()

def cat_chunks(chunks):
    return np.concatenate([chunks[k].detach().cpu().float().numpy().ravel() for k in sorted(chunks)])

def vidnum(v):
    import re
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else 0

# ---------- 모델 ----------
class Res1D(nn.Module):
    def __init__(self, c, tdim, dil=1):
        super().__init__()
        self.n1 = nn.GroupNorm(8, c); self.c1 = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
        self.n2 = nn.GroupNorm(8, c); self.c2 = nn.Conv1d(c, c, 5, padding=2)
        self.t = nn.Linear(tdim, 2 * c)
    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.t(t).unsqueeze(-1).chunk(2, dim=1)
        h = self.c2(F.silu(self.n2(h) * (1 + sc) + sh))
        return x + h

class UNet1D(nn.Module):
    def __init__(self, base=64, mults=(1, 2, 4), tdim=256):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(128, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        cs = [base * m for m in mults]
        self.inp = nn.Conv1d(2, cs[0], 5, padding=2)
        self.downs = nn.ModuleList(); self.downc = nn.ModuleList()
        for i, c in enumerate(cs):
            self.downs.append(nn.ModuleList([Res1D(c, tdim), Res1D(c, tdim)]))
            self.downc.append(nn.Conv1d(c, cs[min(i + 1, len(cs) - 1)], 4, stride=2, padding=1) if i < len(cs) - 1 else nn.Identity())
        self.mid = nn.ModuleList([Res1D(cs[-1], tdim, dil=2), Res1D(cs[-1], tdim, dil=4)])
        self.ups = nn.ModuleList(); self.upc = nn.ModuleList(); self.fuse = nn.ModuleList()
        for i in reversed(range(len(cs) - 1)):
            self.upc.append(nn.ConvTranspose1d(cs[i + 1], cs[i], 4, stride=2, padding=1))
            self.fuse.append(nn.Conv1d(2 * cs[i], cs[i], 1))
            self.ups.append(nn.ModuleList([Res1D(cs[i], tdim), Res1D(cs[i], tdim)]))
        self.out = nn.Sequential(nn.GroupNorm(8, cs[0]), nn.SiLU(), nn.Conv1d(cs[0], 1, 5, padding=2))
    def time_emb(self, t):
        half = 64
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        ang = t.float()[:, None] * freqs[None]
        return self.tmlp(torch.cat([ang.sin(), ang.cos()], dim=1))
    def forward(self, x, cond, t):
        te = self.time_emb(t)
        h = self.inp(torch.cat([x, cond], dim=1))
        skips = []
        for blocks, dc in zip(self.downs, self.downc):
            for b in blocks: h = b(h, te)
            skips.append(h); h = dc(h)
        for b in self.mid: h = b(h, te)
        return self._up(h, skips, te)
    def _up(self, h, skips, te):
        idx = len(skips) - 2
        for uc, fu, blocks in zip(self.upc, self.fuse, self.ups):
            h = uc(h)
            s = skips[idx]; idx -= 1
            if h.shape[-1] != s.shape[-1]:
                h = F.pad(h, (0, s.shape[-1] - h.shape[-1]))
            h = fu(torch.cat([h, s], dim=1))
            for b in blocks: h = b(h, te)
        return self.out(h)

# ---------- diffusion 스케줄 ----------
def alphabar(t):  # cosine
    return np.cos((t / TSTEPS + 0.008) / 1.008 * math.pi / 2) ** 2

AB = np.array([alphabar(t) for t in range(TSTEPS + 1)]); AB = AB / AB[0]

def ddim_sample(model, cond, w=1.0, steps=50, seed=0, device="cuda"):
    """cond: (B,1,L) 텐서. 반환 (B,1,L)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(cond.shape, generator=g).to(device)
    ts = np.linspace(TSTEPS - 1, 0, steps).round().astype(int)
    zeros = torch.zeros_like(cond)
    with torch.no_grad():
        for i, t in enumerate(ts):
            tt = torch.full((cond.shape[0],), int(t), device=device, dtype=torch.long)
            ec = model(x, cond, tt)
            if abs(w - 1.0) > 1e-6:
                eu = model(x, zeros, tt)
                eps = eu + w * (ec - eu)
            else:
                eps = ec
            ab = AB[t]; ab_prev = AB[int(ts[i + 1])] if i + 1 < len(ts) else 1.0
            x0 = (x - math.sqrt(1 - ab) * eps) / math.sqrt(ab)
            x0 = x0.clamp(-5, 5)
            x = math.sqrt(ab_prev) * x0 + math.sqrt(1 - ab_prev) * eps
    return x

# ---------- pairs ----------
def cmd_pairs(args):
    pk = glob.glob(f"{PF}/runs/ev_CoTAF_TW_L_TRAINPOOL/*/saved_test_outputs/*.pickle")
    assert pk, "TRAINPOOL 덤프 pickle 없음 — 체인 S1 실패?"
    d = pickle.load(open(pk[0], "rb"))
    Xtr, Ytr, Xva, Yva = [], [], [], []
    nvid = 0
    for vid in d["predictions"]:
        p = pulse_space(cat_chunks(d["predictions"][vid]))
        l = pulse_space(cat_chunks(d["labels"][vid]))
        n = min(len(p), len(l))
        if n < L: continue
        nvid += 1
        val = (vidnum(vid) % 10 == 0)
        for st in range(0, n - L + 1, STRIDE):
            tw = l[st:st + L]
            if tw.std() < 1e-6 or not fundamental_ok(tw): continue
            cw = znorm(p[st:st + L]); tw = znorm(tw)
            (Xva if val else Xtr).append(cw.astype(np.float32))
            (Yva if val else Ytr).append(tw.astype(np.float32))
    os.makedirs(RD, exist_ok=True)
    np.savez(f"{RD}/pairs.npz", Xtr=np.array(Xtr), Ytr=np.array(Ytr), Xva=np.array(Xva), Yva=np.array(Yva))
    print(f"PAIRS vids={nvid} train={len(Xtr)} val={len(Xva)}")

# ---------- train ----------
def val_mae(model, Xva, Yva, w, device, steps=25, bs=512):
    errs = []
    for i in range(0, len(Xva), bs):
        c = torch.from_numpy(Xva[i:i + bs]).unsqueeze(1).to(device)
        r = ddim_sample(model, c, w=w, steps=steps, seed=1234 + i, device=device).squeeze(1).cpu().numpy()
        for j in range(len(r)):
            errs.append(abs(win_hr(r[j]) - win_hr(Yva[i + j])))
    return float(np.mean(errs))

def cmd_train(args):
    torch.manual_seed(100); np.random.seed(100)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(f"{RD}/pairs.npz")
    Xtr, Ytr, Xva, Yva = z["Xtr"], z["Ytr"], z["Xva"], z["Yva"]
    base = float(np.mean([abs(win_hr(x) - win_hr(y)) for x, y in zip(Xva, Yva)]))
    print(f"train={len(Xtr)} val={len(Xva)} | cond baseline val HR MAE={base:.3f}")
    model = UNet1D().to(device)
    ema = UNet1D().to(device); ema.load_state_dict(model.state_dict())
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    nstep = args.steps; bs = 256; best = (1e9, None)
    for step in range(1, nstep + 1):
        idx = np.random.randint(0, len(Xtr), bs)
        x0 = torch.from_numpy(Ytr[idx]).unsqueeze(1).to(device)
        cond = torch.from_numpy(Xtr[idx]).unsqueeze(1).to(device)
        drop = torch.rand(bs, device=device) < 0.1
        cond = cond * (~drop)[:, None, None]
        t = torch.randint(0, TSTEPS, (bs,), device=device)
        ab = torch.from_numpy(AB[t.cpu().numpy()]).float().to(device)[:, None, None]
        eps = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        loss = F.mse_loss(model(xt, cond, t), eps)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.999).add_(pm, alpha=0.001)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        if step % 2000 == 0 or step == nstep:
            sub = min(len(Xva), 768)
            m = val_mae(ema, Xva[:sub], Yva[:sub], 1.0, device)
            print(f"step {step} loss {loss.item():.4f} valMAE(w=1) {m:.3f}", flush=True)
            if m < best[0]:
                best = (m, {k: v.cpu().clone() for k, v in ema.state_dict().items()})
    ema.load_state_dict(best[1])
    ws = {}
    for w in (1.0, 1.5, 2.0):
        ws[w] = val_mae(ema, Xva, Yva, w, device)
        print(f"guidance w={w} valMAE={ws[w]:.3f}")
    wbest = min(ws, key=ws.get)
    os.makedirs(RD, exist_ok=True)
    torch.save(best[1], f"{RD}/ckpt.pt")
    json.dump(dict(w=wbest, val_mae=ws[wbest], val_sweep=ws, cond_baseline=base,
                   best_step_val=best[0]), open(f"{RD}/meta.json", "w"), indent=1)
    print(f"TRAIN_DONE w*={wbest} valMAE={ws[wbest]:.3f} (baseline {base:.3f})")

# ---------- apply ----------
def refine_full(model, sig, w, seed, device):
    n = len(sig)
    if n < 40: return sig.copy()
    x = pulse_space(sig)
    pad = max(0, L - n)
    if pad: x = np.pad(x, (0, pad), mode="reflect")
    starts = list(range(0, len(x) - L + 1, STRIDE))
    if starts[-1] != len(x) - L: starts.append(len(x) - L)
    wins = np.stack([znorm(x[s:s + L]) for s in starts]).astype(np.float32)
    c = torch.from_numpy(wins).unsqueeze(1).to(device)
    r = ddim_sample(model, c, w=w, steps=50, seed=seed, device=device).squeeze(1).cpu().numpy()
    hann = np.hanning(L) + 1e-3
    acc = np.zeros(len(x)); den = np.zeros(len(x))
    for s, rw in zip(starts, r):
        acc[s:s + L] += rw * hann; den[s:s + L] += hann
    out = (acc / den)[:n]
    return np.diff(out, prepend=out[:1])

def cmd_apply(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(f"{RD}/meta.json")); w = float(meta["w"])
    model = UNet1D().to(device)
    model.load_state_dict(torch.load(f"{RD}/ckpt.pt", map_location=device))
    model.eval()
    src, dst = args.src, args.dst
    done = 0
    for ax in AXES:
        pks = glob.glob(f"{PF}/runs/ev_{src}_{ax}/*/saved_test_outputs/*.pickle")
        if not pks:
            print(f"skip {ax} (no pickle)"); continue
        d = pickle.load(open(pks[0], "rb"))
        newp = {}
        for vid in d["predictions"]:
            ch = d["predictions"][vid]
            sig = cat_chunks(ch)
            seed = int(hashlib.md5(f"{vid}".encode()).hexdigest()[:8], 16) % (2 ** 31)
            ref = refine_full(model, sig, w, seed, device)
            nd = {}; off = 0
            for k in sorted(ch):
                ln = ch[k].numel()
                nd[k] = torch.from_numpy(ref[off:off + ln].astype(np.float32))
                off += ln
            newp[vid] = nd
        od = f"{PF}/runs/ev_{dst}_{ax}/refined/saved_test_outputs"
        os.makedirs(od, exist_ok=True)
        pickle.dump(dict(predictions=newp, labels=d["labels"], label_type=d.get("label_type"), fs=d.get("fs", FS)),
                    open(f"{od}/refined_outputs.pickle", "wb"))
        done += 1
        print(f"APPLY {src}->{dst} {ax} vids={len(newp)}", flush=True)
    print(f"APPLY_DONE {dst} axes={done}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pairs")
    tp = sub.add_parser("train"); tp.add_argument("--steps", type=int, default=20000)
    app = sub.add_parser("apply"); app.add_argument("--src", required=True); app.add_argument("--dst", required=True)
    a = ap.parse_args()
    {"pairs": cmd_pairs, "train": cmd_train, "apply": cmd_apply}[a.cmd](a)
