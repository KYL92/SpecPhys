#!/usr/bin/env python3
"""18단계 확장(체인 32): diffusion 기반 변형 모델 스윕. 기반 모듈(diffusion_refiner.py)은 수정하지 않고 import.

variants (tag → 저장: refiner/ckpt_<tag>.pt, meta_<tag>.json):
  REG  — 동일 U-Net 직접 회귀(cond→x0, diffusion 없음): "diffusion 과정 자체가 필요한가" ablation
  X0   — x0-parameterization DDPM (ε 대신 x0 예측)
  DIT  — DiT-1D transformer denoiser (patchify + AdaLN-zero)
  W600 — 20초 장창(L=600) ε-U-Net: harmonic 모호성에 대한 창 길이 효과

사용: pairs --win 600 | train --tag T --mode {ddpm,reg} --arch {unet,dit} --param {eps,x0} --win N | apply --tag T --src M --dst M2
"""
import argparse, glob, hashlib, importlib.util, json, math, os, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = "/SPECPHYS_ROOT"
spec = importlib.util.spec_from_file_location("dr", f"{ROOT}/scripts/diffusion_refiner.py")
dr = importlib.util.module_from_spec(spec); spec.loader.exec_module(dr)
PF, RD, AXES, AB, TSTEPS, FS = dr.PF, dr.RD, dr.AXES, dr.AB, dr.TSTEPS, dr.FS

# ---------- DiT-1D ----------
class DiTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)
    def forward(self, h, te):
        sa, ha, ga, sm, hm, gm = self.ada(te).unsqueeze(1).chunk(6, dim=-1)
        x = self.ln1(h) * (1 + sa) + ha
        h = h + ga * self.attn(x, x, x, need_weights=False)[0]
        x = self.ln2(h) * (1 + sm) + hm
        return h + gm * self.mlp(x)

class DiT1D(nn.Module):
    def __init__(self, L=300, patch=10, dim=256, depth=6, heads=8):
        super().__init__()
        assert L % patch == 0
        self.patch = patch; self.ntok = L // patch
        self.embed = nn.Linear(2 * patch, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.ntok, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.tmlp = nn.Sequential(nn.Linear(128, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.lnf = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaf = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.adaf.weight); nn.init.zeros_(self.adaf.bias)
        self.out = nn.Linear(dim, patch)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def time_emb(self, t):
        half = 64
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        ang = t.float()[:, None] * freqs[None]
        return self.tmlp(torch.cat([ang.sin(), ang.cos()], dim=1))
    def forward(self, x, cond, t):
        B = x.shape[0]
        h = torch.cat([x, cond], dim=1)                       # (B,2,L)
        h = h.reshape(B, 2, self.ntok, self.patch).permute(0, 2, 1, 3).reshape(B, self.ntok, 2 * self.patch)
        h = self.embed(h) + self.pos
        te = self.time_emb(t)
        for b in self.blocks: h = b(h, te)
        sf, hf = self.adaf(te).unsqueeze(1).chunk(2, dim=-1)
        h = self.out(self.lnf(h) * (1 + sf) + hf)             # (B,ntok,patch)
        return h.reshape(B, 1, self.ntok * self.patch)

def build_model(arch, L):
    return DiT1D(L=L) if arch == "dit" else dr.UNet1D()

# ---------- 샘플러 (param: eps|x0, CFG) ----------
def sample(model, cond, w=1.0, steps=50, seed=0, device="cuda", param="eps", mode="ddpm"):
    if mode == "reg":
        with torch.no_grad():
            t0 = torch.zeros(cond.shape[0], device=device, dtype=torch.long)
            return model(torch.zeros_like(cond), cond, t0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(cond.shape, generator=g).to(device)
    ts = np.linspace(TSTEPS - 1, 0, steps).round().astype(int)
    zeros = torch.zeros_like(cond)
    with torch.no_grad():
        for i, t in enumerate(ts):
            tt = torch.full((cond.shape[0],), int(t), device=device, dtype=torch.long)
            oc = model(x, cond, tt)
            o = oc
            if abs(w - 1.0) > 1e-6:
                ou = model(x, zeros, tt)
                o = ou + w * (oc - ou)
            ab = AB[t]; ab_prev = AB[int(ts[i + 1])] if i + 1 < len(ts) else 1.0
            if param == "eps":
                x0 = (x - math.sqrt(1 - ab) * o) / math.sqrt(ab)
                eps = o
            else:
                x0 = o
                eps = (x - math.sqrt(ab) * x0) / max(math.sqrt(1 - ab), 1e-8)
            x0 = x0.clamp(-5, 5)
            x = math.sqrt(ab_prev) * x0 + math.sqrt(1 - ab_prev) * eps
    return x

# ---------- pairs (창 길이 파라미터화) ----------
def cmd_pairs(a):
    L = a.win; ST = L // 2
    pk = glob.glob(f"{PF}/runs/ev_CoTAF_TW_L_TRAINPOOL/*/saved_test_outputs/*.pickle")
    assert pk, "TRAINPOOL 덤프 pickle 없음"
    d = pickle.load(open(pk[0], "rb"))
    Xtr, Ytr, Xva, Yva = [], [], [], []
    for vid in d["predictions"]:
        p = dr.pulse_space(dr.cat_chunks(d["predictions"][vid]))
        l = dr.pulse_space(dr.cat_chunks(d["labels"][vid]))
        n = min(len(p), len(l))
        if n < L: continue
        val = (dr.vidnum(vid) % 10 == 0)
        for st in range(0, n - L + 1, ST):
            tw = l[st:st + L]
            if tw.std() < 1e-6 or not dr.fundamental_ok(tw): continue
            (Xva if val else Xtr).append(dr.znorm(p[st:st + L]).astype(np.float32))
            (Yva if val else Ytr).append(dr.znorm(tw).astype(np.float32))
    os.makedirs(RD, exist_ok=True)
    out = f"{RD}/pairs{L}.npz" if L != 300 else f"{RD}/pairs.npz"
    np.savez(out, Xtr=np.array(Xtr), Ytr=np.array(Ytr), Xva=np.array(Xva), Yva=np.array(Yva))
    print(f"PAIRS win={L} train={len(Xtr)} val={len(Xva)} -> {out}")

# ---------- train ----------
def val_mae(model, Xva, Yva, w, device, param, mode, steps=25, bs=512):
    errs = []
    for i in range(0, len(Xva), bs):
        c = torch.from_numpy(Xva[i:i + bs]).unsqueeze(1).to(device)
        r = sample(model, c, w=w, steps=steps, seed=1234 + i, device=device, param=param, mode=mode).squeeze(1).cpu().numpy()
        errs += [abs(dr.win_hr(r[j]) - dr.win_hr(Yva[i + j])) for j in range(len(r))]
    return float(np.mean(errs))

def cmd_train(a):
    torch.manual_seed(100); np.random.seed(100)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pf = f"{RD}/pairs{a.win}.npz" if a.win != 300 else f"{RD}/pairs.npz"
    z = np.load(pf); Xtr, Ytr, Xva, Yva = z["Xtr"], z["Ytr"], z["Xva"], z["Yva"]
    base = float(np.mean([abs(dr.win_hr(x) - dr.win_hr(y)) for x, y in zip(Xva, Yva)]))
    print(f"[{a.tag}] mode={a.mode} arch={a.arch} param={a.param} win={a.win} train={len(Xtr)} val={len(Xva)} baseMAE={base:.3f}", flush=True)
    model = build_model(a.arch, a.win).to(device)
    ema = build_model(a.arch, a.win).to(device); ema.load_state_dict(model.state_dict())
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    bs = 256 if a.win == 300 else 128
    best = (1e9, None)
    for step in range(1, a.steps + 1):
        idx = np.random.randint(0, len(Xtr), bs)
        x0 = torch.from_numpy(Ytr[idx]).unsqueeze(1).to(device)
        cond = torch.from_numpy(Xtr[idx]).unsqueeze(1).to(device)
        if a.mode == "reg":
            t0 = torch.zeros(bs, device=device, dtype=torch.long)
            loss = F.mse_loss(model(torch.zeros_like(cond), cond, t0), x0)
        else:
            drop = torch.rand(bs, device=device) < 0.1
            cond = cond * (~drop)[:, None, None]
            t = torch.randint(0, TSTEPS, (bs,), device=device)
            ab = torch.from_numpy(AB[t.cpu().numpy()]).float().to(device)[:, None, None]
            eps = torch.randn_like(x0)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            tgt = eps if a.param == "eps" else x0
            loss = F.mse_loss(model(xt, cond, t), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.999).add_(pm, alpha=0.001)
        if step % 2000 == 0 or step == a.steps:
            sub = min(len(Xva), 768)
            m = val_mae(ema, Xva[:sub], Yva[:sub], 1.0, device, a.param, a.mode)
            print(f"[{a.tag}] step {step} loss {loss.item():.4f} valMAE(w=1) {m:.3f}", flush=True)
            if m < best[0]:
                best = (m, {k: v.cpu().clone() for k, v in ema.state_dict().items()})
    ema.load_state_dict(best[1])
    if a.mode == "reg":
        ws = {1.0: val_mae(ema, Xva, Yva, 1.0, device, a.param, a.mode)}
    else:
        ws = {w: val_mae(ema, Xva, Yva, w, device, a.param, a.mode) for w in (1.0, 1.5, 2.0)}
    for w, v in ws.items(): print(f"[{a.tag}] guidance w={w} valMAE={v:.3f}")
    wbest = min(ws, key=ws.get)
    torch.save(best[1], f"{RD}/ckpt_{a.tag}.pt")
    json.dump(dict(tag=a.tag, mode=a.mode, arch=a.arch, param=a.param, win=a.win, w=wbest,
                   val_mae=ws[wbest], val_sweep={str(k): v for k, v in ws.items()}, cond_baseline=base),
              open(f"{RD}/meta_{a.tag}.json", "w"), indent=1)
    print(f"TRAIN_DONE [{a.tag}] w*={wbest} valMAE={ws[wbest]:.3f} (baseline {base:.3f})")

# ---------- apply ----------
def refine_full(model, sig, w, seed, device, L, param, mode):
    n = len(sig)
    if n < 40: return sig.copy()
    x = dr.pulse_space(sig)
    if len(x) < L: x = np.pad(x, (0, L - len(x)), mode="reflect")
    ST = L // 2
    starts = list(range(0, len(x) - L + 1, ST))
    if starts[-1] != len(x) - L: starts.append(len(x) - L)
    wins = np.stack([dr.znorm(x[s:s + L]) for s in starts]).astype(np.float32)
    c = torch.from_numpy(wins).unsqueeze(1).to(device)
    r = sample(model, c, w=w, steps=50, seed=seed, device=device, param=param, mode=mode).squeeze(1).cpu().numpy()
    hann = np.hanning(L) + 1e-3
    acc = np.zeros(len(x)); den = np.zeros(len(x))
    for s, rw in zip(starts, r):
        acc[s:s + L] += rw * hann; den[s:s + L] += hann
    out = (acc / den)[:n]
    return np.diff(out, prepend=out[:1])

def cmd_apply(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(f"{RD}/meta_{a.tag}.json"))
    L, w, param, mode = int(meta["win"]), float(meta["w"]), meta["param"], meta["mode"]
    model = build_model(meta["arch"], L).to(device)
    model.load_state_dict(torch.load(f"{RD}/ckpt_{a.tag}.pt", map_location=device)); model.eval()
    for ax in AXES:
        pks = glob.glob(f"{PF}/runs/ev_{a.src}_{ax}/*/saved_test_outputs/*.pickle")
        if not pks: print(f"skip {ax}"); continue
        d = pickle.load(open(pks[0], "rb"))
        newp = {}
        for vid in d["predictions"]:
            ch = d["predictions"][vid]
            sig = dr.cat_chunks(ch)
            seed = int(hashlib.md5(f"{vid}".encode()).hexdigest()[:8], 16) % (2 ** 31)
            ref = refine_full(model, sig, w, seed, device, L, param, mode)
            nd = {}; off = 0
            for k in sorted(ch):
                ln = ch[k].numel()
                nd[k] = torch.from_numpy(ref[off:off + ln].astype(np.float32)); off += ln
            newp[vid] = nd
        od = f"{PF}/runs/ev_{a.dst}_{ax}/refined/saved_test_outputs"
        os.makedirs(od, exist_ok=True)
        pickle.dump(dict(predictions=newp, labels=d["labels"], label_type=d.get("label_type"), fs=d.get("fs", FS)),
                    open(f"{od}/refined_outputs.pickle", "wb"))
        print(f"APPLY[{a.tag}] {a.src}->{a.dst} {ax} vids={len(newp)}", flush=True)
    print(f"APPLY_DONE {a.dst}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("pairs"); pp.add_argument("--win", type=int, default=300)
    tp = sub.add_parser("train")
    tp.add_argument("--tag", required=True); tp.add_argument("--mode", default="ddpm", choices=["ddpm", "reg"])
    tp.add_argument("--arch", default="unet", choices=["unet", "dit"]); tp.add_argument("--param", default="eps", choices=["eps", "x0"])
    tp.add_argument("--win", type=int, default=300); tp.add_argument("--steps", type=int, default=20000)
    app = sub.add_parser("apply"); app.add_argument("--tag", required=True)
    app.add_argument("--src", required=True); app.add_argument("--dst", required=True)
    a = ap.parse_args()
    {"pairs": cmd_pairs, "train": cmd_train, "apply": cmd_apply}[a.cmd](a)
