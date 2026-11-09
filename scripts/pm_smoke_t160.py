#!/usr/bin/env python3
"""[체인 74, 57 문서] PhysMamba 가 160 프레임 청크를 받는지(시간 축 완전 합성곱 여부) + 공개/풀 체크포인트 적재 확인. pro6000 컨테이너 안에서 실행:
  /SPECPHYS_CONDA/envs/vface-rppg/bin/python scripts/pm_smoke_t160.py"""
import os, sys, torch
sys.path.insert(0, "/SPECPHYS_ROOT/repos/rPPG-Toolbox")
from neural_methods.model.PhysMamba import PhysMamba
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; TB = "/SPECPHYS_ROOT/repos/rPPG-Toolbox"; dev = "cuda"


def load(m, p):
    sd = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return m.load_state_dict(sd, strict=False)


m = PhysMamba(frames=128).to(dev)
r = load(m, f"{TB}/final_model_release/UBFC-rPPG_PhysMamba_DiffNormalized.pth"); print("public ckpt missing/unexpected:", len(r.missing_keys), len(r.unexpected_keys))
for T in (128, 160):
    m = PhysMamba(frames=T).to(dev); load(m, f"{TB}/final_model_release/UBFC-rPPG_PhysMamba_DiffNormalized.pth"); x = torch.randn(2, 3, T, 128, 128, device=dev); m.eval()
    with torch.no_grad(): y = m(x)
    print(f"T={T} eval out {tuple(y.shape)}")
    m.train(); y = m(x); y.pow(2).mean().backward(); print(f"T={T} train out {tuple(y.shape)} backward ok, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"); m.zero_grad()
os.environ["RPPG_NOISE_HEAD"] = "1"; m2 = PhysMamba(frames=160).to(dev)
r = load(m2, f"{PFA}/runs/finetune_PM_HYB4_L/PreTrainedModels/SYNTH_PM_HYB4_L_physformer_Epoch9.pth"); print("PM_HYB4_L ckpt missing/unexpected:", len(r.missing_keys), len(r.unexpected_keys))
with torch.no_grad(): y = m2.eval()(torch.randn(1, 3, 160, 128, 128, device=dev)); print("HYB4 T=160 out", tuple(y.shape))
print("PM_SMOKE_OK")
