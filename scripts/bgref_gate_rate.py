#!/usr/bin/env python3
"""[66 문서 §2.2] 물리 규칙 게이트의 열림 비율(입력 통계만, 오차는 보지 않음): 학습 풀(플리커 있는 이중 렌더 클립 / 실제 노이즈 합성 클립 / 그 밖의 실제·합성 무작위 표본)과 분리 시험 세트.
  cd repos/rPPG-Toolbox && RPPG_CACHE_U8=1 RPPG_SKINW=0.2 RPPG_PARSE_CH=1 NF_BGREF_MODE=gate python ../../scripts/bgref_gate_rate.py TRAIN_YAML → results/bgref_gate_rate.json"""
import argparse, json, os, re, sys
import numpy as np, torch
sys.path.insert(0, os.getcwd()); os.environ.pop("RPPG_TAUG", None)
from config import get_config
from dataset.data_loader.VIPLHRLoader import VIPLHRLoader
from neural_methods.model.FactorizePhys.FactorizePhysNF import BgRef
L = "/SPECPHYS_ROOT/_workspace/data/physformer/cache/DataFileLists"
def grp(name, sep):
    i = int(re.sub(r"\D", "", name) or 0)
    if sep: return "sepx_global" if 1001 <= i <= 1012 else "sepx_skinlocal" if 1013 <= i <= 1024 else "old_stress"
    return "dualrender_flicker" if (261 <= i <= 308 or 2001 <= i <= 2048) else "real_plus_synth_noise" if 301 <= i <= 341 else "real_mmpd" if 700000 <= i < 800000 else "real_pure" if 90000 <= i < 100000 else "other"
br = BgRef(); res = {}; rng = np.random.default_rng(0)
for csv, sep in ((f"{L}/cotrain_allf_hyb4_dr_u8_pool.csv", False), (f"{L}/sepeval_u8.csv", True)):
    cfg = get_config(argparse.Namespace(config_file=sys.argv[1])); cfg.defrost(); cfg.TRAIN.DATA.FILE_LIST_PATH = csv; cfg.freeze()
    ds = VIPLHRLoader("train", cfg.TRAIN.DATA.DATA_PATH, cfg.TRAIN.DATA)
    names = [os.path.basename(p).split("_input")[0] for p in ds.inputs]
    idx = [i for i in range(len(ds)) if sep or grp(names[i], False) in ("dualrender_flicker", "real_plus_synth_noise")]
    if not sep: idx += list(rng.choice([i for i in range(len(ds)) if grp(names[i], False) not in ("dualrender_flicker", "real_plus_synth_noise")], 600, replace=False))
    for i in idx:
        d = torch.from_numpy(np.asarray(ds[i][0]))[None]; x = torch.cat([d, d[:, :, -1:]], 2)
        with torch.no_grad(): br(x[:, :3].float(), x[:, 3:6, 0].float())
        g = grp(names[i], sep); r = res.setdefault(g, {"n": 0, "gate_G": 0, "gate_any": 0, "kapG": [], "qG": []})
        r["n"] += 1; r["gate_G"] += int(br.last_gate[0, 1] > 0); r["gate_any"] += int(br.last_gate[0].max() > 0); r["kapG"].append(float(br.last_kap[0, 1])); r["qG"].append(float(br.last_q[0, 1]))
out = {g: dict(n=r["n"], gate_G=r["gate_G"] / r["n"], gate_any=r["gate_any"] / r["n"], kapG_median=float(np.median(r["kapG"])), qG_median=float(np.median(r["qG"]))) for g, r in res.items()}
for g, v in out.items(): print(g, v, flush=True)
json.dump(out, open("/SPECPHYS_ROOT/_workspace/data/results/bgref_gate_rate.json", "w"), indent=1)
