#!/usr/bin/env python3
"""[66 §6, 탐색] 분리 시험 세트별 게이트 열림 형태(입력 통계): 세 채널 모두 / 일부 채널만 / 닫힘, 그리고 채널별 κ·q 중앙값.
  cd repos/rPPG-Toolbox && RPPG_CACHE_U8=1 RPPG_SKINW=0.2 RPPG_PARSE_CH=1 NF_BGREF_MODE=gate python ../../scripts/bgref_gate_sets.py TRAIN_YAML → results/bgref_gate_sets.json"""
import argparse, json, os, re, sys
import numpy as np, torch
sys.path.insert(0, os.getcwd()); os.environ.pop("RPPG_TAUG", None)
from config import get_config
from dataset.data_loader.VIPLHRLoader import VIPLHRLoader
from neural_methods.model.FactorizePhys.FactorizePhysNF import BgRef
SETS = {1000: "SEPX24", 1100: "OFFBAND6", 1200: "PMOT6", 1300: "INBAND6", 1400: "NEAR6", 1500: "TEST_C3"}
cfg = get_config(argparse.Namespace(config_file=sys.argv[1])); cfg.defrost(); cfg.TRAIN.DATA.FILE_LIST_PATH = "/SPECPHYS_ROOT/_workspace/data/physformer/cache/DataFileLists/sepeval_u8.csv"; cfg.freeze()
ds = VIPLHRLoader("train", cfg.TRAIN.DATA.DATA_PATH, cfg.TRAIN.DATA); br = BgRef(); res = {}
for i in range(len(ds)):
    nm = os.path.basename(ds.inputs[i]).split("_input")[0]; sid = int(re.sub(r"\D", "", nm)); base = (sid // 100) * 100
    st = SETS[base] + (":G" if (base == 1000 and sid - base <= 12) else ":L" if base == 1000 else "")
    d = torch.from_numpy(np.asarray(ds[i][0]))[None]; x = torch.cat([d, d[:, :, -1:]], 2)
    with torch.no_grad(): br(x[:, :3].float(), x[:, 3:6, 0].float())
    g = br.last_gate[0].numpy(); r = res.setdefault(st, {"n": 0, "all": 0, "partial": 0, "none": 0, "kap": [], "q": []})
    r["n"] += 1; r["all" if g.sum() == 3 else "none" if g.sum() == 0 else "partial"] += 1; r["kap"].append(br.last_kap[0].numpy().tolist()); r["q"].append(br.last_q[0].numpy().tolist())
out = {k: dict(n=v["n"], all=v["all"] / v["n"], partial=v["partial"] / v["n"], none=v["none"] / v["n"], kap_RGB=np.median(v["kap"], 0).round(2).tolist(), q_RGB=np.median(v["q"], 0).round(2).tolist()) for k, v in sorted(res.items())}
for k, v in out.items(): print(k, v, flush=True)
json.dump(out, open("/SPECPHYS_ROOT/_workspace/data/results/bgref_gate_sets.json", "w"), indent=1)
