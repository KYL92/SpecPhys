#!/usr/bin/env python3
"""[66 문서] 패치 57 점검(학습 없음): RPPG_PARSE_CH=1 일 때 학습 풀(u8)과 평가 축(float Raw 캐시) 표본이 6채널이고, 부위 채널이 사이드카를 늘린 값과 같은지.
  cd repos/rPPG-Toolbox && RPPG_CACHE_U8=1 RPPG_SKINW=0.2 RPPG_PARSE_CH=1 python ../../scripts/bgref_smoke.py TRAIN_YAML EV_YAML"""
import argparse, os, sys
import cv2, numpy as np
sys.path.insert(0, os.getcwd()); os.environ.pop("RPPG_TAUG", None)
from config import get_config
from dataset.data_loader.VIPLHRLoader import VIPLHRLoader
from dataset.data_loader.UBFCrPPGLoader import UBFCrPPGLoader
ty, ey = sys.argv[1], sys.argv[2]
cfg = get_config(argparse.Namespace(config_file=ty)); cfg.defrost(); cfg.TRAIN.DATA.FILE_LIST_PATH = "/SPECPHYS_ROOT/_workspace/data/physformer/cache/DataFileLists/cotrain_allf_hyb4_dr_u8_pool.csv"; cfg.freeze()
ds = VIPLHRLoader("train", cfg.TRAIN.DATA.DATA_PATH, cfg.TRAIN.DATA)
for i in (0, len(ds.inputs) // 2, len(ds.inputs) - 1):
    d, lab, nm, c = ds[i]; P = np.load(ds.inputs[i].replace("_input", "_parse")).astype(np.float32) / 255.0
    ref = np.stack([cv2.resize(P[k], (72, 72), interpolation=cv2.INTER_LINEAR) for k in range(3)], 0)
    print("train", nm, c, d.shape, "parse ch max|diff| =", float(np.abs(d[3:, 0] - ref).max()), "constant over time:", bool(np.allclose(d[3:, 0], d[3:, -1])))
ce = get_config(argparse.Namespace(config_file=ey))
dt = UBFCrPPGLoader("test", ce.TEST.DATA.DATA_PATH, ce.TEST.DATA)
d, lab, nm, c = dt[0]; print("eval", ce.TEST.DATA.EXP_DATA_NAME, nm, c, d.shape, "parse ch range", float(d[3:].min()), float(d[3:].max()))
assert d.shape[0] == 6
print("BGREF_SMOKE_OK")
