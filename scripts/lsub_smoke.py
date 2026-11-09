#!/usr/bin/env python3
"""[65 문서] 패치 56 점검(학습 없음): 트레이너의 잡음 영상 변환(_lsub_frames)을 입력 사이드카에 적용한 결과가 로더가 만든 학습 입력과 정확히 같은지, 잡음 영상과 입력의 특징 차이 크기.
  cd repos/rPPG-Toolbox && RPPG_CACHE_U8=1 RPPG_SKINW=0.2 python ../../scripts/lsub_smoke.py"""
import argparse, os, re, sys, types
import numpy as np, torch
sys.path.insert(0, os.getcwd()); os.environ.pop("RPPG_TAUG", None)
from config import get_config
from dataset.data_loader.VIPLHRLoader import VIPLHRLoader
from neural_methods.trainer.FactorizePhysTrainer import FactorizePhysTrainer
Y = sys.argv[1] if len(sys.argv) > 1 else "/SPECPHYS_ROOT/_workspace/data/physformer/yaml_VIPL/finetune_NF_LSUB_SMOKE.yaml"   # 아무 FactorizePhys VIPL 학습 yaml (RESIZE·CHUNK_LENGTH 만 쓴다)
cfg = get_config(argparse.Namespace(config_file=Y)); cfg.defrost(); cfg.TRAIN.DATA.FILE_LIST_PATH = "/SPECPHYS_ROOT/_workspace/data/physformer/cache/DataFileLists/cotrain_allf_hyb4_dr_u8_pool.csv"; cfg.freeze()
ds = VIPLHRLoader("train", cfg.TRAIN.DATA.DATA_PATH, cfg.TRAIN.DATA)
T = types.SimpleNamespace(config=cfg); fr = FactorizePhysTrainer._lsub_frames
idx = [i for i, p in enumerate(ds.inputs) if os.path.exists(p.replace("_input", "_noisevid"))]
print("chunks", len(ds.inputs), "with noise video", len(idx))
mx = 0.0
for i in idx[::25]:
    data, label, name, cid = ds[i]
    a = fr(T, ds.inputs[i], ds.inputs[i])[:, :-1]          # 입력 사이드카에 같은 변환
    mx = max(mx, float(np.abs(a - np.asarray(data)).max()))
print("max |trainer transform(input) - loader data| =", mx)
assert mx == 0.0, "transform mismatch"
print("LSUB_SMOKE_OK")
