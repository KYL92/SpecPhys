#!/usr/bin/env python3
"""UBFC 레이아웃 학습 풀(COTRAIN_*)을 uint8 캐시로 1회 전처리 (VIPL 교차검증 셀과 청크 목록 병합용).
사용: RPPG_MP_QUOTA=6 python scripts/pool_preprocess_u8.py COTRAIN_ALLF_TW cotrain_allf_tw_u8
결과: cache/{CACHE_NAME}/subjectN_input*.npy (uint8) + _u8scale.npy + DataFileLists/{CACHE_NAME}_0.0_1.0.csv"""
import os, sys, time
sys.path.insert(0, "/SPECPHYS_ROOT/repos/rPPG-Toolbox")
os.environ["RPPG_CACHE_U8"] = "1"
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"
if __name__ == "__main__":
    pool, cname = sys.argv[1], sys.argv[2]
    os.chdir("/SPECPHYS_ROOT/repos/rPPG-Toolbox")
    from config import _C
    from dataset.data_loader.UBFCrPPGLoader import UBFCrPPGLoader
    cfg = _C.clone(); cfg.defrost(); d = cfg.TRAIN.DATA
    d.FS = 30; d.DATASET = "UBFC-rPPG"; d.DO_PREPROCESS = True; d.DATA_FORMAT = "NCDHW"
    d.DATA_PATH = f"{PFA}/{pool}"; d.CACHED_PATH = f"{PFA}/cache/{cname}"; d.FILE_LIST_PATH = f"{PFA}/cache/DataFileLists/{cname}_0.0_1.0.csv"
    d.EXP_DATA_NAME = cname; d.BEGIN = 0.0; d.END = 1.0
    pp = d.PREPROCESS; pp.DATA_TYPE = ["DiffNormalized"]; pp.DATA_AUG = ["None"]; pp.LABEL_TYPE = "DiffNormalized"; pp.DO_CHUNK = True; pp.CHUNK_LENGTH = 160
    pp.CROP_FACE.DO_CROP_FACE = True; pp.CROP_FACE.BACKEND = "HC"; pp.CROP_FACE.USE_LARGE_FACE_BOX = True; pp.CROP_FACE.LARGE_BOX_COEF = 1.5
    pp.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION = False; pp.RESIZE.H = 128; pp.RESIZE.W = 128
    t0 = time.time()
    L = UBFCrPPGLoader("pool", d.DATA_PATH, d)
    print(f"POOL_PREPROCESS_DONE {cname} chunks={len(L)} sec={time.time()-t0:.0f}")
