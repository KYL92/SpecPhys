#!/usr/bin/env python3
"""VIPL-HR 전체(2,378 VIS 영상) 1회 전처리 → uint8 캐시 cache/vipl_u8 (+ {index}_meta.json 사이드카).
이후 모든 학습/평가는 DO_PREPROCESS=false + 사용자 지정 파일 목록 csv(vipl_filelist.py)로 이 캐시를 공유한다.
사용: RPPG_MP_QUOTA=6 python scripts/vipl_preprocess.py [RAW_ROOT]"""
import os, sys, time
sys.path.insert(0, "/SPECPHYS_ROOT/repos/rPPG-Toolbox")
os.environ["RPPG_CACHE_U8"] = "1"
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"
RAW = sys.argv[1] if len(sys.argv) > 1 else "/SPECPHYS_RAW/vipl_hr_raw"
if __name__ == "__main__":
    os.chdir("/SPECPHYS_ROOT/repos/rPPG-Toolbox")
    from config import _C
    from dataset.data_loader.VIPLHRLoader import VIPLHRLoader
    cfg = _C.clone(); cfg.defrost(); d = cfg.TEST.DATA
    d.FS = 30; d.DATASET = "VIPL-HR"; d.DO_PREPROCESS = True; d.DATA_FORMAT = "NCDHW"
    d.DATA_PATH = RAW; d.CACHED_PATH = f"{PFA}/cache/vipl_u8"; d.FILE_LIST_PATH = f"{PFA}/cache/DataFileLists/vipl_u8_0.0_1.0.csv"
    d.EXP_DATA_NAME = "vipl_u8"; d.BEGIN = 0.0; d.END = 1.0
    pp = d.PREPROCESS; pp.DATA_TYPE = ["DiffNormalized"]; pp.DATA_AUG = ["None"]; pp.LABEL_TYPE = "DiffNormalized"; pp.DO_CHUNK = True; pp.CHUNK_LENGTH = 160
    pp.CROP_FACE.DO_CROP_FACE = True; pp.CROP_FACE.BACKEND = "HC"; pp.CROP_FACE.USE_LARGE_FACE_BOX = True; pp.CROP_FACE.LARGE_BOX_COEF = 1.5
    pp.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION = False; pp.RESIZE.H = 128; pp.RESIZE.W = 128
    t0 = time.time()
    L = VIPLHRLoader("vipl", RAW, d)
    print(f"VIPL_PREPROCESS_DONE chunks={len(L)} sec={time.time()-t0:.0f}")
