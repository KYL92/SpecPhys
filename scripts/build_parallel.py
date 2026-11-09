#!/usr/bin/env python3
"""[65 문서] build_synth_dataset.build_subject 를 피험자 단위로 병렬 실행 (결과는 순차 실행과 같다: 피험자마다 고유 시드).
  cd REPO_ROOT && python scripts/build_parallel.py CONFIG.json [NPROC]"""
import json, os, sys
from multiprocessing import Pool
sys.path.insert(0, "/SPECPHYS_ROOT/generator")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2; cv2.setNumThreads(1)
from build_synth_dataset import build_subject
cfg = json.load(open(sys.argv[1])); n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
os.makedirs(cfg["out"], exist_ok=True)
def one(sub):
    if os.path.exists(os.path.join(cfg["out"], f"subject{sub['id']}", "meta.json")): return sub["id"], "skip"
    return sub["id"], build_subject(sub, cfg.get("fps", 30), cfg.get("size", 256), cfg["out"])
with Pool(n) as p:
    for sid, r in p.imap_unordered(one, cfg["subjects"]): print(f"[par] subject{sid} {r}", flush=True)
print("[par] done", cfg["out"], flush=True)
