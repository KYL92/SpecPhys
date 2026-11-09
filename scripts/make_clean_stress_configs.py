#!/usr/bin/env python3
"""[66 문서 §7, 탐색] 분리 시험 클립의 '깨끗한 배경' 판: 기존 설정(같은 시드·같은 파형·같은 잡음) + skin_mask_parser(맥파 마스크를 얼굴 피부에 묶음). 출력 폴더 이름 끝에 _CLEAN.
  python scripts/make_clean_stress_configs.py"""
import json, copy
C = "/SPECPHYS_ROOT/_workspace/data/physformer/configs"
for c in ("stress_sepx24", "stress_offband6", "stress_inband6", "stress_near6", "stress_pmot6", "test_C3"):
    d = copy.deepcopy(json.load(open(f"{C}/{c}.json"))); d["out"] = d["out"].rstrip("/") + "_CLEAN"
    for s in d["subjects"]: s["skin_mask_parser"] = True
    json.dump(d, open(f"{C}/{c}_clean.json", "w"), indent=1); print(c, "->", d["out"])
