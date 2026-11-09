#!/usr/bin/env python3
"""[67 문서] 최종 풀의 생성 얼굴 학습 클립 80개를 '깨끗한 배경' 판으로 다시 만드는 설정: 원래 설정(같은 시드·같은 파형·같은 잡음) + skin_mask_parser. 출력 폴더 끝 _C.
결정성 점검용으로 세트마다 첫 피험자만 원래 설정 그대로(_DETCHK 폴더) 다시 만드는 설정도 쓴다.
  python scripts/make_clean_train_configs.py"""
import json, copy
C = "/SPECPHYS_ROOT/_workspace/data/physformer/configs"
for c in ("train20h_tw", "train6hi_tw", "train6lo_tw", "train24dn", "train24dn2"):
    d = json.load(open(f"{C}/{c}.json")); dc = copy.deepcopy(d); dc["out"] = d["out"].rstrip("/") + "_C"
    for s in dc["subjects"]: s["skin_mask_parser"] = True
    json.dump(dc, open(f"{C}/{c}_c.json", "w"), indent=1)
    dd = copy.deepcopy(d); dd["out"] = d["out"].rstrip("/") + "_DETCHK"; dd["subjects"] = dd["subjects"][:1]
    json.dump(dd, open(f"{C}/{c}_detchk.json", "w"), indent=1); print(c, "->", dc["out"], "| detchk", dd["subjects"][0]["id"])
