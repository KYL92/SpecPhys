#!/usr/bin/env python3
"""[61 문서 §9.3, 2026-11-04] MMPD 공통 정답(판정 스크립트 공용).
공개 가중치·이전 모델의 MMPD 집계는 다른 전처리 파이프라인에서 나와 영상별 라벨 심박이 우리 u8 파이프라인과 다르다
(공개 RhythmFormer 9.1 % 영상에서 >1 bpm, 4편 >20 bpm; 공개 FactorizePhys 2.6 %). 영상 단위 쌍대 비교는 같은 정답으로 채점해야 하므로,
모든 모델의 MMPD 행을 우리 셀 라벨 심박의 영상별 중앙값 MG(셀·시드 간 차이 0; vipl_wave_gt_judge 의 MMPD 정답과 같은 정의)로 다시 채점한다.
  import mmpd_common_gt as MGC; rows = MGC.load(path)   # gt ← MG, err ← |pred − MG| (MG 에 없는 영상은 그대로)"""
import glob, json, os, re
import numpy as np
R = "/SPECPHYS_ROOT/_workspace/data/results"
def _files(m, suf):
    pat = re.compile(rf"agg_{re.escape(m)}(_s\d+)?{suf}\.json$")
    return [f for f in [f"{R}/agg_{m}{suf}.json"] + sorted(glob.glob(f"{R}/agg_{m}_s[0-9]*{suf}.json")) if os.path.exists(f) and pat.fullmatch(os.path.basename(f))]
def key(r): return (r["subject"], r["light"], r["motion"], r["exercise"], r["skin"])
_acc = {}
for _m in ("NF_HYB_D_L", "RF_SPEC_L", "NF_HYB_U8_L"):
    for _f in _files(_m, "_MMPD"):
        for _r in json.load(open(_f)):
            if 7 <= _r["subject"] <= 33: _acc.setdefault(key(_r), []).append(_r["gt"])
MG = {k: float(np.median(v)) for k, v in _acc.items()}
def fix(r):
    k = key(r)
    if k in MG and "pred" in r: return dict(r, gt=MG[k], err=abs(float(r["pred"]) - MG[k]), gt_own=r.get("gt"))
    return r
def load(path):
    d = json.load(open(path)); rows = d.get("rows", d) if isinstance(d, dict) else d
    return [fix(r) for r in rows]
