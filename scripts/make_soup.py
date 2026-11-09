#!/usr/bin/env python3
"""[체인 45 / 트랙 B2] 하이브리드 분해 모델 CoTAM_HYB4_L 5시드 Epoch9 가중치의 균등 평균(uniform soup).
- 키 집합·shape·dtype 일치 검증, 부동소수 텐서(BatchNorm running_mean/var 포함)는 평균,
  정수 버퍼(num_batches_tracked)는 반올림 평균.
- 저장: runs/finetune_<OUT>/PreTrainedModels/SYNTH_<OUT>_physformer_Epoch9.pth (DataParallel 'module.' 접두 유지)
사용: python make_soup.py [BASE=CoTAM_HYB4_L] [OUT=CoTAM_HYB4_SOUP_L]
"""
import os, sys, json
import torch

PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
BASE = sys.argv[1] if len(sys.argv) > 1 else "CoTAM_HYB4_L"
OUT = sys.argv[2] if len(sys.argv) > 2 else "CoTAM_HYB4_SOUP_L"
SEEDS = ("", "_s200", "_s300", "_s400", "_s500")

paths = [f"{PF}/runs/finetune_{BASE}{s}/PreTrainedModels/SYNTH_{BASE}{s}_physformer_Epoch9.pth" for s in SEEDS]
sds = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
# 키 합집합: s400/s500은 체인 41 이후 코드(conf_head 생성)로 학습되어 conf_head 키(학습 미사용, 무작위 초기화)를 추가로 가짐.
# 공통 키는 5시드 평균, 일부 시드에만 있는 키는 보유 시드만 평균(현 코드의 strict 로드에 필요) — 보고서에 명기.
keys = []
for sd in sds:
    for k in sd:
        if k not in keys: keys.append(k)
partial = {k: [i for i, sd in enumerate(sds) if k in sd] for k in keys if any(k not in sd for sd in sds)}
for k in keys:
    ref = next(sd[k] for sd in sds if k in sd)
    for p, sd in zip(paths, sds):
        if k in sd:
            assert sd[k].shape == ref.shape and sd[k].dtype == ref.dtype, f"shape/dtype 불일치 {k} in {p}"

soup, info = {}, dict(n_keys=len(keys), n_float=0, n_int=0, noise_head_keys=0, max_seed_spread=0.0)
for k in keys:
    ts = [sd[k] for sd in sds if k in sd]
    if ts[0].is_floating_point():
        st = torch.stack([t.float() for t in ts])
        soup[k] = st.mean(0).to(ts[0].dtype)
        info["n_float"] += 1
        info["max_seed_spread"] = max(info["max_seed_spread"], float((st.max(0).values - st.min(0).values).max()))
    else:
        soup[k] = torch.round(torch.stack([t.double() for t in ts]).mean(0)).to(ts[0].dtype)
        info["n_int"] += 1
    if "_n." in k or "conf_head" in k:
        info["noise_head_keys"] += 1

od = f"{PF}/runs/finetune_{OUT}/PreTrainedModels"
os.makedirs(od, exist_ok=True)
op = f"{od}/SYNTH_{OUT}_physformer_Epoch9.pth"
torch.save(soup, op)
info["out"] = op; info["sources"] = paths; info["partial_keys"] = partial
print(json.dumps(info, indent=1, ensure_ascii=False))
print("SOUP_DONE")
