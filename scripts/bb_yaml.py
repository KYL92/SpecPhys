#!/usr/bin/env python3
"""[체인 74, 57 문서] 다른 백본(RhythmFormer·PhysMamba)을 같은 하이브리드 풀로 학습·평가하기 위한 yaml 생성기 (fp_yaml.py 의 백본 일반화).
  python scripts/bb_yaml.py tpl BACKBONE [--lr 1e-4] [--epochs 10]
        → yaml_BB/tpl_{BACKBONE}.yaml : 툴박스 공식 학습 설정(UBFC-rPPG 체크포인트 RESUME, 마지막 epoch 사용, 검증 없음) — vipl_yaml.py train/ev --template 용 틀
  python scripts/bb_yaml.py ev_axis NAME BACKBONE CKPT AXIS [--preprocess 0|1]
        → yaml_{NAME}/ev_{NAME}_{AXIS}.yaml : yaml_CoTAM_HYB4_L/ev_CoTAM_HYB4_L_{AXIS}.yaml 의 DATA(경로·데이터셋)에 백본 전처리(해상도·청크·타입)를 덮어씀, 캐시 {prefix}_{axis}
백본 사양: RhythmFormer = Standardized 입력/라벨, 128×128, 청크 160, NDCHW, DROP 0.2, 공개 가중치 UBFC-rPPG_RhythmFormer.pth
          PhysMamba    = DiffNormalized 입력/라벨, 128×128, 청크 160(상류 128; 패치 42 frames=청크 길이로 160 프레임 u8 캐시 공용), NCDHW, DROP 0.1, 공개 가중치 UBFC-rPPG_PhysMamba_DiffNormalized.pth"""
import os, sys, yaml

PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; TB = "/SPECPHYS_ROOT/repos/rPPG-Toolbox"
BB = {
    "RhythmFormer": dict(cfg="UBFC-rPPG_UBFC-rPPG_PURE_RHYTHMFORMER_BASIC.yaml", dtype="Standardized", ltype="Standardized", chunk=160, fmt="NDCHW", drop=0.2, resume="UBFC-rPPG_RhythmFormer.pth", prefix="rf"),
    "PhysMamba": dict(cfg="UBFC-rPPG_UBFC-rPPG_PURE_PHYSMAMBA.yaml", dtype="DiffNormalized", ltype="DiffNormalized", chunk=160, fmt="NCDHW", drop=0.1, resume="UBFC-rPPG_PhysMamba_DiffNormalized.pth", prefix="pm"),
}


def overlay(sec, b, do_pre, cache_name):
    d = sec["DATA"]; pre = d["PREPROCESS"]
    pre["DATA_TYPE"] = [b["dtype"]]; pre["LABEL_TYPE"] = b["ltype"]; pre["RESIZE"] = {"H": 128, "W": 128}; pre["CHUNK_LENGTH"] = b["chunk"]; pre["DO_CHUNK"] = True
    d["DATA_FORMAT"] = b["fmt"]; d["DO_PREPROCESS"] = bool(do_pre); d["EXP_DATA_NAME"] = cache_name
    if d["DO_PREPROCESS"]:
        lst = f"{PFA}/cache/DataFileLists/{cache_name}_{float(d.get('BEGIN', 0.0))}_{float(d.get('END', 1.0))}.csv"
        if os.path.exists(lst) and os.path.isdir(f"{PFA}/cache/{cache_name}") and os.listdir(f"{PFA}/cache/{cache_name}"):
            d["DO_PREPROCESS"] = False; print(f"  [bb_yaml] cache reuse {cache_name}", file=sys.stderr)


def dump(c, p):
    os.makedirs(os.path.dirname(p), exist_ok=True); yaml.safe_dump(c, open(p, "w"), sort_keys=False, default_flow_style=False); print(p)


if __name__ == "__main__":
    mode = sys.argv[1]; a = sys.argv[2:]
    if mode == "tpl":
        lr = 1e-4; ep = 10
        if "--lr" in a: i = a.index("--lr"); lr = float(a[i + 1]); del a[i:i + 2]
        if "--epochs" in a: i = a.index("--epochs"); ep = int(a[i + 1]); del a[i:i + 2]
        bname = a[0]; b = BB[bname]; c = yaml.safe_load(open(f"{TB}/configs/train_configs/{b['cfg']}"))
        c["TOOLBOX_MODE"] = "train_and_test"; c["TRAIN"]["LR"] = lr; c["TRAIN"]["EPOCHS"] = ep; c["TRAIN"]["BATCH_SIZE"] = 4; c["TRAIN"]["PLOT_LOSSES_AND_LR"] = False
        c["MODEL"] = {"NAME": bname, "DROP_RATE": b["drop"], "RESUME": f"{TB}/final_model_release/{b['resume']}", "MODEL_DIR": f"{PFA}/runs/finetune_TPL/PreTrainedModels"}
        c["TEST"]["USE_LAST_EPOCH"] = True; c["INFERENCE"]["BATCH_SIZE"] = 4; c["INFERENCE"]["MODEL_PATH"] = ""
        for sec in ("TRAIN", "VALID", "TEST"): overlay(c[sec], b, False, f"{b['prefix']}_tpl")
        dump(c, f"{PFA}/yaml_BB/tpl_{bname}.yaml")
    elif mode == "ev_axis":
        do_pre = 1
        if "--preprocess" in a: i = a.index("--preprocess"); do_pre = int(a[i + 1]); del a[i:i + 2]
        name, bname, ckpt, ax = a[0], a[1], a[2], a[3]; b = BB[bname]
        c = yaml.safe_load(open(f"{PFA}/yaml_CoTAM_HYB4_L/ev_CoTAM_HYB4_L_{ax}.yaml"))
        c["MODEL"] = {"NAME": bname, "DROP_RATE": b["drop"]}
        exp = c["TEST"]["DATA"]["EXP_DATA_NAME"]; overlay(c["TEST"], b, do_pre, f"{b['prefix']}_{exp}")
        c["INFERENCE"]["MODEL_PATH"] = ckpt; c["INFERENCE"]["BATCH_SIZE"] = 4
        c["LOG"]["PATH"] = f"{PFA}/runs/ev_{name}_{ax}"; c["TEST"]["OUTPUT_SAVE_DIR"] = f"{PFA}/runs/ev_{name}_{ax}/outputs"
        dump(c, f"{PFA}/yaml_{name}/ev_{name}_{ax}.yaml")
    else:
        raise SystemExit(__doc__)
