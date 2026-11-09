#!/usr/bin/env python3
"""[체인 53, 36 문서] FactorizePhys 셀 yaml 생성 — PhysFormer 셀 yaml(yaml_PF_U{k}/)을 틀로 두고 FactorizePhys 전용 키를 덮어쓴다.
  python scripts/fp_yaml.py train NAME PFNAME LR [--valid_dir DIR] [--epochs N]   → yaml_{NAME}/finetune_{NAME}.yaml (Raw 72×72·Standardized 라벨, UBFC FSAM_Res RESUME, 캐시 fp_*)
  python scripts/fp_yaml.py ev NAME PFNAME AXIS [--preprocess 0|1]                 → yaml_{NAME}/ev_{NAME}_{AXIS}.yaml (캐시 fp_{axis}, 첫 모델만 전처리)
MODEL 블록은 툴박스 학습 설정(UBFC-rPPG_PURE_FactorizePhys_FSAM_Res.yaml: MD_FSAM True·MD_INFERENCE True)을 학습·평가에 동일하게 쓴다."""
import os, sys, yaml
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; TB = "/SPECPHYS_ROOT/repos/rPPG-Toolbox"
TPL = yaml.safe_load(open(f"{TB}/configs/train_configs/UBFC-rPPG_PURE_FactorizePhys_FSAM_Res.yaml"))
RESUME = f"{TB}/final_model_release/UBFC-rPPG_FactorizePhys_FSAM_Res.pth"


def overlay_pre(sec, do_pre, name):
    d = sec["DATA"]; pre = d["PREPROCESS"]
    pre["DATA_TYPE"] = ["Raw"]; pre["LABEL_TYPE"] = "Standardized"; pre["RESIZE"] = {"H": 72, "W": 72}; pre["CHUNK_LENGTH"] = 160; pre["DO_CHUNK"] = True
    d["DO_PREPROCESS"] = bool(do_pre); d["EXP_DATA_NAME"] = "fp_" + name
    # 10-19: 완성된 캐시(툴박스 파일 목록 csv + 캐시 디렉토리)가 있으면 재전처리하지 않는다 — 학습률 탐색 3셀·Phase 1/2 셀이 같은 fp_* 캐시를 공유
    if d["DO_PREPROCESS"]:
        b, e = float(d.get("BEGIN", 0.0)), float(d.get("END", 1.0))
        lst = f"{PFA}/cache/DataFileLists/fp_{name}_{b}_{e}.csv"
        if os.path.exists(lst) and os.path.isdir(f"{PFA}/cache/fp_{name}") and os.listdir(f"{PFA}/cache/fp_{name}"):
            d["DO_PREPROCESS"] = False; print(f"  [fp_yaml] cache reuse fp_{name} (DO_PREPROCESS false)", file=sys.stderr)


def dump(c, p):
    os.makedirs(os.path.dirname(p), exist_ok=True); yaml.safe_dump(c, open(p, "w"), sort_keys=False, default_flow_style=False); print(p)


if __name__ == "__main__":
    mode = sys.argv[1]; a = sys.argv[2:]
    if mode == "train":
        valid_dir = None; epochs = 10
        if "--valid_dir" in a: i = a.index("--valid_dir"); valid_dir = a[i + 1]; del a[i:i + 2]
        if "--epochs" in a: i = a.index("--epochs"); epochs = int(a[i + 1]); del a[i:i + 2]
        name, pfname, lr = a[0], a[1], a[2]
        c = yaml.safe_load(open(f"{PFA}/yaml_{pfname}/finetune_{pfname}.yaml"))
        c["MODEL"] = {"DROP_RATE": 0.1, "NAME": "FactorizePhys", "FactorizePhys": dict(TPL["MODEL"]["FactorizePhys"]), "RESUME": RESUME, "MODEL_DIR": f"{PFA}/runs/finetune_{name}/PreTrainedModels"}
        c["TRAIN"]["LR"] = lr; c["TRAIN"]["EPOCHS"] = epochs; c["TRAIN"]["MODEL_FILE_NAME"] = f"SYNTH_{name}_factorizephys"; c["TRAIN"]["PLOT_LOSSES_AND_LR"] = False
        overlay_pre(c["TRAIN"], True, c["TRAIN"]["DATA"]["EXP_DATA_NAME"])
        # 테스트 = UBFC held-out(REALTEST, 피험자 42~49) Raw 캐시
        c["TEST"]["DATA"]["DATA_PATH"] = f"{PFA}/REALTEST"; c["TEST"]["DATA"]["DATASET"] = "UBFC-rPPG"; c["TEST"]["DATA"]["EXP_DATA_NAME"] = "real_test"; overlay_pre(c["TEST"], True, "real_test")
        c["TEST"]["USE_LAST_EPOCH"] = True
        if valid_dir:  # 학습률 탐색: 학습 피험자 일부를 검증으로, 검증 손실 최소 epoch 선택(평가 축 비관여)
            c["VALID"]["DATA"]["DATA_PATH"] = valid_dir; c["VALID"]["DATA"]["DATASET"] = "UBFC-rPPG"; c["VALID"]["DATA"]["EXP_DATA_NAME"] = os.path.basename(valid_dir).lower(); c["VALID"]["DATA"]["BEGIN"] = 0.0; c["VALID"]["DATA"]["END"] = 1.0; overlay_pre(c["VALID"], True, os.path.basename(valid_dir).lower()); c["TEST"]["USE_LAST_EPOCH"] = False
        else:
            overlay_pre(c["VALID"], False, c["VALID"]["DATA"]["EXP_DATA_NAME"])
        c["LOG"]["PATH"] = f"{PFA}/runs/finetune_{name}"; c["TEST"]["OUTPUT_SAVE_DIR"] = f"{PFA}/runs/finetune_{name}/outputs"
        c["INFERENCE"]["BATCH_SIZE"] = 4
        dump(c, f"{PFA}/yaml_{name}/finetune_{name}.yaml")
    elif mode == "ev":
        do_pre = 1
        if "--preprocess" in a: i = a.index("--preprocess"); do_pre = int(a[i + 1]); del a[i:i + 2]
        name, pfname, ax = a[0], a[1], a[2]
        c = yaml.safe_load(open(f"{PFA}/yaml_{pfname}/ev_{pfname}_{ax}.yaml"))
        c["MODEL"] = {"DROP_RATE": 0.1, "NAME": "FactorizePhys", "FactorizePhys": dict(TPL["MODEL"]["FactorizePhys"])}
        overlay_pre(c["TEST"], do_pre, c["TEST"]["DATA"]["EXP_DATA_NAME"])
        c["INFERENCE"]["MODEL_PATH"] = f"{PFA}/runs/finetune_{name}/PreTrainedModels/SYNTH_{name}_factorizephys_Epoch9.pth"; c["INFERENCE"]["BATCH_SIZE"] = 4
        c["LOG"]["PATH"] = f"{PFA}/runs/ev_{name}_{ax}"; c["TEST"]["OUTPUT_SAVE_DIR"] = f"{PFA}/runs/ev_{name}_{ax}/outputs"
        dump(c, f"{PFA}/yaml_{name}/ev_{name}_{ax}.yaml")
    else:
        raise SystemExit(__doc__)
