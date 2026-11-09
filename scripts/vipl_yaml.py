#!/usr/bin/env python3
"""VIPL-HR 실험용 yaml 생성 (_workspace/data/physformer/yaml_VIPL/). 템플릿 = 공식 모델 CoTAF_TW_L yaml.
  python scripts/vipl_yaml.py ev    MODEL MODEL_PATH [--template YAML] → ev_{MODEL}_VIPL.yaml   (only_test, vipl_all.csv; --template = 다른 백본 ev yaml 틀)
  python scripts/vipl_yaml.py train NAME FOLD TRAIN_CSV [EPOCHS] [--lr LR] [--scratch] [--resume CKPT] [--std] [--test_csv CSV] [--valid_csv CSV] → finetune_{NAME}.yaml   (train_and_test, test=vipl_test{FOLD}.csv; --valid_csv: VALID=VIPL 검증 목록 + USE_LAST_EPOCH false = 검증 최적 epoch 선택, 32 문서 §5)
공통: DATASET VIPL-HR, DO_PREPROCESS false, FILE_LIST_PATH 를 csv 로 직접 지정(uint8 공유 캐시 cache/vipl_u8)."""
import os, sys, yaml
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"
RAW = "/SPECPHYS_RAW/vipl_hr_raw"; Y = f"{PFA}/yaml_VIPL"; LISTS = f"{PFA}/cache/DataFileLists"


def vipl_data(csv):
    return dict(FS=30, DATASET="VIPL-HR", DO_PREPROCESS=False, DATA_FORMAT="NCDHW", DATA_PATH=RAW,
                CACHED_PATH=f"{PFA}/cache", EXP_DATA_NAME="vipl_u8", FILE_LIST_PATH=csv, BEGIN=0.0, END=1.0)


def load(p):
    return yaml.safe_load(open(p))


def dump(c, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    yaml.safe_dump(c, open(p, "w"), sort_keys=False, default_flow_style=False)
    print(p)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ev":
        # ev MODEL MODEL_PATH [--template YAML]  (--template: 다른 백본의 ev yaml 을 틀로 — MODEL 절·DATA_TYPE 유지, 33 문서 체인 46z4 RhythmFormer/FactorizePhys VIPL zero-shot)
        a = sys.argv[2:]; tpl = None
        if "--template" in a: i = a.index("--template"); tpl = a[i + 1]; del a[i:i + 2]
        model, mpath = a[0], a[1]
        c = load(tpl or f"{PFA}/yaml_CoTAF_TW_L/ev_CoTAF_TW_L_PURE.yaml")
        run = f"{PFA}/runs/ev_{model}_VIPL"
        c["LOG"]["PATH"] = run; c["TEST"]["OUTPUT_SAVE_DIR"] = run + "/outputs"
        vd = vipl_data(f"{LISTS}/vipl_all.csv")
        if tpl: vd.pop("DATA_FORMAT")  # 템플릿 백본의 입력 배치(NDCHW 등) 유지 — RhythmFormer 는 NDCHW(10-17 RFbase 실패 원인)
        c["TEST"]["DATA"].update(vd)
        c["INFERENCE"]["MODEL_PATH"] = mpath; c["INFERENCE"]["BATCH_SIZE"] = 1
        dump(c, f"{Y}/ev_{model}_VIPL.yaml")
    elif mode == "train":
        # train NAME FOLD TRAIN_CSV [EPOCHS] [--lr LR] [--scratch] [--test_csv CSV]   (31 문서: 문헌 레시피 = 25 --lr 1e-4 --scratch)
        args = sys.argv[2:]; opts = {}
        while "--lr" in args: i = args.index("--lr"); opts["lr"] = float(args[i + 1]); del args[i:i + 2]
        while "--test_csv" in args: i = args.index("--test_csv"); opts["test_csv"] = args[i + 1]; del args[i:i + 2]
        while "--valid_csv" in args: i = args.index("--valid_csv"); opts["valid_csv"] = args[i + 1]; del args[i:i + 2]
        if "--scratch" in args: opts["scratch"] = True; args.remove("--scratch")
        while "--resume" in args: i = args.index("--resume"); opts["resume"] = args[i + 1]; del args[i:i + 2]   # [45 문서] 2단계 미세조정: 출발 체크포인트 지정
        while "--template" in args: i = args.index("--template"); opts["template"] = args[i + 1]; del args[i:i + 2]   # [48 문서] 다른 백본(FactorizePhys/NF-Phys)의 finetune yaml 을 틀로: MODEL 절·PREPROCESS(Raw 72×72·Standardized 라벨) 유지
        if "--std" in args: opts["std"] = True; args.remove("--std")   # 입력 Standardized(u8 캐시 즉석 계산, LITS 셀)
        name, fold, csv = args[0], int(args[1]), args[2]
        epochs = int(args[3]) if len(args) > 3 else 10
        c = load(opts.get("template") or f"{PFA}/yaml_CoTAF_TW_L/finetune_CoTAF_TW_L.yaml")
        run = f"{PFA}/runs/finetune_{name}"
        c["LOG"]["PATH"] = run; c["MODEL"]["MODEL_DIR"] = run + "/PreTrainedModels"
        _sfx = "physformer" if "physformer" in str(c["MODEL"]["NAME"]).lower() else str(c["MODEL"]["NAME"]).lower()
        c["TRAIN"]["MODEL_FILE_NAME"] = f"SYNTH_{name}_{_sfx}"; c["TRAIN"]["EPOCHS"] = epochs
        def _vd(x):  # [57 문서] --template 이면 템플릿 백본의 입력 배치(RhythmFormer NDCHW) 유지 — ev 모드와 동일 규칙
            d = vipl_data(x)
            if "template" in opts: d.pop("DATA_FORMAT")
            return d
        c["TRAIN"]["DATA"].update(_vd(csv))
        c["TEST"]["OUTPUT_SAVE_DIR"] = run + "/outputs"
        c["TEST"]["DATA"].update(_vd(opts.get("test_csv", f"{LISTS}/vipl_test{fold}.csv")))
        c["INFERENCE"]["BATCH_SIZE"] = 1
        if "lr" in opts: c["TRAIN"]["LR"] = opts["lr"]
        if "valid_csv" in opts:  # [32 문서 §5] 검증 기반 epoch 선택
            c["VALID"]["DATA"].update(_vd(opts["valid_csv"])); c["TEST"]["USE_LAST_EPOCH"] = False
        if opts.get("scratch"): c["MODEL"]["RESUME"] = ""
        if opts.get("resume"): c["MODEL"]["RESUME"] = opts["resume"]
        if opts.get("std"):
            for sec in ("TRAIN", "VALID", "TEST"):
                if sec in c and "DATA" in c[sec]: c[sec]["DATA"]["PREPROCESS"]["DATA_TYPE"] = ["Standardized"]
        dump(c, f"{Y}/finetune_{name}.yaml")
    else:
        raise SystemExit(__doc__)
