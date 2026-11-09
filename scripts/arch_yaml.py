#!/usr/bin/env python3
"""[45 문서 ⑨] 공개 가중치 통제 스트레스 평가 yaml 생성: python scripts/arch_yaml.py MODEL AXIS → yaml_arch/ev_{MODEL}_arch_{AXIS}.yaml
MODEL ∈ EfficientPhys PhysNet DeepPhys Tscan (틀: yaml15/ev_{EfficientPhys|PhysNet}_PURE.yaml; DeepPhys/Tscan 은 EfficientPhys 틀 + MODEL 절·DATA_TYPE 교체).
축 정의(DATASET·DATA_PATH·BEGIN·END·FS)는 채택 모델 yaml_CoTAF_TW_L/ev_CoTAF_TW_L_{AXIS}.yaml 의 TEST.DATA 에서 복사(동일 영상·동일 분할 보장)."""
import os, sys, yaml
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; REL = "/SPECPHYS_ROOT/repos/rPPG-Toolbox/final_model_release"
model, axis = sys.argv[1], sys.argv[2]
tpl = f"{PFA}/yaml15/ev_{'PhysNet' if model == 'PhysNet' else 'EfficientPhys'}_PURE.yaml"
c = yaml.safe_load(open(tpl)); src = yaml.safe_load(open(f"{PFA}/yaml_CoTAF_TW_L/ev_CoTAF_TW_L_{axis}.yaml"))["TEST"]["DATA"]
for k in ("DATASET", "DATA_PATH", "BEGIN", "END", "FS"): c["TEST"]["DATA"][k] = src[k]
for k in ("DO_DYNAMIC_DETECTION", "DYNAMIC_DETECTION_FREQUENCY", "USE_LARGE_FACE_BOX", "LARGE_BOX_COEF", "DO_CROP_FACE"):
    if k in src.get("PREPROCESS", {}).get("CROP_FACE", {}): c["TEST"]["DATA"]["PREPROCESS"]["CROP_FACE"][k] = src["PREPROCESS"]["CROP_FACE"][k]
name = f"{model}_arch"; c["TEST"]["DATA"]["EXP_DATA_NAME"] = f"arch_{model}_{axis}"; c["TEST"]["DATA"]["DO_PREPROCESS"] = True
c["LOG"]["PATH"] = f"{PFA}/runs/ev_{name}_{axis}"; c["TEST"]["OUTPUT_SAVE_DIR"] = f"{PFA}/runs/ev_{name}_{axis}/outputs"
if model == "Tscan":
    c["MODEL"] = {"DROP_RATE": 0.2, "NAME": "Tscan", "TSCAN": {"FRAME_DEPTH": 10}}; c["TEST"]["DATA"]["PREPROCESS"]["DATA_TYPE"] = ["DiffNormalized", "Standardized"]; c["INFERENCE"]["MODEL_PATH"] = f"{REL}/MA-UBFC_tscan.pth"
elif model == "DeepPhys":
    c["MODEL"] = {"DROP_RATE": 0.2, "NAME": "DeepPhys"}; c["TEST"]["DATA"]["PREPROCESS"]["DATA_TYPE"] = ["DiffNormalized", "Standardized"]; c["INFERENCE"]["MODEL_PATH"] = f"{REL}/UBFC-rPPG_DeepPhys.pth"
elif model == "EfficientPhys": c["INFERENCE"]["MODEL_PATH"] = f"{REL}/UBFC-rPPG_EfficientPhys.pth"
elif model == "PhysNet": c["INFERENCE"]["MODEL_PATH"] = f"{REL}/UBFC-rPPG_PhysNet_DiffNormalized.pth"
os.makedirs(f"{PFA}/yaml_arch", exist_ok=True); out = f"{PFA}/yaml_arch/ev_{name}_{axis}.yaml"; yaml.safe_dump(c, open(out, "w"), sort_keys=False); print(out)
