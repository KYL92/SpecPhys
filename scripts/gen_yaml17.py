#!/usr/bin/env python3
"""yaml17 생성: MMPD 배치 평가(6배치×3 캐시군) + CoTAM(=CoTAF 풀 + MMPD 1~6) 학습/평가."""
import glob, os, yaml, shutil
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"; Y = f"{PF}/yaml17"; os.makedirs(Y, exist_ok=True)
MMPD = "/SPECPHYS_RAW/MMPD"
BATCHES = {1: range(1, 7), 2: range(7, 13), 3: range(13, 19), 4: range(19, 25), 5: range(25, 31), 6: range(31, 34)}
for k, subs in BATCHES.items():
    d = f"{PF}/MMPD_B{k}"; os.makedirs(d, exist_ok=True)
    for s in subs:
        dst = f"{d}/subject{s}"
        if not os.path.islink(dst): os.symlink(f"{MMPD}/subject{s}", dst)
INFO = dict(LIGHT=[1, 2, 3, 4], MOTION=[1, 2, 3, 4], EXERCISE=[1, 2], SKIN_COLOR=[3, 4, 5, 6], GENDER=[1, 2], GLASSER=[1, 2], HAIR_COVER=[1, 2], MAKEUP=[1, 2])
RUNS = f"{PF}/runs"; REL = "/SPECPHYS_ROOT/repos/rPPG-Toolbox/final_model_release"
MODELS = {  # name: (template yaml, MODEL_PATH, cache group)
 "base":   (f"{PF}/yaml14/ev_CoTAF_DLCN.yaml", f"{REL}/UBFC-rPPG_PhysFormer_DiffNormalized.pth", "pf"),
 "CoTNH":  (f"{PF}/yaml14/ev_CoTAF_DLCN.yaml", f"{RUNS}/finetune_CoTNH/PreTrainedModels/SYNTH_CoTNH_physformer_Epoch9.pth", "pf"),
 "CoTA":   (f"{PF}/yaml14/ev_CoTAF_DLCN.yaml", f"{RUNS}/finetune_CoTA/PreTrainedModels/SYNTH_CoTA_physformer_Epoch9.pth", "pf"),
 "CoTAF":  (f"{PF}/yaml14/ev_CoTAF_DLCN.yaml", f"{RUNS}/finetune_CoTAF/PreTrainedModels/SYNTH_CoTAF_physformer_Epoch9.pth", "pf"),
 "CoTAF2": (f"{PF}/yaml14/ev_CoTAF_DLCN.yaml", f"{RUNS}/finetune_CoTAF2/PreTrainedModels/SYNTH_CoTAF2_physformer_Epoch9.pth", "pf"),
 "CoTAM":  (f"{PF}/yaml14/ev_CoTAF_DLCN.yaml", f"{RUNS}/finetune_CoTAM/PreTrainedModels/SYNTH_CoTAM_physformer_Epoch9.pth", "pf"),
 "RhythmFormer": (f"{PF}/yaml15/ev_RhythmFormer_DLCN.yaml", f"{REL}/UBFC-rPPG_RhythmFormer.pth", "rf"),
 "RFA":    (f"{PF}/yaml15/ev_RhythmFormer_DLCN.yaml", f"{RUNS}/finetune_RFA/PreTrainedModels/ALLF2_rhythmformer_Epoch9.pth", "rf"),
 "FactorizePhys": (f"{PF}/yaml15/ev_FactorizePhys_DLCN.yaml", f"{REL}/UBFC-rPPG_FactorizePhys_FSAM_Res.pth", "fp"),
 "FFA":    (f"{PF}/yaml15/ev_FactorizePhys_DLCN.yaml", f"{RUNS}/finetune_FFA/PreTrainedModels/ALLF2_factorizephys_Epoch9.pth", "fp"),
}
def load(p): return yaml.safe_load(open(p))
def ev_yaml(name, k, preprocess):
    tpl, mp, grp = MODELS[name]; c = load(tpl); ax = f"MMPD_B{k}"
    c["LOG"]["PATH"] = f"{RUNS}/ev_{name}_{ax}"; c["TEST"]["OUTPUT_SAVE_DIR"] = f"{RUNS}/ev_{name}_{ax}/outputs"
    d = c["TEST"]["DATA"]; d["DATASET"] = "MMPD"; d["DATA_PATH"] = f"{PF}/{ax}"; d["EXP_DATA_NAME"] = f"mmpd_{grp}_B{k}"
    d["DO_PREPROCESS"] = preprocess; d["INFO"] = INFO; d["PREPROCESS"]["CROP_FACE"]["DO_CROP_FACE"] = True
    c["INFERENCE"]["MODEL_PATH"] = mp; c["INFERENCE"]["BATCH_SIZE"] = 1
    out = f"{Y}/{'pre' if preprocess else 'ev'}_{name}_{ax}.yaml"; yaml.safe_dump(c, open(out, "w"), sort_keys=False); return out
for k in BATCHES:
    for name, (_, _, grp) in MODELS.items():
        ev_yaml(name, k, False)
    for first in ("base", "RhythmFormer", "FactorizePhys"): ev_yaml(first, k, True)
# CoTAM 학습: CoTAF 설정 복제, 풀만 COTRAIN_ALLFM
c = load(f"{PF}/yaml12/finetune_CoTAF.yaml")
for key in ("LOG",): c[key]["PATH"] = f"{RUNS}/finetune_CoTAM"
c["MODEL"]["MODEL_DIR"] = f"{RUNS}/finetune_CoTAM/PreTrainedModels"; c["TRAIN"]["MODEL_FILE_NAME"] = "SYNTH_CoTAM_physformer"
c["TRAIN"]["DATA"]["DATA_PATH"] = f"{PF}/COTRAIN_ALLFM"; c["TRAIN"]["DATA"]["EXP_DATA_NAME"] = "cotrain_allfm"; c["TRAIN"]["DATA"]["DO_PREPROCESS"] = True
c["TEST"]["OUTPUT_SAVE_DIR"] = f"{RUNS}/finetune_CoTAM/outputs"
yaml.safe_dump(c, open(f"{Y}/finetune_CoTAM.yaml", "w"), sort_keys=False)
# CoTAM 기존 축 평가: CoTAF yaml 복제
for src in glob.glob(f"{PF}/yaml12/ev_CoTAF_*.yaml") + glob.glob(f"{PF}/yaml14/ev_CoTAF_*.yaml"):
    ax = os.path.basename(src)[len("ev_CoTAF_"):-5]; c = load(src)
    c["LOG"]["PATH"] = f"{RUNS}/ev_CoTAM_{ax}"; c["TEST"]["OUTPUT_SAVE_DIR"] = f"{RUNS}/ev_CoTAM_{ax}/outputs"
    c["INFERENCE"]["MODEL_PATH"] = MODELS["CoTAM"][1]; c["TEST"]["DATA"]["DO_PREPROCESS"] = False
    yaml.safe_dump(c, open(f"{Y}/ev_CoTAM_{ax}.yaml", "w"), sort_keys=False)
print("yaml17:", len(os.listdir(Y)), "files")
