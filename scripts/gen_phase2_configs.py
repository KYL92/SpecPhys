#!/usr/bin/env python3
"""2단계 실험 config 생성: T1(clean-only) 미세조정, TRAIN20 확장 미세조정(시드 3회),
조건·외삽·실데이터(UBFC) 평가 yaml 일괄 생성."""
import os

ROOT = "/SPECPHYS_ROOT"
PF = f"{ROOT}/_workspace/data/physformer"
TB = f"{ROOT}/repos/rPPG-Toolbox"
PRETRAINED = f"{TB}/final_model_release/UBFC-rPPG_PhysFormer_DiffNormalized.pth"

MODELS = {
    "base": PRETRAINED,
    "T2": f"{PF}/runs/finetune/PreTrainedModels/SYNTH_finetune_physformer_Epoch9.pth",
    "T1": f"{PF}/runs/finetune_T1/PreTrainedModels/SYNTH_T1clean_physformer_Epoch9.pth",
    "T2L_s100": f"{PF}/runs/finetune_T2L_s100/PreTrainedModels/SYNTH_T2L_physformer_Epoch9.pth",
    "T2L_s200": f"{PF}/runs/finetune_T2L_s200/PreTrainedModels/SYNTH_T2L_physformer_Epoch9.pth",
    "T2L_s300": f"{PF}/runs/finetune_T2L_s300/PreTrainedModels/SYNTH_T2L_physformer_Epoch9.pth",
}

PRE = """      DATA_TYPE: ['DiffNormalized']
      DATA_AUG: ['None']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 160
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 128
        W: 128"""


def data_block(path, name, begin, end, do_pre):
    return f"""    FS: 30
    DATASET: UBFC-rPPG
    DO_PREPROCESS: {do_pre}
    DATA_FORMAT: NCDHW
    DATA_PATH: "{path}"
    CACHED_PATH: "{PF}/cache"
    EXP_DATA_NAME: "{name}"
    BEGIN: {begin}
    END: {end}
    PREPROCESS:
{PRE}"""


def testset(tag):
    if tag == "UBFC":
        return f"{ROOT}/_workspace/data/UBFC_real/DATASET_2", "real_UBFC"
    return f"{PF}/TEST_{tag}", f"synth_{tag}"


def only_test(model, tag, do_pre):
    path, name = testset(tag)
    return f"""BASE: ['']
TOOLBOX_MODE: "only_test"
DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1
LOG:
  PATH: {PF}/runs/ev_{model}_{tag}
MODEL:
  NAME: PhysFormer
  DROP_RATE: 0.1
TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  USE_LAST_EPOCH: True
  OUTPUT_SAVE_DIR: {PF}/runs/ev_{model}_{tag}/outputs
  DATA:
{data_block(path, name, 0.0, 1.0, do_pre)}
INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: FFT
  MODEL_PATH: "{MODELS[model]}"
"""


def finetune(name, train_root, model_dir, log_dir, file_name):
    return f"""BASE: ['']
TOOLBOX_MODE: "train_and_test"
DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1
LOG:
  PATH: {log_dir}
MODEL:
  NAME: PhysFormer
  DROP_RATE: 0.1
  RESUME: "{PRETRAINED}"
  MODEL_DIR: {model_dir}
TRAIN:
  BATCH_SIZE: 4
  EPOCHS: 10
  LR: 5e-5
  MODEL_FILE_NAME: {file_name}
  PLOT_LOSSES_AND_LR: False
  DATA:
{data_block(train_root, name, 0.0, 0.8, True)}
VALID:
  DATA:
{data_block(train_root, name, 0.8, 1.0, True)}
TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  USE_LAST_EPOCH: True
  OUTPUT_SAVE_DIR: {log_dir}/outputs
  DATA:
{data_block(f"{PF}/TEST_C5", "synth_C5", 0.0, 1.0, False)}
INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: FFT
"""


Y = f"{PF}/yaml2"
os.makedirs(Y, exist_ok=True)
open(f"{Y}/finetune_T1.yaml", "w").write(
    finetune("synth_train_clean", f"{PF}/TRAIN_CLEAN",
             f"{PF}/runs/finetune_T1/PreTrainedModels", f"{PF}/runs/finetune_T1",
             "SYNTH_T1clean_physformer"))
for s in (100, 200, 300):
    open(f"{Y}/finetune_T2L_s{s}.yaml", "w").write(
        finetune("synth_train20", f"{PF}/TRAIN20",
                 f"{PF}/runs/finetune_T2L_s{s}/PreTrainedModels",
                 f"{PF}/runs/finetune_T2L_s{s}", "SYNTH_T2L_physformer"))

# 전처리 1회 담당 (base × 신규 세트)
prep = [("base", "E1"), ("base", "E2"), ("base", "E3"), ("base", "UBFC")]
rest = ([("T2", t) for t in ["E1", "E2", "E3", "UBFC"]] +
        [("T1", t) for t in ["C1", "C2", "C3", "C4", "C5", "C6", "E1", "E2", "E3", "UBFC"]] +
        [(f"T2L_s{s}", t) for s in (100, 200, 300)
         for t in ["C3", "C4", "C5", "C6", "E1", "E2", "E3"]] +
        [("T2L_s100", "UBFC")])
for m, t in prep:
    open(f"{Y}/ev_{m}_{t}.yaml", "w").write(only_test(m, t, "True"))
for m, t in rest:
    open(f"{Y}/ev_{m}_{t}.yaml", "w").write(only_test(m, t, "False"))
print("prep runs:", [f"{m}_{t}" for m, t in prep])
print("rest runs:", len(rest))
