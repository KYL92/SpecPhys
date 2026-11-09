#!/usr/bin/env python3
"""PhysFormer 실험용 rPPG-Toolbox config YAML 생성.
- 조건별 baseline only_test (사전학습 체크포인트)
- 미세조정 train_and_test (MODEL.RESUME=사전학습)
- 미세조정 후 조건별 only_test (미세조정 체크포인트)
"""
import os

ROOT = "/SPECPHYS_ROOT"
PF = f"{ROOT}/_workspace/data/physformer"
TB = f"{ROOT}/repos/rPPG-Toolbox"
PRETRAINED = f"{TB}/final_model_release/UBFC-rPPG_PhysFormer_DiffNormalized.pth"
FINETUNED = f"{PF}/runs/finetune/PreTrainedModels/SYNTH_finetune_physformer_Epoch9.pth"
CONDS = ["C1", "C2", "C3", "C4", "C5", "C6"]

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


def data_block(path, name, begin, end, do_pre="True"):
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


def only_test(cond, model_path, tag):
    return f"""BASE: ['']
TOOLBOX_MODE: "only_test"
DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1
LOG:
  PATH: {PF}/runs/{tag}_{cond}
MODEL:
  NAME: PhysFormer
  DROP_RATE: 0.1
TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  USE_LAST_EPOCH: True
  OUTPUT_SAVE_DIR: {PF}/runs/{tag}_{cond}/outputs
  DATA:
{data_block(f"{PF}/TEST_{cond}", f"synth_{cond}", 0.0, 1.0)}
INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: FFT
  MODEL_PATH: "{model_path}"
"""


def finetune():
    return f"""BASE: ['']
TOOLBOX_MODE: "train_and_test"
DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1
LOG:
  PATH: {PF}/runs/finetune
MODEL:
  NAME: PhysFormer
  DROP_RATE: 0.1
  RESUME: "{PRETRAINED}"
  MODEL_DIR: {PF}/runs/finetune/PreTrainedModels
TRAIN:
  BATCH_SIZE: 4
  EPOCHS: 10
  LR: 5e-5
  MODEL_FILE_NAME: SYNTH_finetune_physformer
  PLOT_LOSSES_AND_LR: False
  DATA:
{data_block(f"{PF}/TRAIN", "synth_train", 0.0, 0.8)}
VALID:
  DATA:
{data_block(f"{PF}/TRAIN", "synth_train", 0.8, 1.0)}
TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  USE_LAST_EPOCH: True
  OUTPUT_SAVE_DIR: {PF}/runs/finetune/outputs
  DATA:
{data_block(f"{PF}/TEST_C5", "synth_C5", 0.0, 1.0)}
INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: FFT
"""


os.makedirs(f"{PF}/yaml", exist_ok=True)
for c in CONDS:
    open(f"{PF}/yaml/base_{c}.yaml", "w").write(only_test(c, PRETRAINED, "base"))
    open(f"{PF}/yaml/ft_{c}.yaml", "w").write(only_test(c, FINETUNED, "ft"))
open(f"{PF}/yaml/finetune.yaml", "w").write(finetune())
print("yaml written:", os.listdir(f"{PF}/yaml"))
