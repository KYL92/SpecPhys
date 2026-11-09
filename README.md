# SpecPhys

Code and experiment records for **SpecPhys: Spectrum-Guided Cross-Dataset Remote Physiological Measurement with PPG-Injected Synthetic Video** (CVPR submission, under review).

> This repository stays private while the paper is under review. It will be made public after the decision, together with a license for our own code.

## Contents

| Part | Path | Description |
|---|---|---|
| Synthetic video generator | `generator/` | Injects a real PPG waveform into the skin pixels of animated generated faces along the blood-volume-pulse direction [0.33, 0.77, 0.53] (R, G, B), adds controlled nuisance (illumination drift and flicker, rigid motion, sensor noise), can render a second pass without the pulse (dual render), and writes lossless FFV1 clips with ground truth in the UBFC-rPPG layout. Entry point: `build_synth_dataset.py --config <json>`. |
| rPPG-Toolbox modifications | `toolbox_patch/` | Patch against rPPG-Toolbox commit `b7500b8` and new files: the two-slot FactorizePhys variant with the noise-subspace projection layer (`FactorizePhysNF.py`), the spectral losses (`NFLoss.py`), a VIPL-HR loader, an 8-bit frame cache, temporal augmentation and further training switches. Every switch is an environment variable that defaults to upstream behaviour; see `docs/toolbox_patches_and_protocol.md`. Distributed under the rPPG-Toolbox license (RAIL v1.1). |
| Experiment scripts | `scripts/` | Dataset converters, frame caches, training and evaluation chains, aggregation (`agg_model2.py`, `agg_vipl.py`, `agg_mmpd_cv2.py`, `mmpd_common_gt.py`), pre-registered statistical judgements, tables and figures. |
| Configurations | `_workspace/data/physformer/configs/`, `_workspace/data/physformer/yaml_*` | Synthetic-set definitions (JSON) and the rPPG-Toolbox YAML templates used for the final model and the baselines. |
| Experiment records | `docs/experiments/` | Pre-registrations, results and the change log of every experiment (in Korean), see `docs/experiments/README.md`. |

## Setup

1. **Environment.** Python 3.10 and PyTorch built for CUDA 12.8 (Blackwell GPUs need cu128 wheels). `setup.sh` builds the two conda environments used in the study, `vface-gen` (face animation) and `vface-rppg` (training, evaluation, aggregation); `python verify_env.py` checks the CUDA kernels.
2. **rPPG-Toolbox.** `bash toolbox_patch/apply.sh` clones rPPG-Toolbox into `repos/rPPG-Toolbox`, checks out commit `b7500b8` and applies the modifications.
3. **Paths.** The scripts were written for one fixed directory layout. Absolute paths are replaced by placeholders: `/SPECPHYS_ROOT` (this repository; data, caches and runs are created under `_workspace/data/`), `/SPECPHYS_RAW` (raw datasets), `/SPECPHYS_CACHE` (large disk for frame caches), `/SPECPHYS_CONDA` (conda root that holds the two environments), `/SPECPHYS_TMP` and `/SPECPHYS_HOME`. Set them once:
   ```bash
   python tools/set_paths.py --root "$(pwd)" --raw /data/rppg --cache /data/cache --conda ~/miniforge3 --tmp /tmp/specphys
   ```

## Data

No dataset, ground truth, frame cache or model weight is distributed here.

- Real datasets: VIPL-HR, MMPD, PURE, UBFC-rPPG, MR-NIRP (car), DLCN and MCD-rPPG must be obtained from their owners under their terms. Converters to the UBFC-rPPG layout are `scripts/convert_*.py`; VIPL-HR is read directly by the patched loader (`scripts/vipl_preprocess.py`, `scripts/vipl_filelist.py`).
- PPG waveform bank: built from the UBFC-rPPG and PURE training ground truth by `scripts/build_ppg_bank*.py`.
- Generated faces: identities from the public SFHQ set, animated with LivePortrait; face-parsing sidecars use the public SegFormer face parser `jonathandinu/face-parsing` (`scripts/face_parse_sidecar.py`).

## Final model

SpecPhys is the cell `NF_NSP2_L` in the scripts (earlier stages: `NF_HYB_L` = SpecPhys-B, `NF_SKREF_REP_L` = the recipe without the projection layer). It is fine-tuned for 10 epochs (learning rate 1e-4) from the public UBFC-rPPG FactorizePhys checkpoint on the 366-clip pool with:

```bash
export RPPG_NF=1 RPPG_SKINW=0.2 NF_SS_TRACE=1 NF_SS_REF=1 NF_ARCH=nsp NF_NSP_LAM=0.01 \
       HYB_NOISE_GT_DIR=<dual-render noise labels> RPPG_CACHE_U8=1 \
       NF_W_DBL=0.5 RPPG_TAUG=1 RPPG_TAUG_P=0.25 RPPG_TAUG_HR_SPLIT=300 RPPG_SEED=<seed>
```

`RPPG_NF`, `RPPG_SKINW`, `NF_ARCH` and `NF_NSP_LAM` must also be set at evaluation time, because checkpoints are loaded non-strictly; the `NF_SS_*` terms are training losses only. The evaluation scripts set the whole list. The runs reported in the paper are the chain scripts `scripts/run_chain118_nsp2.sh` to `scripts/run_chain122_abl_indomain.sh` (selection, confirmation on seeds 1600–2500, 2×2 ablation), and `scripts/run_chain123_contrib.sh` to `scripts/run_chain128_contrib.sh` (contribution studies). Paper tables come from `scripts/final_config_tables.py` and `scripts/paper_tables_nf.py`.

## Notes

- This is research code from a two-machine setup. Chain scripts are kept exactly as they were run (with placeholders for paths and hosts), so they document the experiments but assume that layout.
- The slide-figure scripts (`scripts/paper_*figs*.js`) need a pptxgenjs theme helper that is not included.

## License

To be added before the public release. Files under `toolbox_patch/` derive from rPPG-Toolbox and are distributed under its Responsible AI License v1.1, including its use-based restrictions (`toolbox_patch/LICENSE-RAIL.txt`). Third-party notices: `NOTICE.md`.
