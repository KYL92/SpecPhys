# rPPG-Toolbox base

- Upstream: https://github.com/ubicomplab/rPPG-Toolbox
- Commit: `b7500b848f84ad7f86e277b4612563b69f4f88f9` (2025-09-15)
- `rppg_toolbox.patch`: changes to existing files (loaders, trainers, models, losses, post-processing, `main.py`, the Mamba build script for sm_120).
- `new_files/`: files added to the toolbox (`FactorizePhysNF.py`, `NFLoss.py`, `VIPLHRLoader.py`, `dg_utils.py`).
- License: Responsible AI License v1.1 (`LICENSE-RAIL.txt`); these modifications are distributed under the same terms.
- The environment variables that switch each modification on are listed in `../docs/toolbox_patches_and_protocol.md`.
