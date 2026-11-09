#!/bin/bash
# 체인 30 — TWM: 정체성(SFHQ 5종, TW와 동일) + 실 PPG 파형 뱅크 + 실 데이터셋 구동 32종 다양화 → CoTAF_TWM_L (단일 요인: 움직임, 기준 TW 10.51 / CoTAF-L 12.19)
set -u; cd /SPECPHYS_ROOT; until [ -f /tmp/chain29.done ]; do sleep 60; done
export PATH=/SPECPHYS_CONDA/envs/vface-gen/bin:$PATH; python scripts/animate_twm.py 2>&1 | grep -E "TWM_" 
source /SPECPHYS_ROOT/scripts/chain_lib.sh
for c in train20h train6hi train6lo; do U=$(echo $c | tr a-z A-Z); rm -rf $PFA/${U}_TWM; python generator/build_synth_dataset.py --config $PFA/configs/${c}_twm.json > /tmp/build_${c}_twm.log 2>&1 || echo BUILD_TWM_FAIL; done
codec_copies TRAIN20H_TWM CMP_SYN_TWM mpeg4; make_pool203 ALLF_TWM TWM; gen_yaml CoTAF_TWM_L ALLF_TWM allf_twm pftwm > /dev/null; echo "CoTAF_TWM_L ALLF_TWM allf_twm pftwm" >> /tmp/pool_registry.txt
train_eval CoTAF_TWM_L allf_twm pftwm; rm -rf $ROOT/_workspace/data/faces/twm_frames; echo CHAIN30_DONE; touch /tmp/chain30.done
