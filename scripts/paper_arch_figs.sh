#!/bin/bash
# [63 문서] 구조 그림 PPT 생성: JS(도형) → 도형 묶기 → LibreOffice 렌더(PDF 미리보기, Docker 이미지 lo-render:latest)
# NODE_PATH 에 pptxgenjs 가 있어야 한다. 재료를 새로 뽑으려면 먼저 python scripts/paper_arch_figs_data.py
set -e; ROOT=/SPECPHYS_ROOT; OUT=$ROOT/_workspace/slides; PY=/SPECPHYS_CONDA/envs/vface-rppg/bin/python
export NODE_PATH=${NODE_PATH:-/SPECPHYS_TMP/node/node_modules}
cd $ROOT/scripts; node paper_arch_figs.js $OUT/specphys_architecture_figs.pptx; $PY paper_arch_figs_group.py $OUT/specphys_architecture_figs.pptx
cd $OUT; timeout 170 docker run --rm --network none -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/work -w /work lo-render:latest --headless --convert-to pdf specphys_architecture_figs.pptx > /dev/null 2>&1
ls -la specphys_architecture_figs.*
