#!/bin/bash
# chain8 완료 마커 대기 → 정본 집계(기존 축 CoTAM, MMPD 전 모델, POS/CHROM) → 논문 v2 표 재생성
export PATH=/SPECPHYS_CONDA/envs/vface-rppg/bin:$PATH
cd /SPECPHYS_ROOT
until [ -f /tmp/chain8.done ] && [ -f /tmp/chain8b.done ] && [ -f /tmp/chain8c.done ]; do sleep 300; done
python scripts/agg_model2.py CoTAM CoTAF_L CoTAM_L
python scripts/agg_mmpd.py base CoTNH CoTA CoTAF CoTAF2 CoTAM CoTAF_L CoTAM_L RhythmFormer RFA FactorizePhys FFA POS CHROM
python scripts/paper_tables_v2.py > /dev/null && echo TABLES_V2_REGENERATED
touch /tmp/post_chain8.done
