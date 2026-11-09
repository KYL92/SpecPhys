#!/bin/bash
# 체인 71 (51 문서) 복합 변형 2종 순차 생성(CPU). 마커 /tmp/pool_<NAME>.done, 로그 /tmp/build_variants_1029.log
cd /SPECPHYS_ROOT; PY=/SPECPHYS_CONDA/envs/vface-rppg/bin/python
run() { local N=$1; shift; [ -f /tmp/pool_$N.done ] && { echo "skip $N"; return; }; echo "$(date '+%m-%d %H:%M') BUILD $N $*"; nice -n 10 $PY scripts/make_corpus_variant.py $N "$@" > /tmp/build_$N.log 2>&1 && echo "$(date '+%m-%d %H:%M') BUILD_${N}_OK $(grep -o 'POOL ALLF_TW.*' /tmp/build_$N.log | cut -c1-80)" || echo "$(date '+%m-%d %H:%M') BUILD_${N}_FAIL"; }
run COMBO --flicker broad --amp 0.02 --motion 0.5 --drift 0.5
run COMBON --flicker keep --amp 0.02 --motion 0.5 --drift 0.5
echo "$(date '+%m-%d %H:%M') BUILD_ALL_DONE"; touch /tmp/build_variants_1029.done
