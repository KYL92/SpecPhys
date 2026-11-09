#!/bin/bash
# 체인 69 (50 문서) 코퍼스 변형 6종 순차 생성(CPU, 외장 디스크 I/O 병목이라 병렬 2개 이하). 마커 /tmp/pool_<NAME>.done, 로그 /tmp/build_variants.log
cd /SPECPHYS_ROOT; PY=/SPECPHYS_CONDA/envs/vface-rppg/bin/python
run() { local N=$1; shift; [ -f /tmp/pool_$N.done ] && { echo "skip $N"; return; }; echo "$(date '+%m-%d %H:%M') BUILD $N $*"; nice -n 10 $PY scripts/make_corpus_variant.py $N "$@" > /tmp/build_$N.log 2>&1 && echo "$(date '+%m-%d %H:%M') BUILD_${N}_OK $(grep -o 'POOL ALLF_TW.*' /tmp/build_$N.log | cut -c1-80)" || echo "$(date '+%m-%d %H:%M') BUILD_${N}_FAIL"; }
run FR03 --flicker broad --amp 0.03
run MIX --flicker mix --amp 0.03
run N02 --flicker keep --amp 0.02
run FR04 --flicker broad --amp 0.04
run WIDE --motion 2.0 --drift 1.5
run NARROW --motion 0.5 --drift 0.5
echo "$(date '+%m-%d %H:%M') BUILD_ALL_DONE"; touch /tmp/build_variants.done
