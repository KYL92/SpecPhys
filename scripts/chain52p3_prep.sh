#!/bin/bash
# 체인 52 Phase 3 준비(34 문서 §4 규칙·CLAUDE.md '쌍대 비교 ≥3 시드'): 10-19~20 디스크 정리로 삭제된 u8 캐시 3개 재생성
#  mmpd_all_u8(660 클립, 스트리밍)·cotrain_allf_tw_u8·cotrain_allf_hyb4_u8 — 목록 csv 는 같은 경로를 가리키므로 그대로 유효. 마커 /tmp/chain52p3_prep.done
source /SPECPHYS_ROOT/scripts/chain_lib.sh
cd $ROOT; MEM_GATE=${MEM_GATE:-10}
gate() { while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$MEM_GATE" ]; do echo "$(date '+%m-%d %H:%M') MEM_WAIT"; sleep 600; done; wait_disk; }
echo "$(date '+%m-%d %H:%M') CHAIN52P3 PREP START"
if [ ! -f $PFA/cache/DataFileLists/mmpd_all_u8_0.0_1.0.csv ] || [ ! -d $PFA/cache/mmpd_all_u8 ]; then gate; nice -n 10 python scripts/nirp_pool_stream.py $PFA/COTRAIN_MMPD_ALL $PFA/cache/mmpd_all_u8 --workers 3 2>&1 | tail -2; fi
for p in ALLF_TW:cotrain_allf_tw_u8 ALLF_HYB4:cotrain_allf_hyb4_u8; do pool=${p%%:*}; cn=${p##*:}
  [ -f $PFA/cache/DataFileLists/${cn}_0.0_1.0.csv ] && [ -d $PFA/cache/$cn ] && continue
  gate; RPPG_MP_QUOTA=3 nice -n 10 python scripts/pool_preprocess_u8.py COTRAIN_$pool $cn > /tmp/pool_$cn.log 2>&1; tail -1 /tmp/pool_$cn.log; done
n=$(ls $PFA/cache/mmpd_all_u8 2>/dev/null | wc -l); echo "$(date '+%m-%d %H:%M') mmpd_all_u8 files=$n"; [ "$n" -gt 7000 ] || { echo "PREP_FAIL mmpd_all_u8"; touch /tmp/chain52p3.fail; exit 1; }
echo "$(date '+%m-%d %H:%M') CHAIN52P3 PREP DONE"; touch /tmp/chain52p3_prep.done
