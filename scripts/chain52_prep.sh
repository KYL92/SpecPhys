#!/bin/bash
# 체인 52 준비(CPU 만, 34 문서 §4): ①7~33 홀수 트라이얼 변환 ②트라이얼 속성 사전 ③전 660 클립 풀 링크 ④uint8 스트리밍 캐시 mmpd_all_u8 ⑤fold json + 목록. 로그 /tmp/chain52_prep.log, 마커 /tmp/chain52_prep.done
source /SPECPHYS_ROOT/scripts/chain_lib.sh
cd $ROOT; export OMP_NUM_THREADS=2
MEM_GATE=${MEM_GATE:-10}; while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$MEM_GATE" ]; do echo "$(date '+%m-%d %H:%M') MEM_WAIT avail<$MEM_GATE G"; sleep 600; done
echo "$(date '+%m-%d %H:%M') PREP START"
nice -n 10 python scripts/convert_mmpd_odd.py 2>&1 | tail -2
nice -n 10 python scripts/mmpd_trials.py
rm -rf $PFA/COTRAIN_MMPD_ALL; mkdir -p $PFA/COTRAIN_MMPD_ALL
for d in $PFA/MMPD_TRAIN/subject7* $PFA/MMPD_ALL/subject7*; do ln -sfn "$(readlink -f $d)" $PFA/COTRAIN_MMPD_ALL/$(basename $d); done
echo "pool clips: $(ls $PFA/COTRAIN_MMPD_ALL | wc -l)"
nice -n 10 python scripts/nirp_pool_stream.py $PFA/COTRAIN_MMPD_ALL $PFA/cache/mmpd_all_u8 --workers 3 2>&1 | tail -3
python scripts/mmpd_filelist.py folds; for k in 1 2 3 4 5; do python scripts/mmpd_filelist.py test $k; python scripts/mmpd_filelist.py train $k; python scripts/mmpd_filelist.py train $k cotrain_allf_tw_u8; python scripts/mmpd_filelist.py train $k cotrain_allf_hyb4_u8; done
python scripts/mmpd_filelist.py lit; python scripts/mmpd_filelist.py lit cotrain_allf_hyb4_u8
echo "$(date '+%m-%d %H:%M') PREP DONE"; touch /tmp/chain52_prep.done
