#!/bin/bash
# 10-27 23:30 사고 복구: 체인 65(CoTAF_NOFLK_L)가 sg video 아래 5중 기동됨(/proc hidepid 때문에 일반 셸 ps 에 안 보여 "죽었다" 오판 후 반복 기동).
# 다섯 트리 전부 종료(같은 캐시 cotrain_allf_twnf 에 워커 15개가 동시에 쓰는 중이라 하나만 남겨도 중단된 쓰기가 캐시를 깨뜨릴 수 있음)
# → 부분 캐시 삭제 → run_chain65_noflk.sh 에 flock 가드 삽입 → 1회 재기동(약 12 분 전처리 손실). 체인 66(단일 실행)은 건드리지 않는다.
# 사용: bash scripts/fix_chain65_dup_1027.sh   (Claude Code 안에서는 `! bash scripts/fix_chain65_dup_1027.sh`)
set -u; cd /SPECPHYS_ROOT; P=_workspace/data/physformer
SGPS() { sg video -c "ps -eo pid,pgid,etime,cmd"; }
pg=$(SGPS | awk '/bash scripts\/run_chain65_noflk.sh/ {print $2}' | sort -u)
echo "chain65 process groups: $(echo $pg | wc -w) -> $pg"
for g in $pg; do sg video -c "kill -TERM -- -$g"; done
left=0; for i in $(seq 1 20); do sleep 2; left=$(SGPS | grep -E "run_chain65_noflk|yaml_CoTAF_NOFLK_L" | grep -vc grep); [ "$left" = 0 ] && break; done
if [ "$left" != 0 ]; then echo "SIGTERM 후에도 $left 개 남음 -> SIGKILL"; for g in $pg; do sg video -c "kill -KILL -- -$g"; done; sleep 3; fi
echo "remaining chain65 processes: $(SGPS | grep -E 'run_chain65_noflk|yaml_CoTAF_NOFLK_L' | grep -vc grep)"
echo "GPU compute apps now:"; sg video -c "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"
rm -rf "$P/cache/cotrain_allf_twnf" "$P/cache/DataFileLists/cotrain_allf_twnf_0.0_1.0.csv"; echo "partial cache cotrain_allf_twnf dropped"
if ! grep -q "flock -n 9" scripts/run_chain65_noflk.sh; then
  sed -i '/^source .*chain_lib.sh$/a exec 9>/tmp/chain65.lock; flock -n 9 || { echo "chain65 already running (lock) -> exit"; exit 0; }   # 2중 기동 방지(10-27 5중 기동 사고)' scripts/run_chain65_noflk.sh
  bash -n scripts/run_chain65_noflk.sh || { echo "syntax error in run_chain65_noflk.sh -> not relaunched"; exit 1; }; echo "flock guard added to run_chain65_noflk.sh"; fi
RPPG_MP_QUOTA=3 RPPG_NUM_WORKERS=4 nohup sg video -c "bash scripts/run_chain65_noflk.sh" >> /tmp/chain65.log 2>&1 &
echo "chain65 relaunched once (pid $!)"; sleep 8
echo "chain65 trees now: $(SGPS | grep -E 'bash scripts/run_chain65_noflk.sh' | grep -v grep | awk '{print $2}' | sort -u | wc -l) (must be 1)"; tail -n 2 /tmp/chain65.log
