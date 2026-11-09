#!/bin/bash
# 10-27 재로그인 후 체인 65(플리커 제거 대조)·66(NF-Phys Phase 2·3) 재개. 두 체인은 멱등: 체크포인트/평가 run 디렉터리가 있으면 건너뛴다.
#  사용: bash scripts/resume_chains_1027.sh      (Claude Code 안에서는 `! bash scripts/resume_chains_1027.sh`)
#  주의(10-27 23:30 사고): /proc 가 hidepid=invisible 이라 `sg video` 로 띄운 체인(실효 gid=video)은 일반 셸(gid=user)의 ps/pgrep 에
#  보이지 않는다 → "죽었다" 고 오판해 체인 65 를 5중 기동한 적 있음. 실행 여부는 반드시 `sg video -c "ps -eo cmd"` 로 보고,
#  체인 스크립트의 flock(/tmp/chainNN.lock) 으로 2중 기동을 막는다. nohup + sg 백그라운드는 호출이 끝나도 살아남는다(ppid 1 로 재부모화).
cd /SPECPHYS_ROOT; P=_workspace/data/physformer
if nvidia-smi --query-gpu=memory.used --format=csv,noheader > /dev/null 2>&1; then PRE=""; echo "GPU ok (groups: $(id -nG))"; else PRE="sg video -c"; echo "GPU 직접 접근 불가(video 그룹 없음) → sg video 로 기동"; fi
# 실행 중 판정: sg video 안의 ps(hidepid 우회) 또는 체인의 flock 보유 여부
running() { sg video -c "ps -eo cmd" 2>/dev/null | grep -q "^bash scripts/run_chain$1.sh" && return 0; flock -n /tmp/chain${1%%_*}.lock true 2>/dev/null || return 0; return 1; }
for c in 65_noflk 66_nf; do n=${c%%_*}
  if running $c; then echo "chain$n already running"; continue; fi
  case $n in 65) art=_workspace/data/results/noflk_judge.txt;; 66) art=_workspace/data/results/nf_judge.txt;; esac
  [ -f /tmp/chain$n.done ] && [ ! -f "$art" ] && { echo "stale /tmp/chain$n.done without $art -> removed"; rm -f /tmp/chain$n.done; }
  [ -f /tmp/chain$n.done ] && { echo "chain$n already done"; continue; }; rm -f /tmp/chain$n.fail
  # 실행 중이 아닐 때만: flock 가드가 없는 체인 스크립트에 삽입(실행 중인 bash 스크립트는 절대 편집하지 않는다)
  grep -q "flock -n 9" scripts/run_chain${c}.sh || { sed -i "/^source .*chain_lib.sh\$/a exec 9>/tmp/chain$n.lock; flock -n 9 || { echo \"chain$n already running (lock) -> exit\"; exit 0; }   # 2중 기동 방지(10-27 사고)" scripts/run_chain${c}.sh; bash -n scripts/run_chain${c}.sh || { echo "chain$n: flock 삽입 후 문법 오류 -> 기동 안 함"; continue; }; echo "chain$n: flock guard inserted"; }
  # 중단으로 결과 없는 평가 run 디렉터리 정리(건너뛰기 오판 방지) — 실행 중이 아닌 체인만, 그 체인의 셀만
  case $n in 65) pat="$P/runs/ev_CoTAF_NOFLK_L_*";; 66) pat="$P/runs/ev_NF_*";; esac
  for d in $pat; do [ -d "$d" ] || continue; ls "$d"/*/saved_test_outputs/*.pickle > /dev/null 2>&1 || { echo "remove incomplete $d"; rm -rf "$d"; }; done
  if [ -n "$PRE" ]; then RPPG_MP_QUOTA=3 RPPG_NUM_WORKERS=4 nohup sg video -c "bash scripts/run_chain${c}.sh" >> /tmp/chain$n.log 2>&1 &
  else RPPG_MP_QUOTA=3 RPPG_NUM_WORKERS=4 nohup bash scripts/run_chain${c}.sh >> /tmp/chain$n.log 2>&1 & fi
  echo "chain$n relaunched (pid $!)"; done
sleep 5; sg video -c "ps -eo pid,etime,cmd" | grep -E "^ *[0-9]+ +[0-9:-]+ +bash scripts/run_chain6[56]" | sort -u -k3 | cut -c1-80; tail -n 2 /tmp/chain65.log /tmp/chain66.log
