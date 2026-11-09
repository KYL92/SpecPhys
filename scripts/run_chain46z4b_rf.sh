#!/bin/bash
# 체인 46z4b v2: RhythmFormer(NDCHW 수정)·FactorizePhys(72×72 리사이즈 패치 32b) UBFC 가중치 VIPL zero-shot 재실행. 메모리 게이트(가용 ≥18 GB) + 워커 4.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_CACHE_U8=1 RPPG_NUM_WORKERS=4
cd $ROOT; REL=$ROOT/repos/rPPG-Toolbox/final_model_release
run1() { local M=$1 CK=$2 TPL=$3
  [ -f $ROOT/_workspace/data/results/agg_${M}_VIPL.json ] && [ "$M" != FPbase ] && { echo "skip $M"; return; }
  while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 12 ]; do echo "$(date '+%m-%d %H:%M') MEM_WAIT"; sleep 600; done
  rm -rf $PFA/runs/ev_${M}_VIPL; python scripts/vipl_yaml.py ev $M $CK --template $TPL > /dev/null
  echo "$(date '+%m-%d %H:%M') EV $M VIPL start"; run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL
  python scripts/agg_vipl.py $M 2>&1 | grep -E "^---|all "; }
run1 RFbase $REL/UBFC-rPPG_RhythmFormer.pth $PFA/yaml15/ev_RhythmFormer_DLCN.yaml
run1 FPbase $REL/UBFC-rPPG_FactorizePhys_FSAM_Res.pth $PFA/yaml15/ev_FactorizePhys_MCD_A.yaml
echo "$(date '+%m-%d %H:%M') CHAIN46Z4B DONE"; touch /tmp/chain46z4b.done
