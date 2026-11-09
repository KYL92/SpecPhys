#!/bin/bash
# 체인 60 (45 문서 ⑨): 공개 가중치 4종(EfficientPhys·PhysNet·DeepPhys·TS-CAN) × 13축 평가(전처리 캐시 arch_*, 축마다 삭제) → stress_table.py. 로그 /tmp/chain60.log, 마커 /tmp/chain60.done
source /SPECPHYS_ROOT/scripts/chain_lib.sh
cd $ROOT; ts() { date '+%m-%d %H:%M'; }
AXES="REALTEST RTNI RTNC C3 C6 E3 PURE MCD_A MCD_B DLCN NIRP_STILL NIRP_SMALL NIRP_LARGE"
echo "$(ts) CHAIN60 START"
for M in EfficientPhys PhysNet DeepPhys Tscan; do
  for A in $AXES; do
    [ -d $PFA/runs/ev_${M}_arch_$A/outputs ] && [ -n "$(ls $PFA/runs/ev_${M}_arch_$A/*/saved_test_outputs/*.pickle 2>/dev/null)" ] && { echo "skip $M $A"; continue; }
    y=$(python scripts/arch_yaml.py $M $A); run_gated 6000 $y /tmp/ev_${M}_arch_$A.log ${M}_arch_$A
    grep -q "Traceback" /tmp/ev_${M}_arch_$A.log && echo "$(ts) ${M}_arch_${A}_ERROR: $(grep -m1 -A1 'Error' /tmp/ev_${M}_arch_$A.log | tail -1 | cut -c1-120)"
    drop_cache arch_${M}_$A; done
  python scripts/agg_model2.py ${M}_arch > /tmp/agg_${M}_arch.log 2>&1; echo "$(ts) AGG ${M}_arch: $(grep -E '^(RTNI|RTNC|PURE|DLCN)' /tmp/agg_${M}_arch.log | cut -c1-40 | tr '\n' ';')"
done
python scripts/stress_table.py > /tmp/stress_table.log 2>&1; tail -n 12 /tmp/stress_table.log
echo "$(ts) CHAIN60 DONE"; touch /tmp/chain60.done
