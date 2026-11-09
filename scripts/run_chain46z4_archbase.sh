#!/bin/bash
# 체인 46z4 (33 문서 §5, 10-17): 기존 방법 공식 UBFC 가중치(PhysFormer·RhythmFormer·FactorizePhys)의 VIPL zero-shot 평가(2,378 영상, uint8 공유 캐시; 읽기 = 라벨 유형별 툴박스 규칙)
source /SPECPHYS_ROOT/scripts/chain_lib.sh
export RPPG_CACHE_U8=1
cd $ROOT; REL=$ROOT/repos/rPPG-Toolbox/final_model_release
[ "$(ls $PFA/cache/vipl_u8/*_u8stats.npy 2>/dev/null | wc -l)" -ge 2378 ] || { echo "u8stats missing -> build"; python scripts/vipl_u8_stats.py 2>&1 | tail -1; }
run1() { local M=$1 CK=$2 TPL=$3
  [ -f $ROOT/_workspace/data/results/agg_${M}_VIPL.json ] && { echo "skip $M"; return; }
  if [ -n "$TPL" ]; then python scripts/vipl_yaml.py ev $M $CK --template $TPL > /dev/null; else python scripts/vipl_yaml.py ev $M $CK > /dev/null; fi
  echo "$(date '+%m-%d %H:%M') EV $M VIPL start"; run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL
  python scripts/agg_vipl.py $M 2>&1 | grep -E "^---|all "; }
run1 PFbase $REL/UBFC-rPPG_PhysFormer_DiffNormalized.pth ""
run1 RFbase $REL/UBFC-rPPG_RhythmFormer.pth $PFA/yaml15/ev_RhythmFormer_DLCN.yaml
run1 FPbase $REL/UBFC-rPPG_FactorizePhys_FSAM_Res.pth $PFA/yaml15/ev_FactorizePhys_MCD_A.yaml
echo "$(date '+%m-%d %H:%M') CHAIN46Z4 DONE"; touch /tmp/chain46z4.done
