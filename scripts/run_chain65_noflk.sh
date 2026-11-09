#!/bin/bash
# 체인 65 (48 문서, 2026-10-27): 플리커 제거 대조 풀 COTRAIN_ALLF_TWNF(203; 공식 코퍼스와 클립 단위 일치, 플리커 항만 0, scripts/make_noflk_pool.py)
#  → 채택 레시피 PhysFormer 셀 CoTAF_NOFLK_L(시드 100) 13축 + MMPD + VIPL zero-shot → vs 공식 모델 CoTAF_TW_L(시드 풀링).
#  사용자 질문(10-27): "코퍼스가 플리커에 치우쳐 VIPL 같은 도메인에 악영향?" 의 직접 검정. 사전 등록 H-F1: VIPL Δ(NOFLK−공식) < −0.3 이면 지지.
#  로그 /tmp/chain65.log, 마커 /tmp/chain65.done. 풀 마커 /tmp/noflk_pool.done 대기.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
exec 9>/tmp/chain65.lock; flock -n 9 || { echo "chain65 already running (lock) -> exit"; exit 0; }   # 2중 기동 방지(10-27 5중 기동 사고)
cd $ROOT; RES=$ROOT/_workspace/data/results; M=CoTAF_NOFLK_L
fail() { echo "$(date '+%m-%d %H:%M') $1 -> abort"; touch /tmp/chain65.fail; exit 1; }
pool_ok() { [ -d $PFA/COTRAIN_ALLF_TWNF ] && [ "$(ls $PFA/COTRAIN_ALLF_TWNF 2>/dev/null | wc -l)" = 203 ] && [ "$(find $PFA/COTRAIN_ALLF_TWNF -xtype l | wc -l)" = 0 ]; }   # 10-27: /tmp 마커·로그는 재부팅에 지워지므로 풀 자체를 검사
until [ -f /tmp/noflk_pool.done ] || pool_ok; do sleep 60; done
pool_ok || fail "POOL_BUILD"
echo "$(date '+%m-%d %H:%M') CHAIN65 START pool=$(ls $PFA/COTRAIN_ALLF_TWNF | wc -l) broken=$(find $PFA/COTRAIN_ALLF_TWNF -xtype l | wc -l)"
gen_yaml $M ALLF_TWNF allf_twnf noflk > /dev/null
CK=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
if [ -f "$CK" ]; then eval_only $M allf_twnf noflk; else train_eval $M allf_twnf noflk 100; fi
[ -f "$CK" ] || fail "${M}_TRAIN"
[ -f $RES/agg_${M}_VIPL.json ] || { python scripts/vipl_yaml.py ev $M "$CK" > /dev/null
  ( export RPPG_CACHE_U8=1; run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL ); python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; }
python scripts/realonly_judge.py $M CoTAF_TW_L | tee $RES/noflk_judge.txt | tail -n 25   # json → results/realonly_judge_${M}_vs_CoTAF_TW_L.json
echo "$(date '+%m-%d %H:%M') CHAIN65 DONE"; touch /tmp/chain65.done
