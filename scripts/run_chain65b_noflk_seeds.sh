#!/bin/bash
# 체인 65b (48 §5.2, 2026-10-28 02:10): 플리커 제거 코퍼스 셀 CoTAF_NOFLK_L 시드 200·300 (멱등, flock).
#  이유: 시드 1 결과(VIPL 문헌 8.38 vs 공식 3시드 9.00, −0.62)가 공식 모델의 시드 분산(8.08~9.65) 안에 있어 3시드 없이는 판정 불가(CLAUDE.md 규칙 ≥3시드).
#  각 시드: 전처리 → PhysFormer 학습 → 13축 → MMPD 6묶음 → VIPL → 3시드 풀링 판정 `realonly_judge.py CoTAF_NOFLK_L CoTAF_TW_L`.
source /SPECPHYS_ROOT/scripts/chain_lib.sh
exec 9>/tmp/chain65b.lock; flock -n 9 || { echo "chain65b already running (lock) -> exit"; exit 0; }
cd $ROOT; RES=$ROOT/_workspace/data/results
fail() { echo "$(date '+%m-%d %H:%M') $1 -> abort"; touch /tmp/chain65b.fail; exit 1; }
[ "$(ls $PFA/COTRAIN_ALLF_TWNF | wc -l)" = 203 ] && [ "$(find $PFA/COTRAIN_ALLF_TWNF -xtype l | wc -l)" = 0 ] || fail "POOL"
echo "$(date '+%m-%d %H:%M') CHAIN65B START"
for S in 200 300; do M=CoTAF_NOFLK_L_s$S
  gen_yaml $M ALLF_TWNF allf_twnf noflk > /dev/null
  CK=$PFA/runs/finetune_$M/PreTrainedModels/SYNTH_${M}_physformer_Epoch9.pth
  if [ -f "$CK" ]; then eval_only $M allf_twnf noflk; else echo "$(date '+%m-%d %H:%M') TRAIN $M start (seed $S)"; train_eval $M allf_twnf noflk $S; fi
  [ -f "$CK" ] || fail "${M}_TRAIN"
  [ -f $RES/agg_${M}_VIPL.json ] || { python scripts/vipl_yaml.py ev $M "$CK" > /dev/null
    ( export RPPG_CACHE_U8=1; run_gated 6000 $PFA/yaml_VIPL/ev_${M}_VIPL.yaml /tmp/ev_${M}_VIPL.log ${M}_VIPL ); python scripts/agg_vipl.py $M 2>&1 | grep -E "all \[문헌"; }
done
python scripts/realonly_judge.py CoTAF_NOFLK_L CoTAF_TW_L | tee $RES/noflk_judge_3seeds.txt | tail -n 25
echo "$(date '+%m-%d %H:%M') CHAIN65B DONE"; touch /tmp/chain65b.done
