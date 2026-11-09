#!/bin/bash
# 체인 24~29 순차 게이트 (사용자 지시 2026-08-30, 융합 제외): 24 트윈 분기 → 25 NIRP 트윈 → 27 동적 검출 → 28 MMPD 3-fold CV → 29 다중 시드
set -u; source /SPECPHYS_ROOT/scripts/chain_lib.sh; cd $ROOT; REG=/tmp/pool_registry.txt; touch $REG
extract_frames_r() { python - <<'PY'
import json,os,subprocess
F="/SPECPHYS_ROOT/_workspace/data/faces"; n=0
for q in json.load(open(f"{F}/scale300r_pairs.json")):
    fd=q["frames"]
    if os.path.isdir(fd) and len(os.listdir(fd))>=300: continue
    mp4=f"{F}/scale300r_anim/f{q['idx']:03d}--{q['driving']}.mp4"; os.makedirs(fd,exist_ok=True)
    subprocess.run(["ffmpeg","-v","error","-y","-i",mp4,"-frames:v","600","-start_number","0",f"{fd}/%06d.png"],capture_output=True); n+=1
print("FRAMES_R extracted",n)
PY
}
# MMPD 7~33 변환(짝수 트라이얼)은 별도 백그라운드에서 진행 중 → /tmp/mmpd_all.done 대기
until [ -f /tmp/chain23.done ]; do sleep 60; done
# ---------- 체인 24 ----------
BR=$(python scripts/decide_branch.py CoTAF_T_L | cut -d' ' -f1); echo "CHAIN24_BRANCH $BR $(python scripts/decide_branch.py CoTAF_T_L)"
if [ "$BR" = "IMPROVED" ]; then
  BASE=ALLF_T; echo "CoTAF_T_L ALLF_T allf_t pft" >> $REG
  for V in "TW 1 0 mpeg4" "TS 0 1 mpeg4" "TC 0 0 sweep"; do set -- $V; NAME=$1; python scripts/make_twin_variant.py $NAME $2 $3
    for c in train20h train6hi train6lo; do U=$(echo $c | tr a-z A-Z); rm -rf $PFA/${U}_$NAME; python generator/build_synth_dataset.py --config $PFA/configs/${c}_$(echo $NAME | tr A-Z a-z).json > /tmp/build_${c}_$NAME.log 2>&1 || echo BUILD_${NAME}_FAIL; done
    codec_copies TRAIN20H_$NAME CMP_SYN_$NAME $4; make_pool203 ALLF_$NAME $NAME; l=$(echo $NAME | tr A-Z a-z)
    gen_yaml CoTAF_${NAME}_L ALLF_$NAME allf_$l pf$l > /dev/null; echo "CoTAF_${NAME}_L ALLF_$NAME allf_$l pf$l" >> $REG; train_eval CoTAF_${NAME}_L allf_$l pf$l; done
else
  BASE=ALLF; extract_frames_r; python scripts/make_syn_twin_configs.py RAT
  rm -rf $PFA/SYN_S300RAT; python generator/build_synth_dataset.py --config $PFA/configs/syn_rat.json > /tmp/build_syn_rat.log 2>&1 || echo BUILD_RAT_FAIL
  make_pool ALLF_S300RAT $PFA/COTRAIN_ALLF_T $PFA/SYN_S300RAT; gen_yaml CoTAF_S300RAT_L ALLF_S300RAT allf_s300rat pfrat > /dev/null; echo "CoTAF_S300RAT_L ALLF_S300RAT allf_s300rat pfrat" >> $REG; train_eval CoTAF_S300RAT_L allf_s300rat pfrat
fi
echo CHAIN24_DONE; touch /tmp/chain24.done
# ---------- 체인 25 (MR-NIRP 야간 실차 트윈) ----------
extract_frames_r; python scripts/make_syn_twin_configs.py NIRPT 100
rm -rf $PFA/SYN_NIRPT; python generator/build_synth_dataset.py --config $PFA/configs/syn_nirpt.json > /tmp/build_syn_nirpt.log 2>&1 || echo BUILD_NIRPT_FAIL
make_pool ALLF_N $PFA/COTRAIN_$BASE $PFA/SYN_NIRPT; gen_yaml CoTAF_N_L ALLF_N allf_n pfn > /dev/null; echo "CoTAF_N_L ALLF_N allf_n pfn" >> $REG; train_eval CoTAF_N_L allf_n pfn
rm -rf $ROOT/_workspace/data/faces/scale300r_frames; echo CHAIN25_DONE; touch /tmp/chain25.done
# (체인 26 융합은 사용자 지시로 제외 — 분포 채우기 = 개별 모델 성능이 주제) 트윈 승자만 기록
W=$(python - <<'PY'
import json,os,numpy as np
R="/SPECPHYS_ROOT/_workspace/data/results"; best=None
for m in ["CoTAF_T_L","CoTAF_TW_L","CoTAF_TS_L","CoTAF_TC_L","CoTAF_S300RAT_L","CoTAF_N_L","CoTAF_S300RA_L"]:
    p=f"{R}/agg_{m}_MMPD.json"
    if not os.path.exists(p): continue
    v=np.mean([r["err"] for r in json.load(open(p)) if r["subject"]>=7]); best=(v,m) if best is None or v<best[0] else best
print(best[1])
PY
); echo "TWIN_WINNER $W"; echo $W > /tmp/twin_winner.txt
# ---------- 체인 27 (동적 얼굴 검출 전처리, 학습 없음) ----------
gen_yaml CoTAF_L_DYN ALLF allf pfdyn CoTAF_L 1 > /dev/null; eval_only CoTAF_L_DYN allf pfdyn; for k in 1 2 3 4 5 6; do drop_cache mmpd_pfdyn_B$k; done
for ax in NIRP_STILL NIRP_SMALL NIRP_LARGE MCD_A DLCN; do drop_cache dyn_$ax; done
echo CHAIN27_DONE; touch /tmp/chain27.done
# ---------- 체인 28 (MMPD 피험자 분리 3-fold CV, 피부톤 5·6 상한) ----------
until [ -f /tmp/mmpd_all.done ]; do sleep 60; done
for k in 1 2 3; do P=ALLF_MCV$k; rm -rf $PFA/COTRAIN_$P; mkdir -p $PFA/COTRAIN_$P; for s in $PFA/COTRAIN_ALLF/subject*; do ln -sfn "$(readlink -f $s)" "$PFA/COTRAIN_$P/$(basename $s)"; done
  python scripts/mmpd_fold_links.py $P $k; echo "POOL_$P n=$(ls $PFA/COTRAIN_$P | wc -l)"
  gen_yaml CoTAF_MCV${k}_L $P allf_mcv$k pfmcv$k > /dev/null; echo "CoTAF_MCV${k}_L $P allf_mcv$k pfmcv$k" >> $REG; train_eval CoTAF_MCV${k}_L allf_mcv$k pfmcv$k; done
python scripts/agg_mmpd_cv.py > /tmp/agg_mmpd_cv.log 2>&1; echo CHAIN28_DONE; touch /tmp/chain28.done
# ---------- 체인 29 (다중 시드: CoTAF-L + 트윈 승자) ----------
read WM WP WC WE < <(grep "^$W " $REG | tail -1); [ -z "${WP:-}" ] && { WP=ALLF_T; WC=allf_t; WE=pft; }
for SEED in 200 300; do
  gen_yaml CoTAF_L_s$SEED ALLF allf pfl$SEED > /dev/null; train_eval CoTAF_L_s$SEED allf pfl$SEED $SEED
  gen_yaml ${W}_s$SEED $WP $WC ${WE}s$SEED > /dev/null; train_eval ${W}_s$SEED $WC ${WE}s$SEED $SEED
  done
echo CHAIN29_DONE; touch /tmp/chain29.done; echo ALL_CHAINS_DONE
