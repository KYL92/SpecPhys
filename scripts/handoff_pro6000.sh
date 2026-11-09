#!/bin/bash
# [54 문서] RTX PRO 6000 머신 이관 패키지 — 이 머신(Alpha)에서 1회 실행 → _workspace/handoff_pro6000/
#   toolbox_local.patch + toolbox_new_files.tar.gz (rPPG-Toolbox 로컬 패치 1~38 기록; 실제 이관은 repos/ 디렉터리 rsync 로 함)
#   pip_freeze_vface-rppg.txt (환경 핀), manifest.tsv (경로·크기·항목 수), rsync_tiers.sh (대상 호스트로 보내는 단계별 rsync), verify_pro6000.sh (대상에서 검증)
# 사용: bash scripts/handoff_pro6000.sh            → 패키지 생성(전송은 하지 않음)
#       bash _workspace/handoff_pro6000/rsync_tiers.sh USER@HOST:/data/vface 0 1 2 3 4   → 단계별 전송(대상 경로는 54 문서의 심볼릭 링크 규약 참조)
set -u
ROOT=/SPECPHYS_ROOT; PFA=$ROOT/_workspace/data/physformer; OUT=$ROOT/_workspace/handoff_pro6000; TB=$ROOT/repos/rPPG-Toolbox
PY=/SPECPHYS_CONDA/envs/vface-rppg/bin/python; PIP=/SPECPHYS_CONDA/envs/vface-rppg/bin/pip
mkdir -p $OUT
echo "[1/4] toolbox patch bundle"
git -C $TB diff > $OUT/toolbox_local.patch
git -C $TB log -1 --format='%H %ad' --date=short > $OUT/toolbox_base_commit.txt
( cd $TB && tar czf $OUT/toolbox_new_files.tar.gz $(git status --short | grep '^??' | cut -c4-) )
git -C $TB status --short > $OUT/toolbox_status.txt
echo "  patched files: $(wc -l < $OUT/toolbox_status.txt), diff lines: $(wc -l < $OUT/toolbox_local.patch)"
echo "[2/4] env pins"
$PIP freeze > $OUT/pip_freeze_vface-rppg.txt
$PY -c "import torch,sys;print('torch',torch.__version__,'cuda',torch.version.cuda,'python',sys.version.split()[0])" > $OUT/env_versions.txt 2>/dev/null
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> $OUT/env_versions.txt 2>/dev/null
echo "[3/4] manifest (du -L follows symlinks)"
printf "tier\tpath\tsize\tentries\tnote\n" > $OUT/manifest.tsv
row() { local tier=$1 p=$2 note=$3; local sz="-" n="-"; [ -e "$p" ] && { sz=$(du -shL "$p" 2>/dev/null | cut -f1); n=$(ls "$p" 2>/dev/null | wc -l); }; printf "%s\t%s\t%s\t%s\t%s\n" "$tier" "$p" "$sz" "$n" "$note" >> $OUT/manifest.tsv; }
row 0 $ROOT/CLAUDE.md "운영 가이드"; row 0 $ROOT/.claude "하네스·스킬(스크립트 포함)"; row 0 $ROOT/scripts "체인·집계·판정 스크립트"; row 0 $ROOT/setup.sh "env 구축"; row 0 $ROOT/verify_env.py "CUDA 스모크"; row 0 $ROOT/onboarding "온보딩 문서"
row 0 $ROOT/_workspace "문서(데이터 제외)"; row 0 $ROOT/_workspace/data/results "집계 json·판정·로그(selfsup_nuis_peaks.json 포함)"; row 0 $TB "rPPG-Toolbox(로컬 패치 적용 상태, final_model_release 포함; runs 제외)"
row 0 $PFA/cache/DataFileLists "툴박스 파일 목록 csv(VIPL 목록 포함)"; row 0 $PFA/noise_gt_hyb2 "DN 교란 사이드카 48"
for y in $PFA/yaml_NF_HYB_L $PFA/yaml_CoTAM_HYB4_L $PFA/yaml24 $PFA/yaml_VIPL $PFA/configs; do row 0 $y "yaml/설정"; done
for r in finetune_NF_HYB_L finetune_NF_HYB_L_s200 finetune_NF_HYB_L_s300 finetune_NF_HYB_ORIG_L finetune_CoTAF_TW_L; do row 0 $PFA/runs/$r/PreTrainedModels "체크포인트"; done
row 0 $PFA/mmpd_trials.json "MMPD trial 조건"; row 0 $PFA/mmpd_folds.json ""; row 0 $PFA/nirp_folds.json ""; row 0 $PFA/vipl_valid_subjects.json ""; row 0 "$PFA/dg_domains_ALLF_HYB4.json" ""
row 1 $PFA/COTRAIN_ALLF_HYB4 "학습 풀 366(심볼릭 링크 → -L 로 실체 복사)"
for d in TEST_C3 TEST_C6 TEST_E3 MCD_A MCD_B PURE_RAW DLCN_TEST REALTEST RTNI RTNC NIRP_STILL NIRP_SMALL NIRP_LARGE; do row 2 $PFA/$d "평가 축"; done
for d in NIRP_GSTILL NIRP_GSMALL NIRP_GLARGE NIRP_STILL940 NIRP_SMALL940 NIRP_GSTILL940 NIRP_GSMALL940 RTNI17 RTNI24; do row 2b $PFA/$d "사후 축(선택)"; done
row 3 $PFA/cache/vipl_u8 "VIPL uint8 캐시(전체 2,378 영상; 재전처리 불필요)"; row 3 /SPECPHYS_RAW/vipl_hr_raw/fold "VIPL fold .mat(agg_vipl 필요)"; row 3 /SPECPHYS_RAW/vipl_hr_raw "VIPL 원본(gt_HR.csv 만 필요; 영상 불필요)"
for b in 1 2 3 4 5 6; do row 4 $PFA/MMPD_B$b "MMPD 묶음(심볼릭 링크 → -L)"; done
row 4 $PFA/MMPD_TRAIN "MMPD 학습 피험자 1–6(풀에 포함; 풀 -L 복사로 충족)"
column -t -s $'\t' $OUT/manifest.tsv | head -60
echo "[4/4] rsync_tiers.sh / verify_pro6000.sh"
cat > $OUT/rsync_tiers.sh <<'EOS'
#!/bin/bash
# 사용: bash rsync_tiers.sh USER@HOST:/data/vface TIER...   (TIER ∈ 0 1 2 2b 3 4; 대상의 /data/vface 아래에 Virtual-Face/, sub-disk/ 가 생긴다)
# 대상 머신에서는 54 문서 규약대로 /SPECPHYS_ROOT → /data/vface/Virtual-Face, /SPECPHYS_CACHE → /data/vface/sub-disk 심볼릭 링크를 만든다.
set -u; DEST=$1; shift; SRC=/SPECPHYS_ROOT; PFA=$SRC/_workspace/data/physformer
RS="rsync -aH --partial --info=progress2"; RSL="rsync -aHL --partial --info=progress2"   # -L: 심볼릭 링크를 실체로
for T in "$@"; do case $T in
 0) $RS --exclude '_workspace/data' --exclude '.claude/worktrees' $SRC/CLAUDE.md $SRC/setup.sh $SRC/verify_env.py $SRC/.claude $SRC/scripts $SRC/onboarding $SRC/_workspace $DEST/Virtual-Face/
    $RS --exclude 'runs' --exclude '__pycache__' $SRC/repos/rPPG-Toolbox $DEST/Virtual-Face/repos/
    mkdir -p /tmp/handoff_pfa && $RS $SRC/_workspace/data/results $DEST/Virtual-Face/_workspace/data/
    $RS --include 'yaml*/' --include 'yaml*/**' --include 'configs/***' --include 'noise_gt_hyb2/***' --include '*.json' --exclude '*' $PFA/ $DEST/Virtual-Face/_workspace/data/physformer/
    $RS $PFA/cache/DataFileLists $DEST/sub-disk/vface_pf_cache/
    for r in finetune_NF_HYB_L finetune_NF_HYB_L_s200 finetune_NF_HYB_L_s300 finetune_NF_HYB_ORIG_L finetune_CoTAF_TW_L; do $RS --relative $PFA/./runs/$r/PreTrainedModels $DEST/Virtual-Face/_workspace/data/physformer/; done ;;
 1) $RSL $PFA/COTRAIN_ALLF_HYB4 $DEST/Virtual-Face/_workspace/data/physformer/ ;;
 2) for d in TEST_C3 TEST_C6 TEST_E3 MCD_A MCD_B PURE_RAW DLCN_TEST REALTEST RTNI RTNC NIRP_STILL NIRP_SMALL NIRP_LARGE; do $RSL $PFA/$d $DEST/Virtual-Face/_workspace/data/physformer/; done ;;
 2b) for d in NIRP_GSTILL NIRP_GSMALL NIRP_GLARGE NIRP_STILL940 NIRP_SMALL940 NIRP_GSTILL940 NIRP_GSMALL940 RTNI17 RTNI24; do $RSL $PFA/$d $DEST/Virtual-Face/_workspace/data/physformer/; done ;;
 3) $RSL $PFA/cache/vipl_u8 $DEST/sub-disk/vface_pf_cache/
    $RS -m --include '*/' --include 'fold/***' --include 'gt_HR.csv' --include '*.txt' --exclude '*' /SPECPHYS_RAW/vipl_hr_raw $DEST/sub-disk/ ;;
 4) for b in 1 2 3 4 5 6; do $RSL $PFA/MMPD_B$b $DEST/Virtual-Face/_workspace/data/physformer/; done ;;
 *) echo "unknown tier $T";; esac; done
EOS
cat > $OUT/verify_pro6000.sh <<EOS
#!/bin/bash
# 대상 머신(PRO 6000)에서: bash verify_pro6000.sh  → 모든 줄이 OK 여야 체인 72 기동 가능
ROOT=/SPECPHYS_ROOT; PFA=\$ROOT/_workspace/data/physformer; PY=/SPECPHYS_CONDA/envs/vface-rppg/bin/python; ok=0; bad=0
chk() { if eval "\$2"; then echo "OK   \$1"; ok=\$((ok+1)); else echo "FAIL \$1"; bad=\$((bad+1)); fi; }
chk "경로 규약 \$ROOT" "[ -d \$ROOT/scripts ] && [ -f \$ROOT/CLAUDE.md ]"
chk "miniforge env python" "[ -x \$PY ]"
chk "torch cu128+ & Blackwell(sm_120) 커널" "\$PY -c 'import torch;assert torch.cuda.is_available();cap=torch.cuda.get_device_capability();print(torch.__version__,torch.version.cuda,cap);x=torch.ones(2,device=\"cuda\");assert float((x@x))==2.0' 2>/dev/null"
chk "nvidia-smi" "nvidia-smi -L > /dev/null 2>&1"
chk "toolbox 로컬 패치 상태(수정 17 + 신규 4)" "[ \"\$(git -C \$ROOT/repos/rPPG-Toolbox status --short | wc -l)\" = 21 ]"
chk "패치 38 존재(NF-SS)" "grep -q 'NF_SS_TRACE' \$ROOT/repos/rPPG-Toolbox/neural_methods/trainer/FactorizePhysTrainer.py && grep -q 'def motion_traces' \$ROOT/repos/rPPG-Toolbox/neural_methods/loss/NFLoss.py"
chk "공개 FactorizePhys 가중치" "[ -f \$ROOT/repos/rPPG-Toolbox/final_model_release/UBFC-rPPG_FactorizePhys_FSAM_Res.pth ]"
chk "cache 심볼릭 → sub-disk" "[ -d \$PFA/cache/DataFileLists ] && [ -L \$PFA/cache ]"
chk "학습 풀 366(깨진 링크 0)" "[ \"\$(ls \$PFA/COTRAIN_ALLF_HYB4 | wc -l)\" = 366 ] && [ \"\$(find \$PFA/COTRAIN_ALLF_HYB4 -xtype l | wc -l)\" = 0 ]"
chk "DN 사이드카 48" "[ \"\$(ls \$PFA/noise_gt_hyb2 | wc -l)\" = 48 ]"
$(for d in REALTEST:8 MCD_A:20 MCD_B:20 PURE_RAW:59 DLCN_TEST:20 RTNI:8 RTNC:8 NIRP_STILL:19 NIRP_SMALL:19 NIRP_LARGE:19 TEST_C3:5 TEST_C6:5 TEST_E3:5 MMPD_B1:6 MMPD_B2:6 MMPD_B3:6 MMPD_B4:6 MMPD_B5:6 MMPD_B6:3; do echo "chk \"축 ${d%%:*} 항목 ${d##*:}\" \"[ \\\"\\\$(ls \\\$PFA/${d%%:*} 2>/dev/null | wc -l)\\\" = ${d##*:} ]\""; done)
chk "VIPL u8 캐시 35380 파일" "[ \"\$(ls \$PFA/cache/vipl_u8 | wc -l)\" = 35380 ]"
chk "VIPL 파일 목록 csv" "[ -f \$PFA/cache/DataFileLists/vipl_all.csv ]"
chk "VIPL fold .mat" "[ -f /SPECPHYS_RAW/vipl_hr_raw/fold/fold1.mat ]"
chk "Phase 0 피크 캐시" "[ -f \$ROOT/_workspace/data/results/selfsup_nuis_peaks.json ]"
chk "기준 집계 json(NF_HYB_L 3시드 VIPL·MMPD)" "[ -f \$ROOT/_workspace/data/results/agg_NF_HYB_L_s300_VIPL.json ] && [ -f \$ROOT/_workspace/data/results/agg_NF_HYB_L_s300_MMPD.json ]"
chk "디스크 여유 ≥300 GB(캐시용, cache 위치)" "[ \"\$(df -BG \$(readlink -f \$PFA/cache) | awk 'NR==2{print \$4+0}')\" -ge 300 ]"
chk "yaml 템플릿(CoTAM_HYB4_L·yaml24·VIPL)" "[ -f \$PFA/yaml_CoTAM_HYB4_L/finetune_CoTAM_HYB4_L.yaml ] && [ -d \$PFA/yaml24 ] && [ -d \$PFA/yaml_VIPL ]"
echo "== OK \$ok / FAIL \$bad"; [ \$bad = 0 ]
EOS
chmod +x $OUT/rsync_tiers.sh $OUT/verify_pro6000.sh
echo "-> $OUT: $(ls $OUT | tr '\n' ' ')"
