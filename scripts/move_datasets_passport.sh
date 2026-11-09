#!/bin/bash
# 원본 데이터셋 → /SPECPHYS_RAW/passport 이동 (rsync 검증 → 심볼릭 링크 교체 → 원본 삭제). 로그 /tmp/move_passport.log
set -u
DST=/SPECPHYS_RAW/passport/vface_datasets
mkdir -p $DST
move_one() { local SRC=$1 NAME=$2
  echo "$(date '+%m-%d %H:%M') MOVE_START $NAME ($(du -xsh $SRC | cut -f1))"
  rsync -a "$SRC/" "$DST/$NAME/" || { echo "RSYNC_FAIL $NAME"; return 1; }
  # 검증: 파일 수·총 바이트 일치
  local c1=$(find "$SRC" -type f | wc -l) c2=$(find "$DST/$NAME" -type f | wc -l)
  local b1=$(du -xsb "$SRC" | cut -f1) b2=$(du -xsb "$DST/$NAME" | cut -f1)
  if [ "$c1" != "$c2" ] || [ "$b1" != "$b2" ]; then echo "VERIFY_FAIL $NAME files $c1/$c2 bytes $b1/$b2"; return 1; fi
  rm -rf "$SRC" && ln -s "$DST/$NAME" "$SRC" && echo "$(date '+%m-%d %H:%M') MOVE_OK $NAME (링크 교체)"
}
{
echo "$(date '+%m-%d %H:%M') PASSPORT MOVE START"
move_one /SPECPHYS_CACHE/datasets/MR-NIRP  MR-NIRP
move_one /SPECPHYS_CACHE/datasets/mcd_rppg mcd_rppg
move_one /SPECPHYS_CACHE/datasets/PURE     PURE
move_one /SPECPHYS_ROOT/UBFC_dataset UBFC_dataset
df -h / /SPECPHYS_CACHE /SPECPHYS_RAW/passport | tail -3
echo "$(date '+%m-%d %H:%M') PASSPORT MOVE DONE"
} >> /tmp/move_passport.log 2>&1
