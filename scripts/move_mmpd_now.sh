#!/bin/bash
set -u
{
echo "$(date '+%m-%d %H:%M') MMPD MOVE START ($(du -xsh /SPECPHYS_RAW/MMPD | cut -f1))"
mkdir -p /SPECPHYS_RAW/datasets_local
SRC=/SPECPHYS_RAW/MMPD; DSTD=/SPECPHYS_RAW/datasets_local/MMPD
rsync -a "$SRC/" "$DSTD/" || { echo RSYNC_FAIL; exit 1; }
c1=$(find "$SRC" -type f | wc -l); c2=$(find "$DSTD" -type f | wc -l)
b1=$(du -xsb "$SRC" | cut -f1); b2=$(du -xsb "$DSTD" | cut -f1)
if [ "$c1" != "$c2" ] || [ "$b1" != "$b2" ]; then echo "VERIFY_FAIL files $c1/$c2 bytes $b1/$b2"; exit 1; fi
rm -rf "$SRC" && ln -s "$DSTD" "$SRC" && echo "$(date '+%m-%d %H:%M') MMPD MOVE_OK (링크 교체)"
df -h / | tail -1
echo "$(date '+%m-%d %H:%M') MMPD MOVE DONE"
} >> /tmp/move_passport.log 2>&1
