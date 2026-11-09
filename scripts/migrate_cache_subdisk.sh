#!/bin/bash
# 캐시를 /SPECPHYS_CACHE 로 이전 (체인 38 종료 대기 후 원자적 전환). 로그: /tmp/cache_migrate.log
PF=/SPECPHYS_ROOT/_workspace/data/physformer
DST=/SPECPHYS_CACHE/vface_pf_cache
{
echo "$(date '+%m-%d %H:%M') MIGRATE wait for CHAIN38 DONE"
for i in $(seq 1 720); do grep -q "CHAIN38 DONE" /tmp/chain38.log 2>/dev/null && break; sleep 60; done
sleep 120  # 잔여 파일 핸들 정리 여유
[ -L "$PF/cache" ] && { echo "already symlink — abort"; exit 0; }
echo "$(date '+%m-%d %H:%M') final rsync"
rsync -a --delete "$PF/cache/" "$DST/" && echo RSYNC_OK || { echo RSYNC_FAIL; exit 1; }
mv "$PF/cache" "$PF/cache_localbak" && ln -s "$DST" "$PF/cache" && echo SWITCH_OK || { echo SWITCH_FAIL; exit 1; }
rm -rf "$PF/cache_localbak"
echo "$(date '+%m-%d %H:%M') MIGRATE_DONE $(du -xsh $DST | cut -f1) → sub-disk; 루트 여유 $(df -h / | awk 'NR==2{print $4}')"
} >> /tmp/cache_migrate.log 2>&1
