#!/bin/bash
# SCALE300: 300 정지 얼굴 → LivePortrait(구동 5종 순환, 20초) → 600프레임 PNG (faces/scale300_frames/f###/)
set -u; export PATH=/SPECPHYS_CONDA/envs/vface-gen/bin:$PATH; cd /SPECPHYS_ROOT
F=_workspace/data/faces/scale300; O=_workspace/data/faces/scale300_anim; FR=_workspace/data/faces/scale300_frames; mkdir -p $O $FR
until [ -s $F/meta.json ]; do sleep 30; done
DR=(d6 d10 d14 d3 d9); n=0; fail=0
for src in $F/f*.png; do b=$(basename $src .png); i=$((10#${b#f})); d=${DR[$((i%5))]}
  if [ ! -d $FR/$b ] || [ $(ls $FR/$b 2>/dev/null | wc -l) -lt 320 ]; then
    [ -s $O/${b}--${d}.mp4 ] || python repos/LivePortrait/inference.py -s $src -d _workspace/data/faces/driving20/$d.mp4 -o $O > /tmp/lp_scale.log 2>&1 || true
    m=$(ls $O/${b}--${d}*.mp4 2>/dev/null | grep -v concat | head -1)
    if [ -n "$m" ]; then mkdir -p $FR/$b; ffmpeg -v error -y -i "$m" -frames:v 600 -start_number 0 $FR/$b/%06d.png; else fail=$((fail+1)); echo "ANIM_FAIL $b"; fi
  fi
  n=$((n+1)); [ $((n%25)) -eq 0 ] && echo "$(date +%H:%M) ANIM $n/300 fail=$fail"
done
echo "ANIM_DONE n=$n fail=$fail"
