#!/usr/bin/env python3
"""[옵션 A, 50 문서 §2b] MR-NIRP Car 추가 RGB 축 변환 — convert_nirp.py 의 변환 함수 재사용(975/940 nm 세션의 RGB 카메라, Bayer→8bit 320², PulseOx 보간, 앞 60 초).
  python scripts/convert_nirp_extra.py A   → 주차장 975: NIRP_GSTILL / NIRP_GSMALL / NIRP_GLARGE
  python scripts/convert_nirp_extra.py B   → 940 nm: NIRP_STILL940 / NIRP_SMALL940(주행), NIRP_GSTILL940 / NIRP_GSMALL940(주차장)
마커 /tmp/nirp_extra_<group>.done"""
import os, sys
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from convert_nirp import convert, PF  # noqa
GROUPS = {"A": {"garage_still_975": "NIRP_GSTILL", "garage_small_motion_975": "NIRP_GSMALL", "garage_large_motion_975": "NIRP_GLARGE"},
          "B": {"driving_still_940": "NIRP_STILL940", "driving_small_motion_940": "NIRP_SMALL940", "garage_still_940": "NIRP_GSTILL940", "garage_small_motion_940": "NIRP_GSMALL940"}}
g = sys.argv[1]
for cond, out in GROUPS[g].items():
    os.makedirs(os.path.join(PF, out), exist_ok=True)
    for s in range(1, 20):
        try: convert(s, cond, os.path.join(PF, out))
        except Exception as e:  # 10-28: subject2_garage_small_motion_940 RGB.zip 에 빈 .pgm 항목 → cv2.imdecode 실패; 해당 피험자만 건너뛰고 기록
            print(f"[nirp-extra] SKIP subject{s}_{cond}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            d = os.path.join(PF, out, f"subject{s}")
            if os.path.isdir(d) and not os.path.exists(os.path.join(d, "vid.avi")): os.rmdir(d)
    print(f"[nirp-extra] {out}: {len(os.listdir(os.path.join(PF, out)))} subjects", flush=True)
open(f"/tmp/nirp_extra_{g}.done", "w").write("ok\n"); print(f"[nirp-extra] group {g} done")
