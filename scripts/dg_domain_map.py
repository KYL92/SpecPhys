#!/usr/bin/env python3
"""[40 문서, 체인 55] 학습 풀(COTRAIN_*)의 subjectN 심볼릭 링크 출처 → 도메인 라벨 json ({chunk_name: domain}).
  python scripts/dg_domain_map.py POOL OUT.json
도메인: ubfc(UBFC 실), syn(가상얼굴 TW: TRAIN20H/6HI/6LO), syn_noise(이중 렌더 잡음 TRAIN24DN*), real_noise(COTRAIN_NOISY_REAL),
        comp(압축 사본 CMP_REAL/CMP_SYN), mcd(MCD_TRAIN), pure(PURE_TRAIN), mmpd(MMPD_TRAIN). 청크 이름 = 'subjectN'(u8 캐시 파일명의 접두)."""
import json, os, sys
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"
RULES = [("UBFC_dataset", "ubfc"), ("TRAIN20H", "syn"), ("TRAIN6HI", "syn"), ("TRAIN6LO", "syn"), ("TRAIN24DN", "syn_noise"),
         ("COTRAIN_NOISY_REAL", "real_noise"), ("CMP_REAL", "comp"), ("CMP_SYN", "comp"), ("MCD_TRAIN", "mcd"), ("PURE_TRAIN", "pure"), ("MMPD_TRAIN", "mmpd")]
pool, out = sys.argv[1], sys.argv[2]
m = {}
for s in sorted(os.listdir(f"{PFA}/COTRAIN_{pool}")):
    t = os.path.realpath(f"{PFA}/COTRAIN_{pool}/{s}"); d = None
    for k, v in RULES:
        if k in t: d = v; break
    if d is None: raise SystemExit(f"unmapped {s} -> {t}")
    m[s] = d
json.dump(m, open(out, "w"), indent=0)
c = {}
for v in m.values(): c[v] = c.get(v, 0) + 1
print(f"{pool}: n={len(m)} domains={c} -> {out}")
