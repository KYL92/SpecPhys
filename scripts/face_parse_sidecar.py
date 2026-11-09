#!/usr/bin/env python3
"""[61 문서 P0] 청크 캐시용 얼굴 부위 분할 사이드카: SegFormer face-parsing(jonathandinu/face-parsing, CelebAMask-HQ 19 부위)으로 청크마다 K 장(균등 간격)을 384 해상도로 분할,
64×64 로 줄인 부위 확률을 {cache}/{name}_parse{i}.npy (uint8, 3×64×64; 0~255) 로 저장:
  [0] 피부(skin+nose) 확률  [1] 배경(background) 확률  [2] 얼굴 내 비피부(눈·눈썹·안경·입·입술) 확률
u8 캐시((L+1,128,128,3) uint8 RGB)와 float Raw 캐시((L,72,72,3) 0~255) 모두 지원. 기존 사이드카는 건너뜀. 단일 GPU 프로세스(데이터 로더 작업자 CUDA 공유 제한 회피).
  python scripts/face_parse_sidecar.py CACHE_DIR [--csv LIST.csv] [--k 8] [--bs 128] [--hc]
[패치 53, 61 문서 §9] --hc: 머리카락·모자·옷(13·14·18) 확률만 {name}_parsehc{i}.npy (uint8 1×64×64) 로 저장(기존 _parse 사이드카는 건드리지 않음)."""
import glob, os, sys, time
import numpy as np, torch, torch.nn.functional as F
MP = "/SPECPHYS_ROOT/_workspace/models/face_parsing"
SKIN, BG, NONSKIN, HC = (1, 2), (0,), (3, 4, 5, 6, 7, 10, 11, 12), (13, 14, 18)
a = sys.argv[1:]; cache = a[0]
K = int(a[a.index("--k") + 1]) if "--k" in a else 8; BS = int(a[a.index("--bs") + 1]) if "--bs" in a else 128
if "--csv" in a:
    import pandas as pd
    files = [f for f in pd.read_csv(a[a.index("--csv") + 1])["input_files"] if os.path.dirname(f).rstrip("/") == cache.rstrip("/")]
else:
    files = sorted(glob.glob(f"{cache}/*_input*.npy"))
HCM = "--hc" in a; SUF = "_parsehc" if HCM else "_parse"
todo = [f for f in files if not os.path.exists(f.replace("_input", SUF))]
print(f"[parse] {cache}: {len(files)} chunks, {len(todo)} to do, K={K}", flush=True)
from transformers import SegformerForSemanticSegmentation
m = SegformerForSemanticSegmentation.from_pretrained(MP).cuda().eval().half()
mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1).half(); std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1).half()
buf, meta = [], []; t0 = time.time(); done = 0
def flush():
    global buf, meta, done
    if not buf: return
    x = torch.from_numpy(np.concatenate(buf, 0)).cuda().permute(0, 3, 1, 2).half() / 255.0
    x = F.interpolate(x, size=(384, 384), mode="bilinear", align_corners=False); x = (x - mean) / std
    with torch.no_grad():
        lab = torch.cat([m(pixel_values=x[i:i + 64]).logits.argmax(1) for i in range(0, len(x), 64)], 0)   # (N,96,96)
    lab = F.interpolate(lab[:, None].float(), size=(64, 64), mode="nearest")[:, 0].long()
    oh = lambda ids: sum((lab == i).float() for i in ids)
    P = oh(HC)[:, None] if HCM else torch.stack([oh(SKIN), oh(BG), oh(NONSKIN)], 1)   # (N,1|3,64,64)
    o = 0
    for f, n in meta:
        p = (P[o:o + n].mean(0) * 255).round().clamp(0, 255).byte().cpu().numpy(); o += n
        np.save(f.replace("_input", SUF), p); done += 1
    buf, meta = [], []
for j, f in enumerate(todo):
    x = np.load(f)
    if x.dtype != np.uint8: x = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    idx = np.linspace(0, len(x) - 1, K).round().astype(int); buf.append(x[idx]); meta.append((f, K))
    if sum(len(b) for b in buf) >= BS: flush()
    if (j + 1) % 2000 == 0: print(f"[parse] {j + 1}/{len(todo)} {((j + 1) * K) / (time.time() - t0):.0f} img/s", flush=True)
flush(); print(f"[parse] DONE {cache}: wrote {done} in {time.time() - t0:.0f}s", flush=True)
