#!/usr/bin/env python3
"""[60 문서 §17, 2026-11-03] 추론 시 앙상블 셀: 두 학습 셀 A·B 의 저장 출력(pickle)을 청크별 z-점수 평균해 새 셀 ENS 의 출력으로 저장하고 정본 집계기를 그대로 돌린다(학습 없음).
  python scripts/ensemble_cells.py ENS A B            (예: RF_ENS_DTA_L RF_SPEC_D_L RF_SPEC_TA_L; 시드 접미사는 각 이름에 포함)
축: VIPL(runs/ev_{M}_VIPL/vipl_u8) · MMPD(runs/finetune_{M}/vipl_u8) · 실 축(runs/ev_{M}_{AX}/*/, A·B 공통 축만). 라벨·label_type·fs 는 A 의 것을 쓴다(동일 캐시 평가이므로 동일)."""
import glob, os, pickle, re, subprocess, sys
import numpy as np, torch
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"; ROOT = "/SPECPHYS_ROOT"
ENS, A, B = sys.argv[1:4]
GATE = None
if "--gate" in sys.argv: i = sys.argv.index("--gate"); GATE = (float(sys.argv[i + 1]), float(sys.argv[i + 2]))   # [§19(b)] 청크 라우팅: hr(A)<LO → A, hr(B)>HI → B, else 평균
ADEF = "--gate-adefault" in sys.argv   # [61 §11.1] 라우팅 변형: hr(B)>HI → B, 그 밖은 A (평균 구간 없음; LO 는 쓰지 않음)
sys.path.insert(0, f"{ROOT}/repos/rPPG-Toolbox")
from neural_methods.loss import NFLoss as _NFL
GST = {"A": 0, "B": 0, "avg": 0}
def pk(m, ax):
    g = glob.glob(f"{PF}/runs/finetune_{m}/vipl_u8/saved_test_outputs/*.pickle") if ax == "MMPD" else glob.glob(f"{PF}/runs/ev_{m}_{ax}/*/saved_test_outputs/*.pickle")
    return g[0] if g else None
def z(x):
    x = x.detach().float().cpu() if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), dtype=torch.float32)
    return (x - x.mean()) / (x.std() + 1e-8)
axes = ["VIPL", "MMPD"] + sorted({os.path.basename(d).replace(f"ev_{A}_", "") for d in glob.glob(f"{PF}/runs/ev_{A}_*")} & {os.path.basename(d).replace(f"ev_{B}_", "") for d in glob.glob(f"{PF}/runs/ev_{B}_*")} - {"VIPL"} if True else set())
axes = [a for a in axes if not re.match(r"^s\d+_", a)]   # 시드 접미사 셀의 디렉터리(ev_{A}_s200_*) 제외
done = []
for ax in axes:
    pa, pb = pk(A, ax), pk(B, ax)
    if not pa or not pb: print(f"[ens] skip {ax}: missing {'A' if not pa else 'B'}"); continue
    da, db = pickle.load(open(pa, "rb")), pickle.load(open(pb, "rb"))
    assert set(da["predictions"]) == set(db["predictions"]), f"{ax}: video sets differ"
    pred = {}
    for v in da["predictions"]:
        ca, cb = da["predictions"][v], db["predictions"][v]; assert set(ca) == set(cb), f"{ax}/{v}: chunk sets differ"
        if GATE is None: pred[v] = {c: (z(ca[c]) + z(cb[c])) / 2.0 for c in ca}
        else:
            pred[v] = {}
            for c in ca:
                za, zb = z(ca[c]).reshape(-1), z(cb[c]).reshape(-1)
                ya, yb = (torch.cumsum(za, 0), torch.cumsum(zb, 0)) if da.get("label_type") == "DiffNormalized" else (za, zb)
                fs = float(da.get("fs") or 30); ha = float(_NFL.label_hr_tb(ya[None], fs)[0]); hb = float(_NFL.label_hr_tb(yb[None], fs)[0])
                if ADEF:
                    if hb > GATE[1]: pred[v][c] = zb.reshape(z(cb[c]).shape); GST["B"] += 1
                    else: pred[v][c] = za.reshape(z(ca[c]).shape); GST["A"] += 1
                elif ha < GATE[0]: pred[v][c] = za.reshape(z(ca[c]).shape); GST["A"] += 1
                elif hb > GATE[1]: pred[v][c] = zb.reshape(z(cb[c]).shape); GST["B"] += 1
                else: pred[v][c] = ((za + zb) / 2.0).reshape(z(ca[c]).shape); GST["avg"] += 1
    out = {"predictions": pred, "labels": {v: {c: (l.detach().cpu() if torch.is_tensor(l) else l) for c, l in d.items()} for v, d in da["labels"].items()}, "label_type": da.get("label_type"), "fs": da.get("fs")}
    od = f"{PF}/runs/finetune_{ENS}/vipl_u8/saved_test_outputs" if ax == "MMPD" else (f"{PF}/runs/ev_{ENS}_VIPL/vipl_u8/saved_test_outputs" if ax == "VIPL" else f"{PF}/runs/ev_{ENS}_{ax}/ens/saved_test_outputs")
    os.makedirs(od, exist_ok=True); pickle.dump(out, open(f"{od}/{ENS}_{ax}_outputs.pickle", "wb")); done.append(ax); print(f"[ens] {ax}: {len(pred)} videos ← {os.path.basename(pa)} + {os.path.basename(pb)}")
env = dict(os.environ); env["PATH"] = "/SPECPHYS_CONDA/envs/vface-rppg/bin:" + env["PATH"]
for cmd in ([["python", "scripts/agg_vipl.py", ENS]] if "VIPL" in done else []) + ([["python", "scripts/agg_mmpd_cv2.py", ENS]] if "MMPD" in done else []) + ([["python", "scripts/agg_model2.py", ENS]] if len(done) > 2 else []):
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True); print(f"[ens] {' '.join(cmd[1:])} rc={r.returncode}"); print("\n".join([l for l in r.stdout.splitlines() if "MAE" in l][:3])); 
    if r.returncode: print(r.stderr[-800:])
print(f"ENSEMBLE_DONE {ENS} axes={done}" + (f" gate={GATE} route={GST}" if GATE else ""))
