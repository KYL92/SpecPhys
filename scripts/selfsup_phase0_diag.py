#!/usr/bin/env python3
"""[53 문서 Phase 0] 자기 지도 분리 연구 사전 진단(학습 없음) — 코퍼스 모델(NF_HYB_L 3시드)이 약한 실 구간의 실패가
'분리 실패(교란 피크 lock-in)' 인지 '사전 분포 붕괴(심박이 코퍼스 전형 대역으로 끌림)' 인지 영상 단위로 분류한다.
  표적: VIPL 운동 후 v7 + 심박 ≥90 bpm 영상(u8 캐시 청크에서 전역 밝기 trace g·전역 이동 속도 trace vx/vy(Lucas–Kanade, 부호 있음) 복원, 30 fps),
        MMPD held-out(7–33) 움직임(회전·대화·보행)·운동 후 영상(원본 .mat, 첫 프레임 Haar 박스), MR-NIRP 대동작 19편(vid.avi).
  영상별 교란 프록시 스펙트럼(0.5–3.5 Hz, 8× 영패딩; 1 s 이동평균 제거) 의 피크(최대의 30% 이상 국소 극대) 를 구해, 모델별 예측 HR 을 분류:
    ok(|err|≤5) | half/double(예측 ≈ GT/2 또는 2·GT) | lockin(예측이 교란 피크 ±3 bpm, 그 피크는 GT 에서 >5 bpm) |
    prior(예측 55–95 bpm & GT ≥100, 교란 피크 무관) | other.   부가: nuis_at_hr(교란 지배 피크가 GT ±5 bpm → 스펙트럼 분리 불가 구간).
  python scripts/selfsup_phase0_diag.py [--models NF_HYB_L FPbase CoTAF_TW_L] [--skip-mmpd]
  → results/selfsup_phase0.{json,txt} (분류 표), results/selfsup_nuis_peaks.json (영상별 피크 캐시; selfsup_judge.py 가 새 셀에 재사용)"""
import glob, json, os, re, sys, time
import numpy as np

PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; R = f"{PFA}/../results"
FS = 30.0; LO, HI = 0.5, 3.5
MODELS = {"VIPL": {"NF_HYB_L": ["NF_HYB_L", "NF_HYB_L_s200", "NF_HYB_L_s300"], "FPbase": ["FPbase"], "CoTAF_TW_L": ["CoTAF_TW_L", "CoTAF_TW_L_s200", "CoTAF_TW_L_s300"]},
          "MMPD": {"NF_HYB_L": ["NF_HYB_L", "NF_HYB_L_s200", "NF_HYB_L_s300"], "FPbase": ["FactorizePhys"], "CoTAF_TW_L": ["CoTAF_TW_L", "CoTAF_TW_L_s200", "CoTAF_TW_L_s300"]},
          "NIRP_LARGE": {"NF_HYB_L": ["NF_HYB_L", "NF_HYB_L_s200", "NF_HYB_L_s300"], "FPbase": ["FactorizePhys"], "CoTAF_TW_L": ["CoTAF_TW_L", "CoTAF_TW_L_s200", "CoTAF_TW_L_s300"]}}
HAAR = "/SPECPHYS_ROOT/repos/rPPG-Toolbox/dataset/haarcascade_frontalface_default.xml"


def hp(x, k=31):
    k = int(k); pad = k // 2; xp = np.pad(x, (pad, pad), mode="edge"); c = np.cumsum(np.insert(xp, 0, 0.0))
    return x - (c[k:] - c[:-k]) / k


def spectrum(x, fs=FS):
    x = hp(np.asarray(x, dtype=np.float64)); x = x - x.mean(); n = len(x); nfft = 8 * n
    P = np.abs(np.fft.rfft(x * np.hanning(n), n=nfft)) ** 2; f = np.fft.rfftfreq(nfft, 1.0 / fs)
    sel = (f >= LO) & (f <= HI); return f[sel] * 60.0, P[sel]


def peaks(bpm, P, rel=0.3, min_sep=4.0):
    """local maxima ≥ rel·max, merged within min_sep bpm; sorted by power desc → [(bpm, rel_power)]"""
    if P.max() <= 0: return []
    idx = [i for i in range(1, len(P) - 1) if P[i] >= P[i - 1] and P[i] >= P[i + 1] and P[i] >= rel * P.max()]
    out = []
    for i in sorted(idx, key=lambda i: -P[i]):
        if all(abs(bpm[i] - b) >= min_sep for b, _ in out): out.append((float(bpm[i]), float(P[i] / P.max())))
    return out


def lk_motion(gray):
    """gray (T,H,W) float → 전역 평행이동 속도 vx, vy (T−1,) px/frame (Lucas–Kanade 1차 전역 해, 부호 있음).
    |프레임 차분| 평균은 정류 신호라 주기 움직임을 2배 주파수에 놓으므로 쓰지 않는다."""
    Ix = (0.5 * (gray[:, :, 2:] - gray[:, :, :-2]))[:-1, 1:-1, :]
    Iy = (0.5 * (gray[:, 2:, :] - gray[:, :-2, :]))[:-1, :, 1:-1]
    It = (gray[1:] - gray[:-1])[:, 1:-1, 1:-1]
    a = (Ix * Ix).sum((1, 2)); b = (Ix * Iy).sum((1, 2)); c = (Iy * Iy).sum((1, 2))
    sxt = (Ix * It).sum((1, 2)); syt = (Iy * It).sum((1, 2)); det = a * c - b * b + 1e-6
    return -(c * sxt - b * syt) / det, -(a * syt - b * sxt) / det


def small_gray(fr, side=64):
    import cv2
    g = fr.astype(np.float32).mean(-1) if fr.ndim == 3 else fr.astype(np.float32)
    h, w = g.shape; sc = side / max(h, w)
    return cv2.resize(g, (max(8, int(round(w * sc))), max(8, int(round(h * sc)))), interpolation=cv2.INTER_AREA), sc


def traces_from_frames(frames_iter):
    """frames_iter yields uint8/float (H,W,3) RGB crops → (g, v): global luminance trace, (vx, vy) global translation px/frame (crop scale)"""
    g, gs, sc = [], [], 1.0
    for fr in frames_iter:
        g.append(float(fr.astype(np.float32).mean())); sg, sc = small_gray(fr); gs.append(sg)
    vx, vy = lk_motion(np.stack(gs)); return np.array(g), (vx / sc, vy / sc)


def motion_spectrum(v):
    """(vx, vy) → (bpm, 합산 스펙트럼: 각 축을 자기 최대로 정규화해 합)"""
    bx, Px = spectrum(v[0]); _, Py = spectrum(v[1])
    return bx, Px / (Px.max() + 1e-12) + Py / (Py.max() + 1e-12)


def classify(pred, gt, pk_g, pk_m, tol_lock=3.0, tol_ok=5.0):
    if pred is None or gt is None: return "missing", None
    err = abs(pred - gt)
    if err <= tol_ok: return "ok", None
    if abs(pred - gt / 2.0) <= tol_ok or abs(pred - 2.0 * gt) <= 2 * tol_ok: return "half_double", None
    for src, pk in (("g", pk_g), ("m", pk_m)):
        for b, rp in pk:
            if abs(pred - b) <= tol_lock and abs(b - gt) > tol_ok: return "lockin", f"{src}@{b:.0f}({rp:.2f})"
    if 55.0 <= pred <= 95.0 and gt >= 100.0: return "prior", None
    return "other", None


def nuis_at_hr(gt, pk_g, pk_m, tol=5.0):
    dom = [pk[0][0] for pk in (pk_g, pk_m) if pk]
    return bool(gt is not None and any(abs(b - gt) <= tol for b in dom))


def load_preds(axis, model_list):
    """vid -> list of (pred, gt) per seed"""
    acc = {}
    for mm in model_list:
        if axis == "VIPL":
            f = f"{R}/agg_{mm}_VIPL.json"
            if not os.path.exists(f): continue
            for r in json.load(open(f)):
                if r.get("gt_lit") is None: continue
                acc.setdefault(r["vid"], []).append((float(r["pred"]), float(r["gt_lit"])))
        elif axis == "MMPD":
            f = f"{R}/agg_{mm}_MMPD.json"
            if not os.path.exists(f): continue
            for r in json.load(open(f)):
                acc.setdefault(r["vid"], []).append((float(r["pred"]), float(r["gt"])))
        else:
            f = f"{R}/agg_{mm}.json"
            if not os.path.exists(f): continue
            for r in json.load(open(f)).get(axis, []):
                acc.setdefault(str(r["vid"]), []).append((float(r["pred"]), float(r["gt"])))
    return acc


# ---------------- nuisance peaks per video ----------------
def vipl_targets():
    rows = json.load(open(f"{R}/agg_NF_HYB_L_VIPL.json"))
    return {r["vid"]: r for r in rows if r.get("gt_lit") is not None and (r.get("scenario") == "v7" or r["gt_lit"] >= 90.0)}


def vipl_traces(vid):
    g, m = [], []
    for f in sorted(glob.glob(f"{PFA}/cache/vipl_u8/{vid}_input*.npy"), key=lambda s: int(re.search(r"_input(\d+)\.npy", s).group(1))):
        x = np.load(f)[:160].astype(np.float32)              # 마지막 1프레임은 다음 청크와 중복
        g.append(x.mean(axis=(1, 2, 3))); gs = x.mean(-1).reshape(160, 64, 2, 64, 2).mean((2, 4)); m.append(np.stack(lk_motion(gs)) / 0.5)   # 64 px 격자(½ 배율) → 원 crop px
    if not g: return None, None
    v = np.concatenate(m, axis=1); return np.concatenate(g), (v[0], v[1])


def haar_box(frame_rgb):
    import cv2
    det = cv2.CascadeClassifier(HAAR)
    gray = cv2.cvtColor(frame_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    faces = det.detectMultiScale(gray, 1.1, 4)
    H, W = gray.shape
    if len(faces) == 0: return int(H * 0.15), int(H * 0.85), int(W * 0.2), int(W * 0.8)
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return max(0, y), min(H, y + h), max(0, x), min(W, x + w)


def mmpd_targets():
    """held-out 7–33: motion 2/3/4 or exercise==1  → vid(저장 이름) -> (mat path, 조건)"""
    tr = json.load(open(f"{PFA}/mmpd_trials.json")); rows = json.load(open(f"{R}/agg_NF_HYB_L_MMPD.json"))
    byname = {r["vid"]: r for r in rows}
    out = {}
    for key, t in tr.items():
        s = int(t["subject"])
        if not (7 <= s <= 33): continue
        if not (t["motion"] in (2, 3, 4) or t["exercise"] == 1): continue
        # 저장 이름 subject{s}_L{light}_MO{motion}_E{exercise}_S{skin}_... 는 집계 행에서 찾는다
        cands = [r for r in rows if r["subject"] == s and r["light"] == t["light"] and r["motion"] == t["motion"] and r["exercise"] == t["exercise"] and r["skin"] == t["skin"]]
        if len(cands) != 1: continue
        mat = None
        for root in (f"/SPECPHYS_RAW/datasets_local/MMPD/subject{s}", f"/SPECPHYS_RAW/MMPD/subject{s}"):
            p = f"{root}/p{s}_{t['trial']}.mat"
            if os.path.exists(p): mat = p; break
        if mat: out[cands[0]["vid"]] = (mat, t)
    return out


def mmpd_traces(mat):
    import scipy.io as sio
    d = sio.loadmat(mat); v = np.array(d["video"])                      # (T,H,W,3) float 0-1 또는 uint8
    if v.dtype != np.uint8: v = (v * 255.0) if v.max() <= 1.0 else v
    y0, y1, x0, x1 = haar_box(v[0]); return traces_from_frames(v[:, y0:y1, x0:x1])


def nirp_traces(sub):
    import cv2
    cap = cv2.VideoCapture(f"{PFA}/NIRP_LARGE/subject{sub}/vid.avi"); box = None
    def it():
        nonlocal box
        while True:
            ok, fr = cap.read()
            if not ok: break
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            if box is None: box = haar_box(fr)
            y0, y1, x0, x1 = box; yield fr[y0:y1, x0:x1]
    g, m = traces_from_frames(it()); cap.release(); return g, m


def main():
    a = sys.argv[1:]; skip_mmpd = "--skip-mmpd" in a
    pk_path = f"{R}/selfsup_nuis_peaks.json"; PK = json.load(open(pk_path)) if os.path.exists(pk_path) else {}
    t0 = time.time()
    # --- VIPL
    tg = vipl_targets(); PK.setdefault("VIPL", {})
    for i, (vid, r) in enumerate(sorted(tg.items())):
        if vid in PK["VIPL"]: continue
        g, m = vipl_traces(vid)
        if g is None: continue
        bg, Pg = spectrum(g); bm, Pm = motion_spectrum(m)
        PK["VIPL"][vid] = {"gt": r["gt_lit"], "scenario": r["scenario"], "pk_g": peaks(bg, Pg), "pk_m": peaks(bm, Pm), "g_std": float(hp(g).std()), "v_rms": float(np.sqrt(np.mean(m[0] ** 2 + m[1] ** 2))), "n": int(len(g))}
        if i % 50 == 0: print(f"  VIPL {i}/{len(tg)} {time.time() - t0:.0f}s", flush=True); json.dump(PK, open(pk_path, "w"))
    json.dump(PK, open(pk_path, "w"))
    # --- MR-NIRP large
    PK.setdefault("NIRP_LARGE", {})
    rows = json.load(open(f"{R}/agg_NF_HYB_L.json"))["NIRP_LARGE"]
    for r in rows:
        vid = str(r["vid"])
        if vid in PK["NIRP_LARGE"]: continue
        g, m = nirp_traces(vid); bg, Pg = spectrum(g); bm, Pm = motion_spectrum(m)
        PK["NIRP_LARGE"][vid] = {"gt": r["gt"], "pk_g": peaks(bg, Pg), "pk_m": peaks(bm, Pm), "g_std": float(hp(g).std()), "v_rms": float(np.sqrt(np.mean(m[0] ** 2 + m[1] ** 2))), "n": int(len(g))}
    json.dump(PK, open(pk_path, "w")); print(f"  NIRP_LARGE done {time.time() - t0:.0f}s", flush=True)
    # --- MMPD
    if not skip_mmpd:
        PK.setdefault("MMPD", {}); tg = mmpd_targets(); print(f"  MMPD targets {len(tg)}", flush=True)
        for i, (vid, (mat, t)) in enumerate(sorted(tg.items())):
            if vid in PK["MMPD"]: continue
            try: g, m = mmpd_traces(mat)
            except Exception as e: print(f"  MMPD {vid} ERR {e}", flush=True); continue
            bg, Pg = spectrum(g); bm, Pm = motion_spectrum(m)
            PK["MMPD"][vid] = {"motion": t["motion"], "exercise": t["exercise"], "pk_g": peaks(bg, Pg), "pk_m": peaks(bm, Pm), "g_std": float(hp(g).std()), "v_rms": float(np.sqrt(np.mean(m[0] ** 2 + m[1] ** 2))), "n": int(len(g))}
            if i % 20 == 0: print(f"  MMPD {i}/{len(tg)} {time.time() - t0:.0f}s", flush=True); json.dump(PK, open(pk_path, "w"))
        json.dump(PK, open(pk_path, "w"))
    # --- classification tables
    LINES = []; OUT = {}
    def emit(s): print(s); LINES.append(s)
    groups = {"VIPL": [("운동 후 v7", lambda v, k: k["scenario"] == "v7"), ("심박 ≥90 (v7 제외)", lambda v, k: k["scenario"] != "v7" and k["gt"] >= 90), ("심박 ≥120", lambda v, k: k["gt"] >= 120)],
              "MMPD": [("보행 m4", lambda v, k: k["motion"] == 4), ("대화 m3", lambda v, k: k["motion"] == 3), ("회전 m2", lambda v, k: k["motion"] == 2), ("운동 후 e1", lambda v, k: k["exercise"] == 1)],
              "NIRP_LARGE": [("대동작 전체", lambda v, k: True)]}
    cats = ["ok", "half_double", "lockin", "prior", "other"]
    for axis, grp in groups.items():
        if axis not in PK or (axis == "MMPD" and skip_mmpd): continue
        emit(f"=== {axis}: 영상 수 {len(PK[axis])} — 분류 비율(%) = 시드별 분류 평균; nuis@HR = 교란 지배 피크가 GT ±5 bpm 인 영상 비율")
        OUT[axis] = {}
        for gname, sel in grp:
            vids = [v for v, k in PK[axis].items() if sel(v, k)]
            if not vids: continue
            nah = np.mean([nuis_at_hr(PK[axis][v]["gt"] if axis != "MMPD" else None, PK[axis][v]["pk_g"], PK[axis][v]["pk_m"]) for v in vids]) if axis != "MMPD" else None
            emit(f"--- {gname} (n={len(vids)})" + (f"  nuis@HR={100 * nah:.0f}%" if nah is not None else "") + f"  g_std 중앙값={np.median([PK[axis][v]['g_std'] for v in vids]):.2f} 이동속도 rms 중앙값={np.median([PK[axis][v]['v_rms'] for v in vids]):.2f} px/frame")
            OUT[axis][gname] = {"n": len(vids), "nuis_at_hr": nah, "models": {}}
            for mname, mlist in MODELS[axis].items():
                preds = load_preds(axis, mlist)
                cnt = {c: 0.0 for c in cats}; tot = 0; lock_src = {"g": 0, "m": 0}; errs = []
                for v in vids:
                    pg = preds.get(v)
                    if not pg: continue
                    gt_ax = PK[axis][v].get("gt")
                    for pred, gt in pg:
                        gtv = gt if gt_ax is None else gt_ax
                        if axis == "MMPD": gtv = gt
                        c, why = classify(pred, gtv, PK[axis][v]["pk_g"], PK[axis][v]["pk_m"]); cnt[c] += 1.0 / len(pg); tot += 1.0 / len(pg); errs.append(abs(pred - gtv))
                        if c == "lockin": lock_src[why[0]] += 1
                        if axis == "MMPD" and nah is None: pass
                if tot == 0: continue
                fr = {c: 100.0 * cnt[c] / tot for c in cats}; fails = tot - cnt["ok"]
                emit(f"  {mname:12s} MAE {np.mean(errs):6.2f} | ok {fr['ok']:4.0f}%  half/double {fr['half_double']:4.0f}%  lock-in {fr['lockin']:4.0f}% (g {lock_src['g']}, m {lock_src['m']})  prior {fr['prior']:4.0f}%  other {fr['other']:4.0f}%  | 실패 중 lock-in {100 * cnt['lockin'] / max(fails, 1e-9):4.0f}% prior {100 * cnt['prior'] / max(fails, 1e-9):4.0f}%")
                OUT[axis][gname]["models"][mname] = {"mae": float(np.mean(errs)), "frac": fr, "lock_src": lock_src, "fail_lockin": float(cnt["lockin"] / max(fails, 1e-9)), "fail_prior": float(cnt["prior"] / max(fails, 1e-9))}
        if axis == "MMPD":   # MMPD 는 GT 를 집계 행에서 가져오므로 nuis@HR 을 모델 무관하게 따로 계산
            for gname, sel in grp:
                vids = [v for v, k in PK[axis].items() if sel(v, k)]; preds = load_preds(axis, MODELS[axis]["NF_HYB_L"])
                vals = [nuis_at_hr(preds[v][0][1], PK[axis][v]["pk_g"], PK[axis][v]["pk_m"]) for v in vids if v in preds]
                if vals: emit(f"  [{gname}] nuis@HR={100 * np.mean(vals):.0f}%"); OUT[axis][gname]["nuis_at_hr"] = float(np.mean(vals))
    open(f"{R}/selfsup_phase0.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/selfsup_phase0.json", "w"), indent=1, ensure_ascii=False)
    print(f"-> {R}/selfsup_phase0.txt, selfsup_phase0.json, selfsup_nuis_peaks.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
