#!/usr/bin/env python3
"""19단계 진단: (1) 학습 풀 구성 진단 + PURE steady GT BPM 커버리지, (2) PURE 오차 원인(피클 재분석).

- 풀: COTRAIN_ALLF(=COTRAIN_ALL + TRAIN6LO; ALLF 디렉토리가 없으면 가상 결합) 와 COTRAIN_BAL(있으면).
- 클립별: 출처(real/synth), 도메인(UBFC/MCD/PURE/SFHQ), 조건(clean/illum/motion/mixed/heavy/compress/noisy-real),
  GT BPM(ground_truth.txt 1행, n×8 zero-pad rfft, 0.6~4.0 Hz), 파형(neurokit/analytic a2).
- PURE: 태스크 라벨(xx-0y: 1 steady, 2 talking, 3 slow-translation, 4 fast-translation,
  5 small-rotation, 6 medium-rotation), 트라이얼별 GT BPM(PURE_RAW json 원본).
- PURE 피클: 모델별 트라이얼별 toolbox 청크 지표(16번 보고서 프로토콜)와 영상 단위 고해상 FFT.
출력: _workspace/data/results/train_pool_balance.{png,json}, pure_error_diagnosis.json
사용: python scripts/analyze_train_pool_balance.py [--pool COTRAIN_ALL] [--bal COTRAIN_BAL]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse, glob, json, pickle, sys
import numpy as np
import scipy.signal as ss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/SPECPHYS_ROOT"
PF = f"{ROOT}/_workspace/data/physformer"
RES = f"{ROOT}/_workspace/data/results"
sys.path.insert(0, f"{ROOT}/repos/rPPG-Toolbox")
FS = 30
TASK = {1: "steady", 2: "talking", 3: "slow-translation", 4: "fast-translation",
        5: "small-rotation", 6: "medium-rotation"}
BINS = np.arange(40, 161, 10)


def hires_bpm(sig, lo=0.6, hi=4.0):
    sig = np.asarray(sig, float)
    sig = sig - sig.mean()
    fr = np.fft.rfftfreq(len(sig) * 8, 1 / FS)
    pw = np.abs(np.fft.rfft(sig, n=len(sig) * 8)) ** 2
    band = (fr >= lo) & (fr <= hi)
    return float(fr[band][pw[band].argmax()] * 60)


def read_gt(d):
    with open(os.path.join(d, "ground_truth.txt")) as f:
        return np.array(f.readline().split(), float)


def classify_noise(nz):
    if not nz:
        return "clean"
    keys = set(k for k, v in nz.items() if v)
    has_i = bool(keys & {"illum_drift", "flicker_amp"})
    has_m = bool(keys & {"motion_px", "rot_deg"})
    if has_i and has_m and nz.get("sensor_std", 0) >= 3:
        return "heavy-mixed"
    if has_i and has_m:
        return "mixed"
    if has_i:
        return "illum"
    if has_m:
        return "motion"
    return "sensor"


NOISY_REAL_VARIANTS = ["illum", "motion", "illum+sensor", "mixed", "heavy-mixed"]


def describe_clip(link_name, real):
    """real = 최종 실체 경로. 반환 dict."""
    rel = os.path.relpath(real, ROOT)
    row = {"clip": link_name, "target": rel}
    num = int(link_name.split("subject")[-1])
    if rel.startswith("UBFC_dataset"):
        row.update(source="real", domain="UBFC", cond="clean", wave="real")
    elif "/MCD_TRAIN/" in rel:
        src = open(os.path.join(real, "source.txt")).read().split()[0]
        state = "steady" if src.endswith("before") else "post-exercise"
        row.update(source="real", domain="MCD", cond=f"clean({state})", wave="real", identity=src.split("_")[0])
    elif "/PURE_TRAIN/" in rel:
        src = open(os.path.join(real, "source.txt")).read().split()[1]
        row.update(source="real", domain="PURE", cond=f"clean({TASK[int(src[3:5])]})", wave="real",
                   identity="PURE" + src[:2], task=TASK[int(src[3:5])])
    elif "/COTRAIN_NOISY_REAL/" in rel:
        # make_noisy_real_train.py: 정렬된 실 subject 순서 i → VARIANTS[i%5]
        row.update(source="real-aug", domain="UBFC", cond="noisy-real", wave="real")
    elif "/CMP_REAL/" in rel:
        row.update(source="real-aug", domain="UBFC", cond="compress(q6)", wave="real")
    elif "/CMP_SYN" in rel:
        row.update(source="synth-aug", domain="SFHQ", cond="compress(q6)", wave="analytic")
    else:  # 합성 실체 (meta.json)
        mp = os.path.join(real, "meta.json")
        if os.path.exists(mp):
            m = json.load(open(mp))
            row.update(source="synth", domain="SFHQ", cond=classify_noise(m.get("noise")),
                       wave="analytic" if m.get("wave_a2") is not None else "neurokit",
                       a2=m.get("wave_a2"), identity=os.path.basename(m["frames_dir"]),
                       achieved_bpm=m.get("achieved_bpm"), spatial=m.get("spatial"))
        else:
            row.update(source="?", domain="?", cond="?", wave="?")
    if "identity" not in row:
        row["identity"] = f"UBFC{num if num < 100 else num % 100 if num < 400 else '?'}"
    return row


def scan_pool(pool_dir, extra_dirs=()):
    rows = []
    subs = sorted(glob.glob(f"{pool_dir}/subject*"), key=lambda s: int(os.path.basename(s)[7:]))
    for s in subs:
        real = os.path.realpath(s)
        r = describe_clip(os.path.basename(s), real)
        r["gt_bpm"] = round(hires_bpm(read_gt(real)), 2)
        r["n_frames"] = len(read_gt(real))
        rows.append(r)
    for ed in extra_dirs:  # 가상 결합(예: TRAIN6LO → ALLF)
        for s in sorted(glob.glob(f"{ed}/subject*"), key=lambda s: int(os.path.basename(s)[7:])):
            r = describe_clip("+" + os.path.basename(ed) + "/" + os.path.basename(s), os.path.realpath(s))
            r["gt_bpm"] = round(hires_bpm(read_gt(s)), 2)
            r["n_frames"] = len(read_gt(s))
            rows.append(r)
    # NOISY_REAL 변형 라벨 복원 (UBFC 정렬 순서 i%5)
    nr = [r for r in rows if r["cond"] == "noisy-real"]
    for i, r in enumerate(sorted(nr, key=lambda r: int(r["target"].split("subject")[-1]))):
        r["cond"] = "noisy-real:" + NOISY_REAL_VARIANTS[i % 5]
        r["identity"] = "UBFC" + r["target"].split("subject")[-1]
    return rows


def summarize(rows):
    def cnt(key):
        out = {}
        for r in rows:
            out[r[key]] = out.get(r[key], 0) + 1
        return dict(sorted(out.items()))
    bpm = np.array([r["gt_bpm"] for r in rows])
    hist, _ = np.histogram(bpm, BINS)
    real = sum(r["source"] in ("real", "real-aug") for r in rows)
    synth = sum(r["source"] in ("synth", "synth-aug") for r in rows)
    clean = sum(r["cond"].startswith("clean") for r in rows)
    noisy = len(rows) - clean
    syn_clean = sum(r["cond"] == "clean" and r["source"] == "synth" for r in rows)
    syn_noisy = synth - syn_clean
    return {"n": len(rows), "real": real, "synth": synth, "clean": clean, "noisy": noisy,
            "synth_clean": syn_clean, "synth_noisy": syn_noisy,
            "by_source": cnt("source"), "by_domain": cnt("domain"), "by_cond": cnt("cond"), "by_wave": cnt("wave"),
            "bpm_min": float(bpm.min()), "bpm_max": float(bpm.max()),
            "bpm_hist_10": {f"{BINS[i]}-{BINS[i+1]}": int(hist[i]) for i in range(len(hist))},
            "bpm_hist_synth": {f"{BINS[i]}-{BINS[i+1]}": int(v) for i, v in enumerate(
                np.histogram([r["gt_bpm"] for r in rows if r["source"].startswith("synth")], BINS)[0])},
            "bpm_hist_real": {f"{BINS[i]}-{BINS[i+1]}": int(v) for i, v in enumerate(
                np.histogram([r["gt_bpm"] for r in rows if r["source"].startswith("real")], BINS)[0])}}


def pure_trials():
    out = []
    for d in sorted(glob.glob(f"{PF}/PURE_RAW/??-??")):
        t = os.path.basename(d)
        j = json.load(open(os.path.join(d, f"{t}.json")))
        wave = np.array([e["Value"]["waveform"] for e in j["/FullPackage"]], float)
        ts = np.array([e["Timestamp"] for e in j["/FullPackage"]], np.int64)
        fs_ppg = 1e9 / np.median(np.diff(ts))
        w = wave - wave.mean()
        fr = np.fft.rfftfreq(len(w) * 8, 1 / fs_ppg)
        pw = np.abs(np.fft.rfft(w, n=len(w) * 8)) ** 2
        band = (fr >= 0.6) & (fr <= 4.0)
        bpm = float(fr[band][pw[band].argmax()] * 60)
        out.append({"trial": t, "subject": int(t[:2]), "task_id": int(t[3:5]), "task": TASK[int(t[3:5])],
                    "gt_bpm": round(bpm, 2), "split": "train(01-06)" if int(t[:2]) <= 6 else "held-out(07-10)"})
    return out


# ---------- PURE 피클 재분석 ----------
def cat_chunks(chunks):
    import torch  # noqa
    return np.concatenate([chunks[k].detach().cpu().float().numpy().ravel() for k in sorted(chunks)])


def toolbox_hr(sig, lo=0.6, hi=3.3):
    from evaluation.post_process import _calculate_fft_hr, _detrend
    x = _detrend(np.cumsum(sig), 100)
    b, a = ss.butter(1, [lo / FS * 2, hi / FS * 2], btype="bandpass")
    x = ss.filtfilt(b, a, np.double(x))
    return float(_calculate_fft_hr(x, fs=FS, low_pass=lo, high_pass=hi))


def video_hr(sig):
    sig = np.cumsum(sig - sig.mean())
    b, a = ss.butter(3, [0.7 / (FS / 2), 4.0 / (FS / 2)], btype="band")
    f = ss.filtfilt(b, a, sig)
    return hires_bpm(f, 0.7, 4.0), f


def harmonic_ratio(f, gt_bpm, tol=3.0):
    fr = np.fft.rfftfreq(len(f) * 8, 1 / FS) * 60
    pw = np.abs(np.fft.rfft(f - f.mean(), n=len(f) * 8)) ** 2
    def band(c):
        m = (fr >= c - tol) & (fr <= c + tol)
        return pw[m].max() if m.any() else 0.0
    p0 = band(gt_bpm) + 1e-12
    return float(band(2 * gt_bpm) / p0), float(band(0.5 * gt_bpm) / p0)


def pure_pickles(trials):
    tinfo = {t["subject"] * 100 + t["task_id"]: t for t in trials}
    res = {}
    for pk in sorted(glob.glob(f"{PF}/runs/ev_*_PURE/*/saved_test_outputs/*.pickle")):
        model = pk.split("/runs/ev_")[1].split("_PURE")[0]
        d = pickle.load(open(pk, "rb"))
        rows = []
        for s in sorted(d["predictions"], key=lambda x: int(x)):
            pr, lb = cat_chunks(d["predictions"][s]), cat_chunks(d["labels"][s])
            hp, hg = toolbox_hr(pr), toolbox_hr(lb)
            hp_hi, pf = video_hr(pr)
            hg_hi, _ = video_hr(lb)
            r2, rh = harmonic_ratio(pf, hg_hi)
            vid = int(s)
            ti = tinfo.get(vid, {})
            ratio = hp_hi / hg_hi
            rows.append({"vid": vid, "trial": ti.get("trial"), "task": ti.get("task"), "split": ti.get("split"),
                         "gt_bpm_json": ti.get("gt_bpm"),
                         "tb_pred": round(hp, 2), "tb_gt": round(hg, 2), "tb_abs_err": round(abs(hp - hg), 2),
                         "hi_pred": round(hp_hi, 2), "hi_gt": round(hg_hi, 2), "hi_abs_err": round(abs(hp_hi - hg_hi), 2),
                         "ratio": round(ratio, 3), "doubled": bool(1.8 <= ratio <= 2.2), "halved": bool(0.45 <= ratio <= 0.55),
                         "pow_2f0_over_f0": round(r2, 3), "pow_halff0_over_f0": round(rh, 3)})
        res[model] = rows
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="COTRAIN_ALL")
    ap.add_argument("--extra", default="TRAIN6LO", help="가상 결합 디렉토리(콤마) — ALLF 재현")
    ap.add_argument("--bal", default="COTRAIN_BAL")
    ap.add_argument("--skip-pickles", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    pools = {}
    if os.path.isdir(f"{PF}/COTRAIN_ALLF"):
        pools["COTRAIN_ALLF"] = scan_pool(f"{PF}/COTRAIN_ALLF")
    else:
        ex = [f"{PF}/{e}" for e in a.extra.split(",") if e and os.path.isdir(f"{PF}/{e}")]
        pools["COTRAIN_ALLF(virtual=ALL+" + "+".join(os.path.basename(e) for e in ex) + ")"] = scan_pool(f"{PF}/{a.pool}", ex)
    if os.path.isdir(f"{PF}/{a.bal}"):
        pools[a.bal] = scan_pool(f"{PF}/{a.bal}")

    trials = pure_trials()
    steady = [t for t in trials if t["task"] == "steady"]
    steady_bpm = np.array([t["gt_bpm"] for t in steady])
    all_bpm = np.array([t["gt_bpm"] for t in trials])
    out = {"pools": {}, "pure": {"trials": trials,
                                 "steady_bpm_hist": {f"{BINS[i]}-{BINS[i+1]}": int(v) for i, v in enumerate(np.histogram(steady_bpm, BINS)[0])},
                                 "all_bpm_hist": {f"{BINS[i]}-{BINS[i+1]}": int(v) for i, v in enumerate(np.histogram(all_bpm, BINS)[0])},
                                 "steady_bpm_min_max": [float(steady_bpm.min()), float(steady_bpm.max())]}}
    for name, rows in pools.items():
        sm = summarize(rows)
        # 미커버 정량화: PURE(전체/steady) 트라이얼 중 풀 GT BPM ±3 이내 학습 클립이 없는 비율, 합성만 기준도
        pool_bpm = np.array([r["gt_bpm"] for r in rows])
        syn_bpm = np.array([r["gt_bpm"] for r in rows if r["source"].startswith("synth")])
        def uncovered(targets, ref, tol=3.0):
            return [float(t) for t in targets if not np.any(np.abs(ref - t) <= tol)]
        sm["pure_steady_uncovered_pm3_anyclip"] = uncovered(steady_bpm, pool_bpm)
        sm["pure_steady_uncovered_pm3_synth"] = uncovered(steady_bpm, syn_bpm)
        sm["pure_all_uncovered_pm3_anyclip"] = uncovered(all_bpm, pool_bpm)
        sm["pure_all_uncovered_pm3_synth"] = uncovered(all_bpm, syn_bpm)
        out["pools"][name] = {"summary": sm, "clips": rows}

    if not a.skip_pickles:
        out["pure_models"] = pure_pickles(trials)
        summ = {}
        for m, rows in out["pure_models"].items():
            def agg(rs):
                if not rs:
                    return None
                return {"n": len(rs), "tb_mae": round(float(np.mean([r["tb_abs_err"] for r in rs])), 2),
                        "hi_mae": round(float(np.mean([r["hi_abs_err"] for r in rs])), 2),
                        "hi_median": round(float(np.median([r["hi_abs_err"] for r in rs])), 2),
                        "doubled": sum(r["doubled"] for r in rs), "halved": sum(r["halved"] for r in rs),
                        "err_gt5": sum(r["hi_abs_err"] > 5 for r in rs),
                        "err_gt5_and_gt_bpm_lt60": sum(r["hi_abs_err"] > 5 and r["hi_gt"] < 60 for r in rs),
                        "doubled_gt_bpm": [r["hi_gt"] for r in rs if r["doubled"]]}
            summ[m] = {"all": agg(rows), "held-out(07-10)": agg([r for r in rows if r["vid"] >= 700]),
                       "by_task": {t: agg([r for r in rows if r["task"] == t]) for t in TASK.values()},
                       "steady_held-out": agg([r for r in rows if r["task"] == "steady" and r["vid"] >= 700])}
        out["pure_model_summary"] = summ

    json.dump(out, open(f"{RES}/train_pool_balance.json", "w"), indent=1, ensure_ascii=False)

    # ---------- 그림 ----------
    n = len(pools)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4.2), squeeze=False)
    for ax, (name, rows) in zip(axes[0], pools.items()):
        rb = [r["gt_bpm"] for r in rows if r["source"].startswith("real")]
        sb = [r["gt_bpm"] for r in rows if r["source"].startswith("synth")]
        ax.hist([rb, sb], BINS, stacked=True, color=["#4C72B0", "#DD8452"], label=["real(+aug)", "synth(+aug)"], alpha=0.9)
        ax2 = ax.twinx()
        ax2.hist(steady_bpm, BINS, histtype="step", color="k", lw=2, label="PURE steady GT (n=%d)" % len(steady_bpm))
        ax2.hist(all_bpm, BINS, histtype="step", color="gray", lw=1, ls="--", label="PURE all 59")
        ax2.set_ylabel("PURE trials")
        ax.set_xlabel("GT BPM (hi-res FFT)"); ax.set_ylabel("train clips"); ax.set_title(f"{name} (n={len(rows)})", fontsize=10)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    plt.tight_layout(); plt.savefig(f"{RES}/train_pool_balance.png", dpi=130)
    print("saved", f"{RES}/train_pool_balance.png/.json")
    for name, p in out["pools"].items():
        print("==", name); print(json.dumps(p["summary"], ensure_ascii=False, indent=1))
    print("== PURE steady hist", out["pure"]["steady_bpm_hist"])
    if "pure_model_summary" in out:
        for m, s in out["pure_model_summary"].items():
            print("==", m, "all", s["all"], "\n   held-out", s["held-out(07-10)"], "\n   steady", s["by_task"]["steady"])


if __name__ == "__main__":
    main()
