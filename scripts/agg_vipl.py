#!/usr/bin/env python3
"""VIPL-HR 정본 집계: 영상 단위 HR (toolbox 프로토콜: cumsum→detrend λ=100→butter1 0.6–3.3Hz→periodogram) + 메타(시나리오·소스·fold).
GT 2종 병기: gt(정본: 라벨 파형에서 동일 프로토콜) / gt_lit(문헌 호환: gt_HR.csv 초당 HR 평균, RhythmNet/PhysFormer 표 기준).
사용:
  python scripts/agg_vipl.py MODEL [MODEL...]      MODEL = 제로샷 모델(runs/ev_{M}_VIPL) | fold 모델(runs/finetune_{M}) | POS | CHROM
  python scripts/agg_vipl.py --cv TAG [TAG...]     VF1_{TAG}..VF5_{TAG} 5-fold 합산 → agg_CV_{TAG}_VIPL.json (각 영상은 자기 held-out fold 에서만)
출력: _workspace/data/results/agg_{NAME}_VIPL.json (rows) + 표(전체 SD/MAE/RMSE/r 양 GT, 시나리오·소스별 MAE)."""
import glob, json, pickle, re, sys
import numpy as np, scipy.io as sio
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from agg_model2 import cat_chunks, toolbox_hr, hr_for, PF
RES = f"{PF}/../results"; CACHE = f"{PF}/cache/vipl_u8"; FOLD = "/SPECPHYS_RAW/vipl_hr_raw/fold"
SCEN = {"v1": "stable", "v2": "motion", "v3": "talking", "v4": "dark", "v5": "bright", "v6": "long-dist", "v7": "exercise", "v8": "phone-stable", "v9": "phone-motion"}
_meta = None; _fold = None


def meta():
    global _meta, _fold
    if _meta is None:
        _meta = {}
        for m in glob.glob(f"{CACHE}/*_meta.json"):
            j = json.load(open(m)); _meta[j["index"]] = j
        _fold = {}
        for k in range(1, 6):
            for s in np.asarray(sio.loadmat(f"{FOLD}/fold{k}.mat")[f"fold{k}"]).ravel().tolist():
                _fold[int(s)] = k
    return _meta, _fold


def row(vid, pred, gt):
    M, F = meta(); j = M[vid]
    gl = j.get("gt_hr_mean")
    return dict(vid=vid, subject=j["subject"], scenario=j["scenario"][:2], extra=j["scenario"].endswith("-2"), source=j["source"],
                fold=F[j["subject"]], n_frames=j["n_out"], gt=gt, gt_lit=gl, pred=pred, err=abs(pred - gt),
                err_lit=(abs(pred - gl) if gl is not None else None), ratio=pred / gt)


READOUT = __import__("os").environ.get("RPPG_AGG_READOUT", "canon")  # [40 문서 §6] 'lit' = 문헌형 읽기 병기(출력 접미 _lit)


def lit_hr(sig, fs=30.0, lo=0.6, hi=3.3, w=300):
    """문헌형 읽기(30 문서 §7.2 변형 D 와 동일 구현): 출력을 누적합·detrend 없이 그대로 평균 제거 → butter(1) 0.6–3.3 Hz → 10 s(300 프레임) 창마다
    periodogram(nfft 4096) 피크 → 창 평균 (PhysFormer 공식 평가의 '짧은 창 피크 평균' 방식). 모든 모델(라벨 유형 무관)에 동일 적용 — 정본(cumsum·영상 전체 periodogram) 과 섞어 쓰지 않는다."""
    from scipy.signal import butter as _bt, filtfilt as _ff, periodogram as _pg
    x = np.asarray(sig, float); b, a = _bt(1, [lo / (fs / 2), hi / (fs / 2)], btype="bandpass")
    def one(seg, nfft):
        y = _ff(b, a, seg - seg.mean()); f, pxx = _pg(y, fs=fs, nfft=nfft, detrend=False); m = (f >= lo) & (f <= hi); return 60.0 * f[m][np.argmax(pxx[m])]
    n = len(x) // w
    if n == 0: return float(one(x, int(2 ** np.ceil(np.log2(len(x))))))
    return float(np.mean([one(x[i * w:(i + 1) * w], 4096) for i in range(n)]))


def rows_from_pickles(pks):
    rows = []
    for pk in pks:
        d = pickle.load(open(pk, "rb"))
        for vid in d["predictions"]:
            p = cat_chunks(d["predictions"][vid]); l = cat_chunks(d["labels"][vid])
            pred = lit_hr(p) if READOUT == "lit" else hr_for(p, d.get("label_type"))
            rows.append(row(str(vid), pred, hr_for(l, d.get("label_type"))))
    return rows


def agg_model(model):
    if model in ("POS", "CHROM"):
        src = json.load(open(f"{RES}/pos_pervideo_VIPL.json"))
        return [row(r["vid"], r[model], r["gt"]) for r in src]
    if re.match(r"^(VF|NF)\d", model):  # [10-28] fold 모델 = VF{k}/NF{k}(접두 뒤 숫자); NF_HYB_L 같은 NF-Phys zero-shot 셀은 제외 — fold 모델(VIPL fold VF{k} / MR-NIRP in-domain NF{k}, VIPL fold 1 고정): train_and_test 의 자체 테스트 출력 (다른 모델의 finetune 출력은 합성 축이므로 절대 fallback 금지)
        pks = sorted(glob.glob(f"{PF}/runs/finetune_{model}/vipl_u8/saved_test_outputs/*.pickle"))
    else:
        pks = sorted(glob.glob(f"{PF}/runs/ev_{model}_VIPL/vipl_u8/saved_test_outputs/*.pickle"))
    return rows_from_pickles(pks)


def agg_cv(tag):
    rows = []
    for k in range(1, 6):
        rk = agg_model(f"VF{k}_{tag}")
        bad = [r for r in rk if r["fold"] != k]
        assert not bad, f"fold leak: VF{k}_{tag} has {len(bad)} videos outside fold {k}"
        rows += rk
    return rows


def metrics(rows, key="err"):
    rows = [r for r in rows if r.get(key) is not None]
    if not rows: return None
    gk = "gt" if key == "err" else "gt_lit"
    g = np.array([r[gk] for r in rows]); p = np.array([r["pred"] for r in rows]); e = p - g
    r_ = float(np.corrcoef(g, p)[0, 1]) if len(rows) > 2 and g.std() > 0 and p.std() > 0 else float("nan")
    return dict(n=len(rows), SD=float(e.std()), MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())), r=r_,
                dbl=int(sum(abs(r["ratio"] - 2) < 0.2 for r in rows)), hlv=int(sum(abs(r["ratio"] - 0.5) < 0.1 for r in rows)))


def fmt(m, label):
    return (f"  {label:22s} n={m['n']:4d} SD={m['SD']:6.2f} MAE={m['MAE']:6.2f} RMSE={m['RMSE']:6.2f} r={m['r']:5.2f} dbl={m['dbl']} hlv={m['hlv']}"
            if m else f"  {label:22s} (none)")


def report(rows, name):
    print(f"--- {name} VIPL-HR (n={len(rows)}; 정본 GT=라벨 파형 / 문헌 GT=gt_HR 평균)")
    print(fmt(metrics(rows, "err"), "all [정본 GT]")); print(fmt(metrics(rows, "err_lit"), "all [문헌 GT]"))
    for sc, nm in SCEN.items():
        sub = [r for r in rows if r["scenario"] == sc]
        if sub: print(fmt(metrics(sub, "err"), f"{sc} {nm}") + f" | lit MAE={metrics(sub, 'err_lit')['MAE']:.2f}")
    for s in (1, 2, 3):
        sub = [r for r in rows if r["source"] == s]
        if sub: print(fmt(metrics(sub, "err"), f"source{s}") + f" | lit MAE={metrics(sub, 'err_lit')['MAE']:.2f}")
    for k in range(1, 6):
        sub = [r for r in rows if r["fold"] == k]
        if sub: print(fmt(metrics(sub, "err"), f"fold{k}") + f" | lit MAE={metrics(sub, 'err_lit')['MAE']:.2f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    cv = args and args[0] == "--cv"
    for m in (args[1:] if cv else args):
        rows = agg_cv(m) if cv else agg_model(m)
        name = f"CV_{m}" if cv else m
        if not rows: print(f"--- {name}: no VIPL outputs yet"); continue
        sfx = "_lit" if READOUT == "lit" else ""
        report(rows, name + sfx); json.dump(rows, open(f"{RES}/agg_{name}_VIPL{sfx}.json", "w"), indent=0)
