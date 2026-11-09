#!/usr/bin/env python3
"""COTRAIN_BAL 조립 (심볼릭링크) + 검증.
구성:
  subject1~41(34)      → UBFC_dataset/DATASET_2 (COTRAIN_ALL 과 동일 34명; 42~49 held-out)
  subject301~341(34)   → COTRAIN_NOISY_REAL (실+노이즈 증강, 정답 불변)
  subject501~510(10)   → CMP_REAL (UBFC mpeg4 q6)
  subject601~612(12)   → CMP_SYN_BAL (TRAIN_BAL clean 12클립의 mpeg4 q6 사본; 본 스크립트가 생성)
  subject701~724,801~824(48) → MCD_TRAIN (정체성 21~44, 안정/운동직후)
  subject9SSTT(35)     → PURE_TRAIN (피험자 01~06 전 태스크; 07~10 held-out)
  subject1001~1036(36) → TRAIN_BAL (BPM 40~160 균등, clean:light:medium 12:12:12)
  subject1201~1212(12) → TRAIN_BAL_LOA2 (v2 저 a2 보강: a2 0/0.1/0.2/0.3 × 3, 12빈 하나씩, clean 8 : light 4) — --no-loa2 로 제외(v1 209)
검증: 링크 무결성(vid+GT), TRAIN_BAL(+LOA2) achieved BPM / 적합 a2, 균형표(실:합성, clean:noisy, BPM·a2 히스토그램), held-out 정체성 누설 0.
출력: _workspace/data/results/cotrain_bal_manifest.json
"""
import argparse, glob, json, os, subprocess, sys
from collections import Counter
import numpy as np

ROOT = "/SPECPHYS_ROOT"
PF = f"{ROOT}/_workspace/data/physformer"
OUT = f"{PF}/COTRAIN_BAL"
CMP = f"{PF}/CMP_SYN_BAL"
sys.path.insert(0, f"{ROOT}/scripts")
from compress_aug import compress_copy  # noqa


def link(name, target):
    p = os.path.join(OUT, name)
    if os.path.islink(p) or os.path.exists(p):
        if os.path.realpath(p) == os.path.realpath(target):
            return
        os.remove(p)
    os.symlink(os.path.realpath(target), p)


def load_metas(d):
    return {int(os.path.basename(s)[7:]): json.load(open(f"{s}/meta.json"))
            for s in sorted(glob.glob(f"{d}/subject*"), key=lambda x: int(os.path.basename(x)[7:]))}


def verify_synth(dname, metas, base):
    """achieved BPM / GT 스펙트럼 피크 / a2·a3 적합 / ffprobe 프레임 수."""
    ver = []
    for i, m in sorted(metas.items()):
        ppg = np.array(open(f"{PF}/{dname}/subject{i}/ground_truth.txt").readline().split(), float)
        n = len(ppg); fr = np.fft.rfftfreq(n * 8, 1 / 30) * 60
        pw = np.abs(np.fft.rfft(ppg - ppg.mean(), n=n * 8)) ** 2
        band = (fr >= 36) & (fr <= 240)
        peak = fr[band][pw[band].argmax()]
        f0 = m["achieved_bpm"]
        def bp(c, tol=2.0):
            mm = (fr >= c - tol) & (fr <= c + tol); return pw[mm].max() if mm.any() else 0.0
        a2_fit = float(np.sqrt(bp(2 * f0) / bp(f0)))
        a3_fit = float(np.sqrt(bp(3 * f0) / bp(f0))) if 3 * f0 <= 240 else None
        pr = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                             "stream=nb_read_frames,codec_name", "-of", "csv=p=0", f"{PF}/{dname}/subject{i}/vid.avi"],
                            capture_output=True, text=True).stdout.strip()
        ver.append({"id": i, "src": dname, "link": f"subject{base + i}", "bpm_target": m["bpm"], "achieved": f0, "peak_is_f0": bool(abs(peak - f0) < 1.5),
                    "cond": m.get("cond"), "noise": m.get("noise"), "spatial": m.get("spatial"), "face": os.path.basename(m["frames_dir"]),
                    "a2_req": m["wave_a2"], "a2_fit": round(a2_fit, 3), "a3_req": m["wave_a3"], "a3_fit": None if a3_fit is None else round(a3_fit, 3),
                    "seed": m["seed"], "n_frames": m["n_frames"], "ffprobe": pr})
    return ver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-loa2", action="store_true", help="v1(209) 구성: TRAIN_BAL_LOA2 링크 제외")
    args = ap.parse_args()
    use_loa2 = not args.no_loa2 and os.path.isdir(f"{PF}/TRAIN_BAL_LOA2")
    assert os.path.isdir(f"{PF}/TRAIN_BAL") and len(glob.glob(f"{PF}/TRAIN_BAL/subject*/meta.json")) == 36, "TRAIN_BAL incomplete"
    os.makedirs(OUT, exist_ok=True)
    # 1) 실 clean — COTRAIN_ALL 과 동일한 링크를 그대로 복제(분할 동일성 보장)
    src_all = f"{PF}/COTRAIN_ALL"
    for s in sorted(glob.glob(f"{src_all}/subject*")):
        n = int(os.path.basename(s)[7:])
        tgt = os.path.realpath(s)
        if n < 200 or 300 <= n < 400 or 500 <= n < 600 or 700 <= n < 900 or n >= 90000:
            link(os.path.basename(s), tgt)
        # 201~226(TRAIN20H/TRAIN6HI), 601~610(CMP_SYN of TRAIN20H) 는 제외 → TRAIN_BAL 로 대체
    # 2) 합성 균형 풀
    metas = load_metas(f"{PF}/TRAIN_BAL")
    for i in metas:
        link(f"subject{1000 + i}", f"{PF}/TRAIN_BAL/subject{i}")
    # 2b) v2 저 a2 보강 (subject1201~1212; 1101~1106 은 TRAIN6LO 옵션용으로 비워 둠)
    metas_lo = {}
    if use_loa2:
        metas_lo = load_metas(f"{PF}/TRAIN_BAL_LOA2")
        assert len(metas_lo) == 12, len(metas_lo)
        for i in metas_lo:
            link(f"subject{1200 + i}", f"{PF}/TRAIN_BAL_LOA2/subject{i}")
    else:
        for p in glob.glob(f"{OUT}/subject12[0-9][0-9]"):
            os.remove(p)
    # 3) 합성 clean 12클립 압축 사본 (기존 CMP_SYN* 와 동일 규약 q6)
    clean_ids = [i for i, m in metas.items() if not m.get("noise")]
    assert len(clean_ids) == 12, clean_ids
    os.makedirs(CMP, exist_ok=True)
    for k, i in enumerate(clean_ids):
        d = f"{CMP}/subject{601 + k}"
        if not os.path.exists(f"{d}/vid.avi"):  # 기존 사본은 재생성하지 않음
            compress_copy(f"{PF}/TRAIN_BAL/subject{i}", d, q=6)
            with open(f"{d}/source.txt", "w") as f:
                f.write(f"TRAIN_BAL/subject{i} mpeg4 q6 bpm={metas[i]['achieved_bpm']}\n")
        link(f"subject{601 + k}", d)

    # ---- 검증 ----
    rows = []
    bad = []
    for s in sorted(glob.glob(f"{OUT}/subject*"), key=lambda x: int(os.path.basename(x)[7:])):
        ok = os.path.exists(f"{s}/vid.avi") and os.path.exists(f"{s}/ground_truth.txt")
        if not ok:
            bad.append(os.path.basename(s))
        rows.append({"clip": os.path.basename(s), "target": os.path.relpath(os.path.realpath(s), ROOT), "ok": ok})
    ver = verify_synth("TRAIN_BAL", metas, 1000) + (verify_synth("TRAIN_BAL_LOA2", metas_lo, 1200) if use_loa2 else [])
    # ---- 균형표 (실:합성, clean:noisy, BPM·a2 히스토그램) ----
    def clip_info(r):
        t = r["target"]
        if "/TRAIN_BAL_LOA2/" in t:
            m = metas_lo[int(t.split("subject")[-1])]; return "synth", ("clean" if not m.get("noise") else "noisy"), m["achieved_bpm"], m["wave_a2"]
        if "/TRAIN_BAL/" in t:
            m = metas[int(t.split("subject")[-1])]; return "synth", ("clean" if not m.get("noise") else "noisy"), m["achieved_bpm"], m["wave_a2"]
        if "/CMP_SYN_BAL/" in t:
            m = metas[int(open(f"{ROOT}/{t}/source.txt").read().split()[0].split("subject")[-1])]; return "synth", "noisy", m["achieved_bpm"], m["wave_a2"]
        dom = "noisy" if ("/COTRAIN_NOISY_REAL/" in t or "/CMP_REAL/" in t) else "clean"
        return "real", dom, None, None
    bal = {"n": len(rows), "real": 0, "synth": 0, "real_clean": 0, "real_noisy": 0, "synth_clean": 0, "synth_noisy": 0,
           "synth_bpm_hist": Counter(), "synth_a2_hist": Counter(), "synth_cond": Counter()}
    for r in rows:
        src, cond, bpm, a2 = clip_info(r)
        bal[src] += 1; bal[f"{src}_{cond}"] += 1
        if src == "synth":
            bal["synth_bpm_hist"][f"{int(bpm // 10) * 10}-{int(bpm // 10) * 10 + 10}"] += 1
            bal["synth_a2_hist"][f"{int(a2 * 10) / 10:.1f}-{int(a2 * 10) / 10 + 0.1:.1f}"] += 1
    for v in ver:
        bal["synth_cond"][v["cond"]] += 1
    bal["synth_cond"]["compressed(q6)"] = sum(1 for r in rows if "/CMP_SYN_BAL/" in r["target"])
    bal = {k: (dict(sorted(v.items(), key=lambda kv: float(kv[0].split("-")[0]))) if isinstance(v, Counter) and k != "synth_cond" else (dict(v) if isinstance(v, Counter) else v)) for k, v in bal.items()}
    # 누설 검사
    ident = {"UBFC": set(), "MCD": set(), "PURE": set(), "SFHQ": set(), "seeds": set()}
    for r in rows:
        t = r["target"]
        if t.startswith("UBFC_dataset"):
            ident["UBFC"].add(int(t.split("subject")[-1]))
        elif "/COTRAIN_NOISY_REAL/" in t or "/CMP_REAL/" in t:
            pass  # 아래에서 원본 UBFC 번호로 환원
        elif "/MCD_TRAIN/" in t:
            ident["MCD"].add(open(f"{ROOT}/{t}/source.txt").read().split()[0].split("_")[0])
        elif "/PURE_TRAIN/" in t:
            ident["PURE"].add(open(f"{ROOT}/{t}/source.txt").read().split()[1][:2])
        elif "/TRAIN_BAL/" in t or "/CMP_SYN_BAL/" in t:
            pass
    for r in rows:  # NOISY_REAL/CMP_REAL 원본 번호
        t = r["target"]
        if "/COTRAIN_NOISY_REAL/" in t:
            ident["UBFC"].add(int(t.split("subject")[-1]))
        if "/CMP_REAL/" in t:  # CMP_REAL 501~510 = UBFC 1,3,4,5,8,9,10,11,12,13 (18 보고서)
            ident["UBFC"].add([1, 3, 4, 5, 8, 9, 10, 11, 12, 13][int(t.split("subject")[-1]) - 501])
    for m in list(metas.values()) + list(metas_lo.values()):
        ident["SFHQ"].add(os.path.basename(m["frames_dir"])); ident["seeds"].add(m["seed"])
    heldout = {
        "UBFC(REALTEST/RTNC/RTNI)": set(range(42, 50)),
        "MCD_A/B": set(open(p).read().split()[0].split("_")[0] for p in glob.glob(f"{PF}/MCD_A/subject*/source.txt") + glob.glob(f"{PF}/MCD_B/subject*/source.txt")),
        "PURE 07-10": {"07", "08", "09", "10"},
        "synth faces(C/E)": set(), "synth seeds(C/E)": set()}
    for c in glob.glob(f"{PF}/configs/test_*.json"):
        for s in json.load(open(c))["subjects"]:
            heldout["synth faces(C/E)"].add(os.path.basename(s["frames_dir"])); heldout["synth seeds(C/E)"].add(s["seed"])
    leak = {"UBFC": sorted(ident["UBFC"] & heldout["UBFC(REALTEST/RTNC/RTNI)"]),
            "MCD": sorted(ident["MCD"] & heldout["MCD_A/B"]),
            "PURE": sorted(ident["PURE"] & heldout["PURE 07-10"]),
            "SFHQ_faces": sorted(ident["SFHQ"] & heldout["synth faces(C/E)"]),
            "seeds": sorted(ident["seeds"] & heldout["synth seeds(C/E)"])}
    man = {"version": "v2(+LOA2)" if use_loa2 else "v1", "n_links": len(rows), "bad": bad, "links": rows, "train_bal_verification": ver,
           "balance": bal, "leak_check": leak,
           "train_identities": {k: sorted(v) for k, v in ident.items()},
           "heldout_identities": {k: sorted(v) for k, v in heldout.items()}}
    os.makedirs(f"{ROOT}/_workspace/data/results", exist_ok=True)
    json.dump(man, open(f"{ROOT}/_workspace/data/results/cotrain_bal_manifest.json", "w"), indent=1, ensure_ascii=False)
    print(f"COTRAIN_BAL links={len(rows)} bad={bad}")
    print("leak:", leak)
    print("balance:", json.dumps(bal, ensure_ascii=False))
    print("| src | id | link | BPM 목표 | achieved | peak=f0 | cond | a2 req | a2 fit | a3 req | a3 fit | frames |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for v in ver:
        print(f"| {v['src']} | {v['id']} | {v['link']} | {v['bpm_target']} | {v['achieved']} | {'Y' if v['peak_is_f0'] else 'N'} | {v['cond']} | {v['a2_req']} | {v['a2_fit']} | {v['a3_req']} | {v['a3_fit']} | {v['ffprobe']} |")


if __name__ == "__main__":
    main()
