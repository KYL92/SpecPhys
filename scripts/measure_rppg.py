#!/usr/bin/env python3
"""measure_rppg.py — 주입 영상에서 rPPG-Toolbox의 실제 POS/CHROM으로 심박을 복원하고
정답(GT) BPM과 비교(MAE)하며 SNR을 계산한다. 무신호 clean-plate 대조도 수행한다.

사용:
  python measure_rppg.py --frames <injected_png_dir> --fps 30 --gt_bpm 72 \
      --baseline <clean_plate_png_dir>
"""
import argparse, glob, os, sys
import numpy as np
import cv2

# rPPG-Toolbox는 numpy<2의 np.mat을 사용 → numpy 2.x 호환 shim
if not hasattr(np, "mat"):
    np.mat = np.asmatrix

RPPG_REPO = "/SPECPHYS_ROOT/repos/rPPG-Toolbox"
sys.path.insert(0, RPPG_REPO)
from unsupervised_methods.methods.POS_WANG import POS_WANG          # noqa
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN  # noqa
from evaluation.post_process import _calculate_fft_hr, _calculate_SNR  # noqa


def load_skin_mean_frames(frames_dir, mask_path):
    """각 프레임의 피부 ROI 공간 평균 RGB를 계산해 (1,1,3) 프레임 스택으로 반환한다.
    실제 rPPG의 skin-ROI 공간 평균과 동일 — POS/CHROM의 내부 프레임 평균에 그대로 대응."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
    out = []
    for p in sorted(glob.glob(os.path.join(frames_dir, "*.png"))):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32)
        mean_rgb = img[mask].mean(axis=0)                # 피부 픽셀만 평균
        out.append(mean_rgb.reshape(1, 1, 3))
    return out


def hr_and_snr(frames_rgb, fs):
    results = {}
    for name, fn in (("POS", POS_WANG), ("CHROM", CHROME_DEHAAN)):
        bvp = np.asarray(fn(frames_rgb, fs)).squeeze()
        hr = _calculate_fft_hr(bvp, fs=fs, low_pass=0.7, high_pass=4.0)
        snr = _calculate_SNR(bvp, hr, fs=fs, low_pass=0.7, high_pass=4.0)
        results[name] = (float(hr), float(snr))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="주입 프레임 폴더")
    ap.add_argument("--mask", required=True, help="피부 ROI 마스크 PNG")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--gt_bpm", type=float, required=True)
    ap.add_argument("--baseline", default=None, help="무신호 clean-plate 폴더(대조)")
    a = ap.parse_args()

    print(f"\n=== 주입 영상 측정 (GT = {a.gt_bpm:.0f} BPM) ===")
    inj = load_skin_mean_frames(a.frames, a.mask)
    print(f"[info] {len(inj)} frames @ {a.fps} fps ({len(inj)/a.fps:.1f}s), 피부 ROI 공간평균")
    r_inj = hr_and_snr(inj, a.fps)
    for m, (hr, snr) in r_inj.items():
        print(f"  {m:5s}  추정HR={hr:6.1f} BPM   MAE={abs(hr-a.gt_bpm):5.1f}   SNR={snr:6.2f} dB")

    if a.baseline:
        print(f"\n=== 무신호 원본 대조 (HR 검출되면 안 됨) ===")
        base = load_skin_mean_frames(a.baseline, a.mask)
        r_base = hr_and_snr(base, a.fps)
        for m, (hr, snr) in r_base.items():
            print(f"  {m:5s}  추정HR={hr:6.1f} BPM   SNR={snr:6.2f} dB")
        print("\n=== 정합성 판정 ===")
        for m in r_inj:
            hi = r_inj[m][0]; hb = r_base[m][0]
            ok = abs(hi - a.gt_bpm) < 6
            print(f"  {m}: 주입={hi:.1f} (GT {a.gt_bpm:.0f}), 원본={hb:.1f}  ->  "
                  f"{'PASS (주입 신호 정확 복원)' if ok else 'CHECK'}")


if __name__ == "__main__":
    main()
