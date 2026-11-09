#!/usr/bin/env python3
"""add_noise.py — rPPG 노이즈 시뮬레이션: 얼굴 영상에 통제된 광학/움직임 노이즈를 추가한다.

노이즈 모델 (연구자가 정의한 rPPG 3대 노이즈원을 조작 가능하게):
  1) 조명(illumination): 저주파 드리프트(랜덤워크 평활) + 주기적 플리커(심박 대역과
     겹칠 수 있는 주파수 — rPPG를 교란하는 최악 시나리오를 통제 재현)
  2) 강체 움직임(rigid motion): 프레임별 평활 랜덤워크 아핀(평행이동+회전+스케일)
     — 고개/머리 이동 모사. 주입된 맥동은 픽셀에 있으므로 얼굴과 함께 움직인다(물리적으로 옳음)
  3) 센서 노이즈: 픽셀별 가우시안

적용 순서: 심박 주입 **후**에 적용한다 — 실제 세계에서 노이즈는 생리신호 위에 얹힌다.
결정론: --seed로 동일 노이즈 궤적 재생성. 궤적은 gt_dir에 CSV로 기록해 감사 가능.
"""
import argparse, csv, glob, os
import cv2
import numpy as np


def smooth_walk(n, sigma, smooth_win, rng):
    """평활 랜덤워크: 저주파 자연 드리프트 궤적."""
    w = np.cumsum(rng.normal(0, sigma, n))
    k = max(3, int(smooth_win) | 1)
    pad = np.pad(w, k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")[:n]


def add_noise(frames_dir, out_dir, fps, seed=0,
              illum_drift=0.0, flicker_amp=0.0, flicker_hz=1.2,
              motion_px=0.0, rot_deg=0.0, scale_jit=0.0,
              sensor_std=0.0, log_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"no frames in {frames_dir}")
    n = len(paths)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps

    # 궤적 생성 (모두 결정론적)
    drift = smooth_walk(n, illum_drift / max(np.sqrt(n), 1), int(fps), rng) if illum_drift > 0 else np.zeros(n)
    drift = drift - drift.mean()
    flicker = flicker_amp * np.sin(2 * np.pi * flicker_hz * t) if flicker_amp > 0 else np.zeros(n)
    gain = np.clip(1.0 + drift + flicker, 0.3, 1.7)
    tx = smooth_walk(n, motion_px / max(np.sqrt(n), 1), int(fps // 2), rng) if motion_px > 0 else np.zeros(n)
    ty = smooth_walk(n, motion_px / max(np.sqrt(n), 1), int(fps // 2), rng) if motion_px > 0 else np.zeros(n)
    rot = smooth_walk(n, rot_deg / max(np.sqrt(n), 1), int(fps // 2), rng) if rot_deg > 0 else np.zeros(n)
    sc = 1.0 + (smooth_walk(n, scale_jit / max(np.sqrt(n), 1), int(fps // 2), rng) if scale_jit > 0 else np.zeros(n))

    h, w = cv2.imread(paths[0]).shape[:2]
    ctr = (w / 2, h / 2)
    for i, p in enumerate(paths):
        img = cv2.imread(p).astype(np.float32)
        if motion_px > 0 or rot_deg > 0 or scale_jit > 0:
            M = cv2.getRotationMatrix2D(ctr, float(rot[i]), float(sc[i]))
            M[:, 2] += (tx[i], ty[i])
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        img = img * gain[i]
        if sensor_std > 0:
            img = img + rng.normal(0, sensor_std, img.shape)
        cv2.imwrite(os.path.join(out_dir, f"{i:06d}.png"),
                    np.clip(img, 0, 255).astype(np.uint8))

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "noise_traj.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "gain", "tx", "ty", "rot_deg", "scale"])
            for i in range(n):
                wr.writerow([i, round(float(gain[i]), 5), round(float(tx[i]), 3),
                             round(float(ty[i]), 3), round(float(rot[i]), 4),
                             round(float(sc[i]), 5)])
    print(f"[noise] {n} frames | illum={illum_drift}/{flicker_amp}@{flicker_hz}Hz "
          f"motion={motion_px}px/{rot_deg}deg sensor={sensor_std} -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="rPPG 노이즈 시뮬레이션 (주입 후 적용)")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--illum_drift", type=float, default=0.0, help="조명 드리프트 진폭 (예 0.15)")
    ap.add_argument("--flicker_amp", type=float, default=0.0, help="플리커 진폭 (예 0.03)")
    ap.add_argument("--flicker_hz", type=float, default=1.2, help="플리커 주파수(Hz, 심박대역 교란)")
    ap.add_argument("--motion_px", type=float, default=0.0, help="이동 지터 강도(px, 예 8)")
    ap.add_argument("--rot_deg", type=float, default=0.0, help="회전 지터 강도(도, 예 1.5)")
    ap.add_argument("--scale_jit", type=float, default=0.0, help="스케일 지터 (예 0.02)")
    ap.add_argument("--sensor_std", type=float, default=0.0, help="센서 가우시안 σ (예 2)")
    ap.add_argument("--log_dir", default=None, help="노이즈 궤적 CSV 저장 위치")
    a = ap.parse_args()
    add_noise(a.frames_dir, a.out_dir, a.fps, a.seed, a.illum_drift,
              a.flicker_amp, a.flicker_hz, a.motion_px, a.rot_deg,
              a.scale_jit, a.sensor_std, a.log_dir)
