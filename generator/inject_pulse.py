#!/usr/bin/env python3
"""
inject_pulse.py — 합성 얼굴 영상 피부 영역에 물리 기반(POS/CHROM Pbv) 심박 맥파를
픽셀단위로 주입하고, 정답(ground-truth) PPG/HR을 저장한다.

물리 모델 (Wang et al. 2017 "Algorithmic Principles of Remote PPG"; de Haan 2013):
  피부 반사의 맥동 성분은 혈액 흡광에 따라 고정된 blood-volume-pulse 서명 벡터
  Pbv ≈ [0.33, 0.77, 0.53] (R,G,B; 녹색 최대) 방향으로 색을 변화시킨다.
  주입:  I_c(x,y,t) <- I_c(x,y,t) * (1 + alpha * p(t) * Pbv_c)
  - p(t): 단위 진폭 PPG 파형 (목표 BPM)
  - alpha: 변조 깊이 (실제 rPPG AC/DC << 1% → 0.002~0.01 권장)

재현성 원칙:
  - 시드 고정으로 파형/노이즈 결정론적 재생성
  - 정답 PPG는 실제 픽셀에 적용된 파형과 위상이 정확히 일치 (평가 기준)
  - 무손실 저장(PNG 프레임 스택)만 사용 — 압축은 <1% 변조를 파괴한다

의존성: numpy, opencv-python, scipy. (선택) neurokit2로 현실적 PPG 파형.
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

# 정규화된 상대 RGB 맥동 진폭 (헤모글로빈 흡광 기반). 입력이 BGR이면 순서 반전.
PBV_RGB = np.array([0.33, 0.77, 0.53], dtype=np.float32)


def make_waveform(n_frames, fps, bpm, hrv_std=0.0, use_neurokit=True, seed=0,
                  wave_a2=None, wave_phi2=0.0, wave_a3=0.0, wave_phi3=0.0):
    """목표 BPM의 단위 진폭 PPG 파형 p(t)와 프레임별 순시 HR을 반환한다.

    wave_a2가 주어지면 해석적 고조파 파형을 사용:
        p(θ) = sin(θ) + a2·sin(2θ+φ2) + a3·sin(3θ+φ3),  θ = 2π·(bpm/60)·t
    a2를 클립별로 다양화(0~0.85)하면 '2차 고조파가 강해도 기본 주파수가 심박'
    이라는 구분을 학습시킬 수 있다 — 교차 데이터셋 고조파 배가 실패의 직접 처방.
    a2 ≤ 0.85로 제한(스펙트럼 지배 주파수가 기본파로 유지되는 범위)."""
    rng = np.random.default_rng(seed)
    duration = n_frames / fps
    if wave_a2 is not None:
        a2 = min(float(wave_a2), 0.85)
        inst_hr = np.full(n_frames, float(bpm)) + \
            (rng.normal(0, hrv_std, size=n_frames) if hrv_std > 0 else 0.0)
        theta = 2 * np.pi * np.cumsum(inst_hr / 60.0) / fps
        ppg = (np.sin(theta) + a2 * np.sin(2 * theta + wave_phi2)
               + wave_a3 * np.sin(3 * theta + wave_phi3))
        ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
        return ppg.astype(np.float32), np.asarray(inst_hr, dtype=np.float32)
    if use_neurokit:
        try:
            import neurokit2 as nk
            # 주의: ppg_simulate 기본값(frequency_modulation=0.2, ibi_randomness=0.1,
            # motion/powerline 노이즈)은 실제 주입 주파수를 목표 BPM에서 이탈시킨다.
            # GT 정확성이 실험의 근간이므로 통제 성분은 모두 0으로 끄고,
            # HRV는 hrv_std 파라미터로만 명시적으로 준다.
            fm = (hrv_std / bpm) if hrv_std > 0 else 0.0
            ppg = nk.ppg_simulate(duration=duration, sampling_rate=fps,
                                  heart_rate=float(bpm),
                                  frequency_modulation=fm, ibi_randomness=0.0,
                                  drift=0, motion_amplitude=0.0,
                                  powerline_amplitude=0.0,
                                  random_state=seed)[:n_frames]
            ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
            inst_hr = np.full(n_frames, bpm, dtype=np.float32)
            return ppg.astype(np.float32), inst_hr
        except Exception:
            pass  # neurokit 미설치 → 정현파 폴백
    # 정현파 폴백 (HRV는 순시 주파수 적분으로 반영)
    t = np.arange(n_frames) / fps
    inst_hr = bpm + (rng.normal(0, hrv_std, size=n_frames) if hrv_std > 0 else 0.0)
    inst_freq = inst_hr / 60.0
    phase = 2 * np.pi * np.cumsum(inst_freq) / fps
    ppg = np.sin(phase).astype(np.float32)
    return ppg, np.asarray(inst_hr, dtype=np.float32)


def load_skin_mask(mask_path, shape):
    """피부 마스크(0/255 또는 파싱맵)를 [0,1] float로 로드. 없으면 중앙 타원 폴백."""
    if mask_path and os.path.exists(mask_path):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return (m > 127).astype(np.float32)
    h, w = shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 4, h // 3), 0, 0, 360, 1.0, -1)
    return mask


def feather(mask, ksize=21):
    """하드 경계 아티팩트 방지: 마스크 경계를 가우시안으로 완화."""
    k = ksize | 1
    return cv2.GaussianBlur(mask, (k, k), 0)


def spatial_amplitude_map(shape, mode="uniform", coverage=1.0, patch_scale=48,
                          amp_mu=1.0, amp_sigma=0.35, seed=0):
    """공간 진폭 맵 A(x,y) ∈ [0,1+]. 피부 관류의 공간 이질성(코·지방 부위의 약한 맥동)을 모사.

    uniform : A=1 (기존 동작)
    patches : 가우시안 랜덤필드를 patch_scale로 평활 → 상위 coverage 비율만 pulsatile로
              선택하고, 선택 영역의 진폭을 N(amp_mu, amp_sigma)에서 샘플(0 이상 클립).
              나머지는 비pulsatile(A=0). 경계는 feather로 완화.
    """
    h, w = shape[:2]
    if mode == "uniform" or coverage >= 0.999:
        return np.ones((h, w), np.float32)
    rng = np.random.default_rng(seed)
    field = rng.normal(0, 1, (h, w)).astype(np.float32)
    k = int(patch_scale) | 1
    field = cv2.GaussianBlur(field, (k, k), 0)
    thr = np.quantile(field, 1.0 - coverage)          # 상위 coverage 비율이 pulsatile
    sel = (field >= thr).astype(np.float32)
    amp = np.clip(rng.normal(amp_mu, amp_sigma, (h, w)), 0.0, None).astype(np.float32)
    amp = cv2.GaussianBlur(amp, (k, k), 0)            # 진폭도 공간적으로 매끄럽게
    return feather(sel, ksize=k) * amp


def inject(frames_dir, out_dir, gt_dir, fps, bpm, alpha, hrv_std, seed,
           mask_path, is_bgr=True, spatial="uniform", coverage=1.0,
           patch_scale=48, amp_mu=1.0, amp_sigma=0.35):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")) +
                   glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not paths:
        raise FileNotFoundError(f"no frames in {frames_dir}")
    n = len(paths)
    ppg, inst_hr = make_waveform(n, fps, bpm, hrv_std, seed=seed)

    first = cv2.imread(paths[0])
    skin = feather(load_skin_mask(mask_path, first.shape))
    amap = spatial_amplitude_map(first.shape, spatial, coverage,
                                 patch_scale, amp_mu, amp_sigma, seed)
    mask = (skin * amap)[..., None]                    # (H,W,1) = 피부 ∩ 공간진폭
    cv2.imwrite(os.path.join(gt_dir, "amplitude_map.png"),
                np.clip(mask[..., 0] * 255, 0, 255).astype(np.uint8))
    pbv = PBV_RGB[::-1] if is_bgr else PBV_RGB  # OpenCV는 BGR

    for i, p in enumerate(paths):
        img = cv2.imread(p).astype(np.float32)
        gain = 1.0 + alpha * ppg[i] * pbv.reshape(1, 1, 3)  # (1,1,3)
        out = img * (1.0 + mask * (gain - 1.0))             # 피부×진폭맵에 적용
        out = np.clip(out, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{i:06d}.png"), out)  # 무손실 PNG

    # 정답 저장 — rPPG 복원값과 비교의 절대 기준
    with open(os.path.join(gt_dir, "gt_ppg.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "ppg", "inst_hr_bpm"])
        for i in range(n):
            w.writerow([i, i / fps, float(ppg[i]), float(inst_hr[i])])
    # 실제 달성 BPM 검증: 파형의 지배 주파수를 제로패딩 FFT로 측정해 기록
    freqs = np.fft.rfftfreq(n * 8, 1.0 / fps)
    power = np.abs(np.fft.rfft(ppg - ppg.mean(), n=n * 8)) ** 2
    band = (freqs >= 0.7) & (freqs <= 4.0)
    achieved_bpm = float(freqs[band][np.argmax(power[band])] * 60.0)
    if abs(achieved_bpm - bpm) > 1.0:
        print(f"[inject][WARN] 달성 BPM {achieved_bpm:.2f} != 목표 {bpm} — GT 라벨은 달성값 기준으로 평가할 것")
    with open(os.path.join(gt_dir, "gt_meta.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fps", "n_frames", "target_bpm", "achieved_bpm", "alpha", "hrv_std", "seed", "pbv_rgb"])
        w.writerow([fps, n, bpm, round(achieved_bpm, 2), alpha, hrv_std, seed, list(PBV_RGB)])
    print(f"[inject] {n} frames | target={bpm} achieved={achieved_bpm:.2f} BPM alpha={alpha} -> {out_dir}")
    print(f"[inject] ground-truth -> {gt_dir}/gt_ppg.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="픽셀단위 심박 맥파 주입 (POS/CHROM Pbv 모델)")
    ap.add_argument("--frames_dir", required=True, help="원본 무손실 프레임 폴더")
    ap.add_argument("--out_dir", required=True, help="주입 프레임 출력 폴더")
    ap.add_argument("--gt_dir", required=True, help="정답 PPG/HR 출력 폴더")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--bpm", type=float, default=72.0)
    ap.add_argument("--alpha", type=float, default=0.005, help="변조 깊이 (0.002~0.01 권장)")
    ap.add_argument("--hrv_std", type=float, default=0.0, help="심박변이 표준편차(BPM)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask", default=None, help="피부 마스크 PNG (없으면 타원 폴백)")
    ap.add_argument("--rgb", action="store_true", help="입력이 RGB이면 지정(기본 BGR)")
    ap.add_argument("--spatial", choices=["uniform", "patches"], default="uniform",
                    help="공간 진폭 맵: uniform=균일, patches=가우시안 관류 이질성")
    ap.add_argument("--coverage", type=float, default=1.0, help="pulsatile 픽셀 비율 (patches)")
    ap.add_argument("--patch_scale", type=int, default=48, help="패치 공간 스케일(px)")
    ap.add_argument("--amp_mu", type=float, default=1.0)
    ap.add_argument("--amp_sigma", type=float, default=0.35)
    a = ap.parse_args()
    inject(a.frames_dir, a.out_dir, a.gt_dir, a.fps, a.bpm, a.alpha,
           a.hrv_std, a.seed, a.mask, is_bgr=not a.rgb, spatial=a.spatial,
           coverage=a.coverage, patch_scale=a.patch_scale,
           amp_mu=a.amp_mu, amp_sigma=a.amp_sigma)
