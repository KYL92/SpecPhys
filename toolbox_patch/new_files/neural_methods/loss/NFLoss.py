"""[Virtual-Face patch 37, 47·48 문서] NF-Phys 손실 보조 함수 — 모두 (B,T) 텐서 입력, 샘플별 값 (B,) 반환.
  hp            1 s 이동평균 제거(고역 통과)
  power         30–200 bpm 1 bpm 격자의 DFT 전력(정확한 bin, PhysFormer complex_absolute 와 같은 방식)
  spec_nll      −log(정규화 전력의 라벨 심박 ±width bpm 질량): 출력 파형·맥파 슬롯의 스펙트럼 손실
  half_penalty  라벨 심박/2 근방 질량 / (심박/2 질량 + 심박 질량): 반감(halving) 억제
  spec_cos      1 − cos(정규화 전력 스펙트럼): 교란 슬롯 vs 교란 trace (위상 무관)
  corr2         시간 영역 상관²: 맥파 슬롯 vs 교란 슬롯 탈상관
  label_hr      라벨 청크의 8× 영패딩 FFT 피크(0.5–3.3 Hz) → bpm (사이드카 없을 때의 심박 라벨)
[패치 38, 53 문서] 자기 지도 분리 신호(교란 정답 없는 실 클립):
  hr_band_mask  라벨 심박 대역(기본·반·배 ±width bpm) 마스크 — 맥파가 있을 수 있는 주파수
  nuis_mask     클립 자신의 교란 프록시(전역 밝기 차분 trace, 전역 이동 속도 vx·vy) 정규화 전력의 최대, 심박 대역 제외 → 교란 마스크 M (샘플별 최대 1)
  spec_excl     출력 파형의 정규화 전력이 M 위에 놓인 질량(스펙트럼 배제 손실)
  motion_traces 전역 평행이동 속도(vx, vy; Lucas–Kanade 1차 해, 부호 있음) — 움직임 교란 프록시
  nuis_target_spec / spec_cos_target  심박 대역을 지운 전역 trace 스펙트럼을 교란 슬롯의 표적으로(스펙트럼 코사인)
  skin_map      시간 평균 프레임의 YCrCb 피부 규칙 → 어텐션 격자(h,w) 피부 비율
  cell_traces / cell_nuis_ratio  어텐션 격자 셀별 밝기 차분 trace → 셀별 교란 질량(샘플별 최대 1)
[패치 39, 55 문서] 후보 ①③:
  cell_motion / local_nuis_mask  격자 셀별 밝기 차분 + 셀별 LK 이동 속도의 정규화 전력 → 주파수별 세기 × 공간 비일관성(셀 간 std/mean) → 국소 교란 마스크(심박 대역 제외)
  make_aug      실 클립에 알려진 교란(플리커+드리프트 곱셈, 주기 평행이동, 센서 잡음) 주입 — 주입 주파수는 라벨 심박 대역을 피함; 주입 교란 trace 반환
  pearson       샘플별 피어슨 상관 (증강 일관성)
[패치 40, 55 문서 ④] label_hr_tb  정본 readout 과 같은 전처리(detrend λ=100 → Butterworth 1차 0.6–3.3 Hz)를 거친 청크 심박 라벨 — 뾰족한 맥파(MMPD)에서 naive FFT 피크가 2배 조화파를 고르는 결함 교정 (NF_HR_TB=1)"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def hp(x, k=31):
    m = F.avg_pool1d(x.unsqueeze(1), k, stride=1, padding=k // 2, count_include_pad=False)[:, 0, :x.shape[-1]]
    return x - m


def power(x, fs, lo=30, hi=200, step=1.0):
    T = x.shape[-1]
    t = torch.arange(T, device=x.device, dtype=x.dtype) / float(fs)
    bpm = torch.arange(lo, hi, step, device=x.device, dtype=x.dtype)
    ang = 2.0 * math.pi * (bpm / 60.0)[:, None] * t[None, :]            # (nb,T)
    xz = x - x.mean(-1, keepdim=True)
    re = xz @ torch.cos(ang).T
    im = xz @ torch.sin(ang).T
    return re ** 2 + im ** 2, bpm


def _norm(P):
    return P / (P.sum(-1, keepdim=True) + 1e-8)


def spec_nll(x, hr, fs, lo=30, hi=200, width=3.0):
    P, bpm = power(x, fs, lo, hi)
    P = _norm(P)
    w = (torch.abs(bpm[None, :] - hr[:, None]) <= width).to(P.dtype)
    return -torch.log((P * w).sum(-1) + 1e-6)


def half_penalty(x, hr, fs, lo=30, hi=200, width=3.0):
    P, bpm = power(x, fs, lo, hi)
    P = _norm(P)
    wh = (torch.abs(bpm[None, :] - hr[:, None] / 2.0) <= width).to(P.dtype) * (hr[:, None] / 2.0 >= lo).to(P.dtype)
    w1 = (torch.abs(bpm[None, :] - hr[:, None]) <= width).to(P.dtype)
    mh = (P * wh).sum(-1)
    m1 = (P * w1).sum(-1)
    return mh / (mh + m1 + 1e-6)


def double_penalty(x, hr, fs, lo=30, hi=200, width=3.0):
    """[Virtual-Face patch 45, 60 문서 §7] 배가 억제: 2·HR 대역 질량 / (2·HR 대역 + HR 대역), 2·HR 이 격자 안(≤hi)일 때만 활성 (half_penalty 의 대칭)."""
    P, bpm = power(x, fs, lo, hi)
    P = _norm(P)
    wd = (torch.abs(bpm[None, :] - hr[:, None] * 2.0) <= width).to(P.dtype) * (hr[:, None] * 2.0 <= hi).to(P.dtype)
    w1 = (torch.abs(bpm[None, :] - hr[:, None]) <= width).to(P.dtype)
    md = (P * wd).sum(-1)
    m1 = (P * w1).sum(-1)
    return md / (md + m1 + 1e-6)


def spec_cos(a, b, fs, lo=30, hi=240):
    Pa, _ = power(a, fs, lo, hi)
    Pb, _ = power(b, fs, lo, hi)
    return 1.0 - F.cosine_similarity(Pa, Pb, dim=-1, eps=1e-8)


def corr2(a, b):
    az = a - a.mean(-1, keepdim=True)
    bz = b - b.mean(-1, keepdim=True)
    return ((az * bz).sum(-1) / (az.norm(dim=-1) * bz.norm(dim=-1) + 1e-6)) ** 2


def label_hr(label, fs, lo=0.5, hi=3.3):
    x = label - label.mean(-1, keepdim=True)
    n = x.shape[-1]
    nfft = 8 * n
    P = torch.fft.rfft(x, n=nfft, dim=-1).abs() ** 2
    f = torch.fft.rfftfreq(nfft, 1.0 / float(fs)).to(x.device)
    P = P.masked_fill(((f < lo) | (f > hi))[None, :], 0.0)
    return f[P.argmax(-1)] * 60.0


# ---- [Virtual-Face patch 38, 53 문서] 자기 지도 분리 신호 보조 함수 ----
def hr_band_mask(bpm, hr, width=5.0, harmonics=(1.0, 0.5, 2.0)):
    """(B,nb) 1 = 라벨 심박 대역(기본·반·배 주파수 ±width bpm)."""
    m = torch.zeros(hr.shape[0], bpm.shape[0], device=bpm.device, dtype=bpm.dtype)
    for h in harmonics:
        m = torch.maximum(m, (torch.abs(bpm[None, :] - hr[:, None] * h) <= width).to(m.dtype))
    return m


def nuis_mask(traces, hr, fs, width=5.0, lo=30, hi=200):
    """교란 프록시 trace 들(각 (B,T)) 의 정규화 전력(1 s 고역 통과 후) 샘플별 최대 → 심박 대역 제외 → (B,nb), 샘플별 최대 1."""
    M = None
    for tr in traces:
        P, bpm = power(hp(tr), fs, lo, hi); P = _norm(P)
        M = P if M is None else torch.maximum(M, P)
    M = M * (1.0 - hr_band_mask(bpm, hr, width))
    return M / (M.max(-1, keepdim=True)[0] + 1e-8)


def spec_excl(x, M, fs, lo=30, hi=200):
    """출력 x (B,T) 의 정규화 전력이 교란 마스크 M (B,nb) 위에 놓인 질량 (B,)."""
    P, _ = power(x, fs, lo, hi)
    return (_norm(P) * M).sum(-1)


def motion_traces(frames):
    """frames (B,3,T,H,W) → 전역 평행이동 속도 (vx, vy) 각 (B,T−1) px/frame — Lucas–Kanade 1차 전역 해(부호 있음: 주기 움직임이 자기 주파수에 놓인다;
    |프레임 차분| 평균은 정류 신호라 2배 주파수에 놓이므로 쓰지 않는다)."""
    g = frames[:, :3].float().mean(1)                                   # (B,T,H,W) luminance
    Ix = (0.5 * (g[:, :, :, 2:] - g[:, :, :, :-2]))[:, :-1, 1:-1, :]
    Iy = (0.5 * (g[:, :, 2:, :] - g[:, :, :-2, :]))[:, :-1, :, 1:-1]
    It = (g[:, 1:] - g[:, :-1])[:, :, 1:-1, 1:-1]
    a = (Ix * Ix).sum((-1, -2)); b = (Ix * Iy).sum((-1, -2)); c = (Iy * Iy).sum((-1, -2))
    sxt = (Ix * It).sum((-1, -2)); syt = (Iy * It).sum((-1, -2))
    det = a * c - b * b + 1e-6
    vx = -(c * sxt - b * syt) / det; vy = -(a * syt - b * sxt) / det
    return vx, vy


def nuis_target_spec(g, hr, fs, width=5.0, lo=30, hi=200):
    """전역 trace g (B,T) 의 정규화 전력에서 라벨 심박 대역(기본·반·배 ±width)을 지운 교란 표적 스펙트럼 (B,nb), 합 1 (detach)."""
    P, bpm = power(hp(g), fs, lo, hi)
    P = _norm(P) * (1.0 - hr_band_mask(bpm, hr, width))
    return _norm(P).detach()


def spec_cos_target(a, Pt, fs, lo=30, hi=200):
    """1 − cos(정규화 전력 스펙트럼(a), 표적 스펙트럼 Pt) (B,) — 교란 슬롯 vs 심박 대역을 지운 전역 trace 스펙트럼."""
    Pa, _ = power(a, fs, lo, hi)
    return 1.0 - F.cosine_similarity(_norm(Pa), Pt, dim=-1, eps=1e-8)


def skin_map(frames, hw):
    """frames (B,3,T,H,W) RGB 0–255 → 시간 평균 프레임의 YCrCb 피부 규칙(Cr 133–173, Cb 77–127) → (B,h,w) 피부 비율 [0,1]."""
    f = frames[:, :3].float().mean(2)
    r, g, b = f[:, 0], f[:, 1], f[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.0; cb = (b - y) * 0.564 + 128.0
    skin = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).to(f.dtype).unsqueeze(1)
    return F.adaptive_avg_pool2d(skin, hw)[:, 0]


def cell_traces(frames, hw):
    """frames (B,3,T,H,W) → 어텐션 격자 셀별 밝기 차분 trace (B, h·w, T−1)."""
    B, C, T, H, W = frames.shape
    x = frames[:, :3].float().mean(1)
    p = F.adaptive_avg_pool2d(x.reshape(B * T, 1, H, W), hw).reshape(B, T, -1)
    return torch.diff(p, dim=1).permute(0, 2, 1)


def cell_nuis_ratio(ct, M, fs, lo=30, hi=200):
    """ct (B,N,T) 셀 trace, M (B,nb) 교란 마스크 → 셀별 교란 질량 Σ_f P̂_cell(f)·M(f), 샘플별 최대 1 → (B,N)."""
    B, N, T = ct.shape
    P, _ = power(hp(ct.reshape(B * N, T)), fs, lo, hi); P = _norm(P).reshape(B, N, -1)
    r = (P * M[:, None, :]).sum(-1)
    return r / (r.max(-1, keepdim=True)[0] + 1e-8)


# ---- [Virtual-Face patch 39, 55 문서] 후보 ① 국소 교란 프록시 / ③ 증강 일관성 ----
def cell_motion(frames, hw):
    """격자 셀별 Lucas–Kanade 평행이동 속도 (각 셀을 독립 영역으로) → vx, vy 각 (B, h·w, T−1) px/frame."""
    g = frames[:, :3].float().mean(1)
    B, T, H, W = g.shape; h, w = hw
    Hc, Wc = (H // h) * h, (W // w) * w; g = g[:, :, :Hc, :Wc]
    g = g.reshape(B, T, h, Hc // h, w, Wc // w).permute(0, 2, 4, 1, 3, 5).reshape(B * h * w, T, Hc // h, Wc // w)
    Ix = (0.5 * (g[:, :, :, 2:] - g[:, :, :, :-2]))[:, :-1, 1:-1, :]
    Iy = (0.5 * (g[:, :, 2:, :] - g[:, :, :-2, :]))[:, :-1, :, 1:-1]
    It = (g[:, 1:] - g[:, :-1])[:, :, 1:-1, 1:-1]
    a = (Ix * Ix).sum((-1, -2)); b = (Ix * Iy).sum((-1, -2)); c = (Iy * Iy).sum((-1, -2))
    sxt = (Ix * It).sum((-1, -2)); syt = (Iy * It).sum((-1, -2)); det = a * c - b * b + 1e-6
    vx = -(c * sxt - b * syt) / det; vy = -(a * syt - b * sxt) / det
    return vx.reshape(B, h * w, T - 1), vy.reshape(B, h * w, T - 1)


def local_nuis_mask(frames, hr, fs, hw=(6, 6), width=5.0, lo=30, hi=200):
    """국소 교란 마스크 (B,nb): 셀별 밝기 차분 + 셀별 이동 속도의 정규화 전력 → 주파수별 평균 세기 × 공간 비일관성(셀 간 std/mean) → 심박 대역 제외, 샘플별 최대 1.
    셀 간 일관된 전역 교란(균일 조명)은 비일관성이 낮아 억제되고, 일부 셀에만 있는 교란(국소 그림자·손·머리카락·국소 움직임)이 강조된다."""
    ct = cell_traces(frames, hw); vx, vy = cell_motion(frames, hw)
    B, N, T = ct.shape; P = []
    for tr in (ct, vx, vy):
        Pk, bpm = power(hp(tr.reshape(B * N, T)), fs, lo, hi); P.append(_norm(Pk).reshape(B, N, -1))
    P = torch.cat(P, 1)
    mean = P.mean(1); inc = P.std(1) / (mean + 1e-8)
    M = mean * inc * (1.0 - hr_band_mask(bpm, hr, width))
    return M / (M.max(-1, keepdim=True)[0] + 1e-8)


def pearson(a, b):
    az = a - a.mean(-1, keepdim=True); bz = b - b.mean(-1, keepdim=True)
    return (az * bz).sum(-1) / (az.norm(dim=-1) * bz.norm(dim=-1) + 1e-6)


def make_aug(frames, hr, fs, flick=(0.8, 2.5), amp=(0.01, 0.04), drift_max=0.05, mot_hz=(0.3, 2.5), mot_px=(1.0, 4.0), noise_max=2.0, avoid=8.0):
    """실 클립 frames (B,3,T,H,W) 0–255 에 알려진 교란을 주입: 곱셈 밝기(플리커 + 드리프트) × 주기 평행이동(grid_sample, 임의 방향) + 가우스 센서 잡음.
    플리커·이동 주파수는 라벨 심박의 기본·반·배 ±avoid bpm 을 피해 다시 뽑는다. 반환 (x_aug, [gain 차분, vx, vy] 각 (B,T−1), 주입 주파수 bpm (B,2))."""
    B, C, T, H, W = frames.shape; dev = frames.device
    t = torch.arange(T, device=dev, dtype=torch.float32) / float(fs)
    def pick(lo, hi):
        f = torch.empty(B, device=dev).uniform_(lo, hi)
        for _ in range(30):
            bad = torch.zeros(B, dtype=torch.bool, device=dev)
            for h in (1.0, 0.5, 2.0): bad |= (f * 60.0 - hr.to(dev).float() * h).abs() < avoid
            if not bool(bad.any()): break
            f = torch.where(bad, torch.empty(B, device=dev).uniform_(lo, hi), f)
        return f
    fi = pick(*flick); a = torch.empty(B, device=dev).uniform_(*amp); ph = torch.rand(B, device=dev) * 2 * math.pi
    d = torch.empty(B, device=dev).uniform_(0.0, drift_max)
    drift = torch.cumsum(torch.randn(B, T, device=dev), 1); drift = drift - drift.mean(1, keepdim=True)
    drift = drift / (drift.abs().max(1, keepdim=True)[0] + 1e-6) * d[:, None]
    gain = 1.0 + a[:, None] * torch.sin(2 * math.pi * fi[:, None] * t[None, :] + ph[:, None]) + drift
    fm = pick(*mot_hz); A = torch.empty(B, device=dev).uniform_(*mot_px); phm = torch.rand(B, device=dev) * 2 * math.pi; ang = torch.rand(B, device=dev) * 2 * math.pi
    disp = A[:, None] * torch.sin(2 * math.pi * fm[:, None] * t[None, :] + phm[:, None])
    dx = disp * torch.cos(ang)[:, None]; dy = disp * torch.sin(ang)[:, None]
    x = frames[:, :3].float()
    xf = x.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)
    theta = torch.zeros(B * T, 2, 3, device=dev); theta[:, 0, 0] = 1.0; theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = (-2.0 * dx / W).reshape(-1); theta[:, 1, 2] = (-2.0 * dy / H).reshape(-1)
    grid = F.affine_grid(theta, list(xf.shape), align_corners=False)
    xs = F.grid_sample(xf, grid, padding_mode="border", align_corners=False).reshape(B, T, 3, H, W).permute(0, 2, 1, 3, 4)
    sig = torch.empty(B, device=dev).uniform_(0.0, noise_max)
    xa = (xs * gain[:, None, :, None, None] + sig[:, None, None, None, None] * torch.randn_like(xs)).clamp(0.0, 255.0)
    return xa, [torch.diff(gain, dim=1), torch.diff(dx, dim=1), torch.diff(dy, dim=1)], torch.stack([fi * 60.0, fm * 60.0], 1)


# ---- [Virtual-Face patch 40, 55 문서 ④] 정본 readout 과 일치하는 청크 심박 라벨 ----
_DETREND_CACHE = {}


def _detrend_np(x, lam=100.0):
    """toolbox _detrend 과 동치; (n, λ) 별 행렬을 1회만 만들어 캐시 (11-01: 매 호출 160×160 역행렬 + BLAS 스레드 과점유로 7 프로세스 동시 학습이 15 s/it 로 느려진 사고)"""
    n = len(x); key = (n, float(lam))
    if key not in _DETREND_CACHE:
        I = np.eye(n); D = np.diff(I, n=2, axis=0); _DETREND_CACHE[key] = I - np.linalg.inv(I + lam * lam * D.T @ D)
    return _DETREND_CACHE[key] @ x


def label_hr_tb(labels, fs, lo=0.6, hi=3.3):
    """labels (B,T) tensor/array (표준화 맥파) → 툴박스 post_process 와 같은 detrend(λ=100) + Butterworth(1차, 0.6–3.3 Hz, filtfilt) 뒤 8× 영패딩 FFT 피크(bpm), torch (B,)."""
    import scipy.signal as ss
    arr = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    b, a = ss.butter(1, [lo / float(fs) * 2, hi / float(fs) * 2], btype="bandpass"); out = []
    for x in arr:
        x = np.asarray(x, np.float64); x = x - x.mean()
        if x.std() < 1e-6: out.append(float("nan")); continue
        y = ss.filtfilt(b, a, _detrend_np(x)); n = len(y); nfft = 8 * n
        P = np.abs(np.fft.rfft(y * np.hanning(n), nfft)) ** 2; f = np.fft.rfftfreq(nfft, 1.0 / float(fs)); m = (f >= lo) & (f <= hi)
        out.append(float(f[m][np.argmax(P[m])] * 60.0))
    o = np.asarray(out, np.float64); o = np.where(np.isfinite(o), o, 75.0)
    return torch.tensor(o, dtype=labels.dtype if torch.is_tensor(labels) else torch.float32, device=labels.device if torch.is_tensor(labels) else "cpu")


def ref_traces(frames, paths, bg_thr=0.875, ns_thr=0.5, min_bg=0.03, min_ns=0.005):
    """[Virtual-Face patch 52, 61 문서] 부위 분할 사이드카(3×64×64: 피부·배경·얼굴 내 비피부 확률)로 잡음 참조 trace 를 만든다.
    frames (B,3,T,H,W) 0~255 RGB, paths = 샘플별 사이드카 경로(None 허용) → ([배경 R,G,B 차분정규화, 얼굴 내 비피부 차분정규화] 각 (B,T−1), 배경 밝기 차분 (B,T−1), ok_bg (B,), ok_ns (B,)).
    배경에는 맥박이 없으므로 크롭 전체 밝기(g_diff)와 달리 심박 성분이 섞이지 않은 조명 참조가 된다."""
    import os as _o, numpy as _n
    B, _, T, H, W = frames.shape; dev = frames.device
    bgm = torch.zeros(B, 1, 1, H, W, device=dev); nsm = torch.zeros(B, 1, 1, H, W, device=dev)
    ok_bg = torch.zeros(B, dtype=torch.bool, device=dev); ok_ns = torch.zeros(B, dtype=torch.bool, device=dev)
    for b, pth in enumerate(paths):
        if not pth or not _o.path.exists(pth): continue
        P = torch.from_numpy(_n.load(pth)).float().to(dev)[None] / 255.0
        P = F.interpolate(P, size=(H, W), mode="bilinear", align_corners=False)[0]
        bm = (P[1] >= bg_thr).float(); nm = (P[2] >= ns_thr).float()
        if float(bm.mean()) >= min_bg: bgm[b, 0, 0] = bm; ok_bg[b] = True
        if float(nm.mean()) >= min_ns: nsm[b, 0, 0] = nm; ok_ns[b] = True
    x = frames[:, :3].float()
    bg = (x * bgm).sum((-1, -2)) / (bgm.sum((-1, -2)) + 1e-6)                 # (B,3,T)
    ns = ((x * nsm).sum((-1, -2)) / (nsm.sum((-1, -2)) + 1e-6)).mean(1)       # (B,T)
    dn = lambda v: (v[..., 1:] - v[..., :-1]) / (v[..., 1:] + v[..., :-1] + 1e-6)
    bgd = dn(bg) * ok_bg[:, None, None].float(); nsd = dn(ns) * ok_ns[:, None].float()
    trs = [bgd[:, 0], bgd[:, 1], bgd[:, 2], nsd]
    if _o.environ.get("NF_SS_REF_HC", "0") == "1":   # [Virtual-Face patch 53, 61 문서 §9] 머리카락·모자·옷 영역 밝기 차분 trace 를 잡음 참조에 추가 (사이드카 {name}_parsehc{i}.npy)
        hcm = torch.zeros(B, 1, 1, H, W, device=dev); ok_hc = torch.zeros(B, dtype=torch.bool, device=dev)
        for b, pth in enumerate(paths):
            hp_ = pth.replace("_parse", "_parsehc") if pth else None
            if not hp_ or not _o.path.exists(hp_): continue
            Q = F.interpolate(torch.from_numpy(_n.load(hp_)).float().to(dev)[None] / 255.0, size=(H, W), mode="bilinear", align_corners=False)[0]
            hm = (Q[0] >= 0.5).float()
            if float(hm.mean()) >= 0.03: hcm[b, 0, 0] = hm; ok_hc[b] = True
        hc = ((x * hcm).sum((-1, -2)) / (hcm.sum((-1, -2)) + 1e-6)).mean(1)       # (B,T)
        trs.append(dn(hc) * ok_hc[:, None].float()); ref_traces.last_hc_ok = int(ok_hc.sum())
    return trs, bgd.mean(1), ok_bg, ok_ns
