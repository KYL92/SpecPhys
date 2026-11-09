#!/usr/bin/env python3
"""build_synth_dataset.py — 합성 얼굴 프레임에 심박 주입(+공간 이질성) 및 노이즈를 적용해
rPPG-Toolbox의 UBFC-rPPG 포맷(subjectN/vid.avi + ground_truth.txt)으로 패키징한다.

- 주입/파형/공간맵은 pixel-pulse-injection 스킬의 inject_pulse 모듈을 단일 소스로 재사용
- 노이즈 궤적은 rppg-noise-augmentation의 smooth_walk 재사용
- vid.avi는 FFV1 무손실 (ffmpeg 파이프) — 압축이 <1% 변조를 파괴하므로 필수
- ground_truth.txt: 1행 PPG, 2행 HR, 3행 시간 (UBFC 관례; 로더는 1행만 사용)

config JSON 예:
{"fps":30, "size":256, "out":"/path/DATA", "subjects":[
  {"id":1, "frames_dir":"...", "bpm":72, "alpha":0.01, "seed":1,
   "spatial":"patches", "coverage":0.6, "patch_scale":40,
   "noise":{"illum_drift":0.1,"flicker_amp":0.02,"flicker_hz":1.2,
            "motion_px":8,"rot_deg":1.5,"scale_jit":0.0,"sensor_std":2}}]}
"""
import argparse, glob, json, os, subprocess, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SKILLS = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(SKILLS, "generator"))
sys.path.insert(0, os.path.join(SKILLS, "generator"))
from inject_pulse import make_waveform, spatial_amplitude_map, feather, PBV_RGB  # noqa
from add_noise import smooth_walk  # noqa


def stable_skin_mask(frames, step=50):
    acc = np.zeros(frames[0].shape[:2], np.float32)
    sample = frames[::step] if len(frames) > step else frames
    for f in sample:
        y = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
        cr, cb = y[:, :, 1], y[:, :, 2]
        acc += ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.float32)
    m = ((acc / len(sample)) > 0.5).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return m.astype(np.float32) / 255.0


_PM = None


def parser_skin_mask(frames, step=100):
    """[66 문서 §7] 얼굴 부위 분할(SegFormer face-parsing, 피부+코)로 맥파 마스크를 얼굴 피부에 묶는다 — YCrCb 피부색 마스크가 따뜻한 색 배경으로 새는 문제(66 §6.3) 수정.
    머리 움직임을 덮도록 step 프레임마다 분할한 피부 영역의 합집합(0/1)을 돌려준다."""
    global _PM
    import torch, torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation
    if _PM is None:
        _PM = SegformerForSemanticSegmentation.from_pretrained("/SPECPHYS_ROOT/_workspace/models/face_parsing").eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    acc = np.zeros(frames[0].shape[:2], np.float32)
    for f in frames[::step]:
        x = torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255.0
        x = (F.interpolate(x, size=(384, 384), mode="bilinear", align_corners=False) - mean) / std
        with torch.no_grad():
            lab = _PM(pixel_values=x).logits.argmax(1)
        lab = F.interpolate(lab[:, None].float(), size=f.shape[:2], mode="nearest")[0, 0].numpy()
        acc = np.maximum(acc, np.isin(lab, (1, 2)).astype(np.float32))
    return acc


def write_ffv1(frames, path, fps):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "ffv1", "-level", "3", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg FFV1 encode failed: {path}")


def build_subject(sub, fps, size, out_root):
    paths = sorted(glob.glob(os.path.join(sub["frames_dir"], "*.png")))
    if not paths:
        raise FileNotFoundError(sub["frames_dir"])
    n = len(paths)
    seed = int(sub.get("seed", 0))
    rng = np.random.default_rng(seed)

    gt_wave = None
    if sub.get("wave_file"):
        # [체인 37] 사전 생성 파형: inject(열화판)를 주입하고 GT는 clean(원 파형)으로 기록 — 열화→복원 감독
        wf = np.load(sub["wave_file"])
        def _fit(x):
            x = np.asarray(x, np.float64)
            if len(x) < n: x = np.tile(x, int(np.ceil(n / len(x))))
            return x[:n]
        ppg = _fit(wf["inject"]); ppg = (ppg - ppg.mean()) / (np.abs(ppg).max() + 1e-9)
        gt_wave = _fit(wf["gt"]); gt_wave = (gt_wave - gt_wave.mean()) / (np.abs(gt_wave).max() + 1e-9)
    elif sub.get("wave_bank"):
        # [16-11 디지털 트윈] 실 PPG 파형 뱅크: 세그먼트를 목표 BPM으로 리샘플(형태·HRV·dicrotic notch 보존)
        import scipy.signal as _ss
        bank = np.load(sub["wave_bank"])
        if sub.get("wave_bank_nearest"):
            # [19단계 TW2] 목표 BPM에 f0가 가까운 상위 20개 중 idx 선택 — 리샘플 왜곡 최소화(실 HR-형태 결합 보존)
            order = np.argsort(np.abs(np.log(np.asarray(bank["f0"], float) / (sub["bpm"] / 60.0))))
            pool_n = max(1, min(20, len(order)))
            k = int(order[int(sub.get("wave_bank_idx", 0)) % pool_n])
        else:
            k = int(sub["wave_bank_idx"]) % len(bank["segments"])
        seg = bank["segments"][k]; f0 = float(bank["f0"][k]); r = (sub["bpm"] / 60.0) / f0
        if sub.get("wave_bank_nearest"):
            print(f"  bank_nearest id={sub['id']} bpm={sub['bpm']} seg_f0={f0*60:.0f}bpm ratio={r:.2f}")
        res = _ss.resample(seg, max(int(round(len(seg) / r)), 8))
        ppg = np.tile(res, int(np.ceil(n / len(res)) + 1))[:n].astype(np.float64)
        if sub.get("bpm_curve"):
            # [68 문서] 클립 안 심박 변화: bpm_curve = [[t_frac, bpm], ...] 사이를 선형 보간한 순간 심박으로 뱅크 구간의 위상을 진행(구간 자체 속도 f0 기준, 모양·HRV 보존)
            cv = np.asarray(sub["bpm_curve"], np.float64); inst = np.interp(np.arange(n) / max(n - 1, 1), cv[:, 0], cv[:, 1]) / 60.0   # (n,) Hz
            rate = inst / f0                                              # 뱅크 구간은 fps(30)로 표본화되어 있다: 출력 프레임마다 구간을 inst/f0 표본씩 진행
            u = np.concatenate([[0.0], np.cumsum(rate[:-1])])
            segt = np.tile(np.asarray(seg, np.float64), int(np.ceil((u[-1] + 2) / len(seg))) + 1)
            ppg = np.interp(u, np.arange(len(segt)), segt)
        ppg = (ppg - ppg.mean()) / (np.abs(ppg).max() + 1e-9)
    else:
        ppg, _ = make_waveform(n, fps, sub["bpm"], sub.get("hrv_std", 0.0), seed=seed,
                               wave_a2=sub.get("wave_a2"), wave_phi2=sub.get("wave_phi2", 0.0),
                               wave_a3=sub.get("wave_a3", 0.0), wave_phi3=sub.get("wave_phi3", 0.0))
    # [체인 40] wave_rms: 주입 파형의 RMS(std)를 명시 고정 — 정규화 규약 차이(std=1 vs max-abs=1)로 인한
    # 주입 진폭 혼입을 통제하는 ablation용. GT 기록은 파형 그대로(스케일 불변 프로토콜).
    if sub.get("wave_rms"):
        ppg = ppg / (ppg.std() + 1e-9) * float(sub["wave_rms"])
        if gt_wave is not None:
            gt_wave = gt_wave / (gt_wave.std() + 1e-9) * float(sub["wave_rms"])
    # 달성 BPM 실측 (평가는 이 값 기준; wave_file이면 GT=clean 기준)
    _ref = gt_wave if gt_wave is not None else ppg
    fr = np.fft.rfftfreq(n * 8, 1 / fps)
    pw = np.abs(np.fft.rfft(_ref - _ref.mean(), n=n * 8)) ** 2
    band = (fr >= 0.7) & (fr <= 4.0)
    achieved = float(fr[band][pw[band].argmax()] * 60)

    frames = [cv2.resize(cv2.imread(p), (size, size), interpolation=cv2.INTER_AREA)
              for p in paths]
    # [16단계 재설계] 피부 색 통계 이식: tint = 대상 도메인 피부 평균 RGB / 소스 얼굴 중앙 평균 RGB (채널 게인, 주입 전 적용)
    if sub.get("tint_rgb"):
        c = size // 2; q = size // 6
        src = np.mean([f[c-q:c+q, c-q:c+q].reshape(-1, 3).mean(0) for f in frames[::50]], axis=0)  # BGR
        tgt = np.array(sub["tint_rgb"][::-1], dtype=np.float64)
        gain = np.clip(tgt / np.maximum(src, 1), 0.4, 1.6).astype(np.float32)
        frames = [np.clip(f.astype(np.float32) * gain, 0, 255).astype(np.uint8) for f in frames]
    if sub.get("skin_mask_parser"):   # [66 문서 §7] YCrCb 마스크 ∧ 얼굴 부위 분할 피부 — 배경 누출 방지
        skin = feather(((stable_skin_mask(frames) > 0.5) & (parser_skin_mask(frames) > 0.5)).astype(np.float32), 15)
    else:
        skin = feather((stable_skin_mask(frames) > 0.5).astype(np.float32), 15)
    if sub.get("spatial") == "anatomy" and sub.get("amap_file"):
        # [16-8] 해부학 기반 A(x,y): 얼굴 파싱 + 사람별 랜덤 유효 맥동 공간 (scripts/anatomical_amplitude_map.py)
        amap = cv2.resize(np.load(sub["amap_file"]).astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    else:
      amap = spatial_amplitude_map((size, size), sub.get("spatial", "uniform"),
                                   sub.get("coverage", 1.0), sub.get("patch_scale", 40),
                                   sub.get("amp_mu", 1.0), sub.get("amp_sigma", 0.35), seed)
    mask = (skin * amap)[..., None]
    pbv = (np.array(sub["pbv_rgb"], dtype=np.float64) if sub.get("pbv_rgb") else PBV_RGB)[::-1].reshape(1, 1, 3)  # BGR; [SCALE300] 피부톤 의존 Pbv 재정의 옵션
    alpha = sub.get("alpha", 0.01)

    nz = sub.get("noise", {}) or {}
    t = np.arange(n) / fps
    drift = (smooth_walk(n, nz.get("illum_drift", 0) / max(np.sqrt(n), 1), int(fps), rng)
             if nz.get("illum_drift", 0) > 0 else np.zeros(n))
    drift = drift - drift.mean()
    flick = (nz.get("flicker_amp", 0) * np.sin(2 * np.pi * nz.get("flicker_hz", 1.2) * t)
             if nz.get("flicker_amp", 0) > 0 else np.zeros(n))
    gain_t = np.clip(1 + drift + flick, 0.3, 1.7)
    mp, rd, sj = nz.get("motion_px", 0), nz.get("rot_deg", 0), nz.get("scale_jit", 0)
    tx = smooth_walk(n, mp / max(np.sqrt(n), 1), int(fps // 2), rng) if mp > 0 else np.zeros(n)
    ty = smooth_walk(n, mp / max(np.sqrt(n), 1), int(fps // 2), rng) if mp > 0 else np.zeros(n)
    # [SCALE300-R] 실 영상 얼굴 상자 궤적(얼굴폭 정규화 dx,dy) → 병진으로 이식 (랜덤워크 대체)
    if nz.get("motion_traj"):
        tr = np.load(nz["motion_traj"])  # (T,2) in face-width units
        T = len(tr); idx = np.minimum(np.arange(n), T - 1); face_px = size * float(nz.get("face_frac", 0.45))
        tx = (tr[idx, 0] - tr[:, 0].mean()) * face_px; ty = (tr[idx, 1] - tr[:, 1].mean()) * face_px; mp = max(mp, 1e-6)
    # [외삽용 신규 유형] 주기적 끄덕임(정현파 병진 — 랜덤워크가 아닌 결정적 주기 운동)
    pm_amp, pm_hz = nz.get("periodic_motion_amp", 0), nz.get("periodic_motion_hz", 0.4)
    if pm_amp > 0:
        ty = ty + pm_amp * np.sin(2 * np.pi * pm_hz * t)
        mp = max(mp, pm_amp)  # 아핀 적용 트리거
    # [외삽용 신규 유형] 색온도 드리프트(채널별 상이한 저주파 게인 — Pbv 색공간을 직접 교란)
    cd = nz.get("color_drift", 0)
    if cd > 0:
        ch_gain = np.stack([1 + smooth_walk(n, cd / max(np.sqrt(n), 1), int(fps), rng)
                            for _ in range(3)], axis=1)  # (n,3) BGR
    else:
        ch_gain = None
    rot = smooth_walk(n, rd / max(np.sqrt(n), 1), int(fps // 2), rng) if rd > 0 else np.zeros(n)
    sc = 1 + (smooth_walk(n, sj / max(np.sqrt(n), 1), int(fps // 2), rng) if sj > 0 else np.zeros(n))
    sensor = nz.get("sensor_std", 0)
    ctr = (size / 2, size / 2)
    # [52 문서 B3] 공간 비균일 교란 — 전역 밝기 trace 로 대표되지 않는 조명 교란 3종(모두 선택적, 기본 없음)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    sh = nz.get("shadow_half")          # {amp, hz, side('left'|'right')}: 얼굴 반쪽만 정현파 밝기 변조(부드러운 경계)
    if sh:
        edge = 1.0 / (1.0 + np.exp(-(xx - size / 2) / (0.04 * size)))      # 0(left) → 1(right)
        sh_map = edge if sh.get("side", "right") == "right" else 1.0 - edge
    lf = nz.get("local_flicker")        # {amp, hz, frac}: 피부 일부(가우시안 블롭, 면적 비율 frac)만 플리커
    if lf:
        ys, xs = np.where(mask[..., 0] > 0.2)
        k = rng.integers(len(ys)); cy, cx = float(ys[k]), float(xs[k])
        r = np.sqrt(float(lf.get("frac", 0.35)) * len(ys) / np.pi)
        lf_map = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (r / 1.5) ** 2))
    sp = nz.get("spec_patch")           # {amp(그레이), radius, speed_hz, span}: 얼굴 위를 정현파로 이동하는 밝은 반사광 패치(가산)
    if sp:
        ys, xs = np.where(mask[..., 0] > 0.2); cy0 = float(ys.mean()); cx0 = float(xs.mean()); span = float(sp.get("span", 0.25)) * size

    out = []
    # [체인 39] noise_gt: 주입 없는(alpha=0) 병렬 경로로 '관측 가능한 노이즈 trace'를 추출 — 교란 궤적·센서 draw 공유
    want_ngt = bool(sub.get("noise_gt")) or bool(sub.get("noise_video"))
    # [65 문서, 체인 124] noise_video: 둘째 벌(주입 진폭 0, 같은 움직임·조명·센서 난수)을 vid_noise.avi(FFV1 무손실)로도 저장 — 픽셀·색 단위 잡음 정답
    want_nv = bool(sub.get("noise_video")); out_n = [] if want_nv else None
    trace_n = np.zeros(n) if want_ngt else None
    trace_t = np.zeros(n) if want_ngt else None  # [체인 41] 전체(주입+노이즈) trace — 노이즈 비율 GT용
    skin_sel = mask[..., 0] > 0.2
    for i, img in enumerate(frames):
        img = img.astype(np.float32)
        img_n = img.copy() if want_ngt else None
        g = 1.0 + alpha * ppg[i] * pbv
        m_i = mask
        if sub.get("spec_exclude"):
            # [16-11 디지털 트윈] 정반사 분리: 고휘도·저채도 하이라이트는 맥동에서 제외 (표면 반사는 혈류를 통과하지 않음)
            mx = img.max(2); mn = img.min(2); sat = (mx - mn) / (mx + 1e-6)
            spec = ((mx > 235) & (sat < 0.12)).astype(np.float32)
            if spec.any():
                spec = cv2.GaussianBlur(spec, (9, 9), 0)
                m_i = mask * (1.0 - spec[..., None])
        img = img * (1.0 + m_i * (g - 1.0))                      # ① 주입
        if mp > 0 or rd > 0 or sj > 0:                            # ② 움직임
            M = cv2.getRotationMatrix2D(ctr, float(rot[i]), float(sc[i]))
            M[:, 2] += (tx[i], ty[i])
            img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
            if want_ngt:
                img_n = cv2.warpAffine(img_n, M, (size, size), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
        img = img * gain_t[i]                                     # ③ 조명
        if want_ngt:
            img_n = img_n * gain_t[i]
        if sh or lf or sp:                                        # ③c [52 문서 B3] 공간 비균일 조명 교란
            gmap = np.ones((size, size), np.float32)
            if sh: gmap = gmap * (1.0 + sh["amp"] * np.sin(2 * np.pi * sh["hz"] * t[i]) * sh_map)
            if lf: gmap = gmap * (1.0 + lf["amp"] * np.sin(2 * np.pi * lf["hz"] * t[i]) * lf_map)
            img = img * gmap[..., None]
            if want_ngt: img_n = img_n * gmap[..., None]
            if sp:
                cx = cx0 + span * np.sin(2 * np.pi * sp["speed_hz"] * t[i]); cy = cy0 + 0.3 * span * np.sin(2 * np.pi * sp["speed_hz"] * t[i] + 1.0)
                blob = sp["amp"] * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * float(sp.get("radius", 18)) ** 2))
                img = img + blob[..., None]
                if want_ngt: img_n = img_n + blob[..., None]
        if ch_gain is not None:
            img = img * ch_gain[i].reshape(1, 1, 3)               # ③b 색온도 드리프트
            if want_ngt:
                img_n = img_n * ch_gain[i].reshape(1, 1, 3)
        if sensor > 0:
            ns = rng.normal(0, sensor, img.shape)                 # ④ 센서 (양 경로 공유 draw)
            img = img + ns
            if want_ngt:
                img_n = img_n + ns
        if want_ngt:
            trace_n[i] = float(np.clip(img_n, 0, 255)[..., 1][skin_sel].mean())
            trace_t[i] = float(np.clip(img, 0, 255)[..., 1][skin_sel].mean())
        out.append(np.clip(img, 0, 255).astype(np.uint8))
        if want_nv:
            out_n.append(np.clip(img_n, 0, 255).astype(np.uint8))

    d = os.path.join(out_root, f"subject{sub['id']}")
    os.makedirs(d, exist_ok=True)
    write_ffv1(out, os.path.join(d, "vid.avi"), fps)
    if want_nv:
        write_ffv1(out_n, os.path.join(d, "vid_noise.avi"), fps)
    with open(os.path.join(d, "ground_truth.txt"), "w") as f:
        f.write(" ".join(f"{v:.6f}" for v in (gt_wave if gt_wave is not None else ppg)) + "\n")
        f.write(" ".join(f"{achieved:.2f}" for _ in range(n)) + "\n")
        f.write(" ".join(f"{i/fps:.4f}" for i in range(n)) + "\n")
    if want_ngt:
        # DiffNormalized 규약(차분 정규화), [체인 41] 세기 보존판(npz: 표준화 없음, 전체 trace 동봉) + 레거시 npy
        def _dn(x):
            v = (x[1:] - x[:-1]) / (x[1:] + x[:-1] + 1e-7)
            return np.concatenate([v, [0.0]]).astype(np.float32)
        nd, td = _dn(trace_n), _dn(trace_t)
        np.savez(os.path.join(d, "noise_gt.npz"), noise_dn=nd, total_dn=td)
        np.save(os.path.join(d, "noise_gt.npy"), (nd / (nd.std() + 1e-7)).astype(np.float32))
    cv2.imwrite(os.path.join(d, "amplitude_map.png"),
                np.clip(mask[..., 0] * 255, 0, 255).astype(np.uint8))
    meta = dict(sub); meta["achieved_bpm"] = round(achieved, 2); meta["n_frames"] = n
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[build] subject{sub['id']}: bpm={sub['bpm']}→{achieved:.2f} "
          f"spatial={sub.get('spatial','uniform')}/{sub.get('coverage',1.0)} noise={nz} ")
    return achieved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    os.makedirs(cfg["out"], exist_ok=True)
    for sub in cfg["subjects"]:
        build_subject(sub, cfg.get("fps", 30), cfg.get("size", 256), cfg["out"])
    print("[build] done:", cfg["out"])
