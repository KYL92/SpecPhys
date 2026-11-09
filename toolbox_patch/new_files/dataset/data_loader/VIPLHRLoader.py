"""[Virtual-Face patch 28] VIPL-HR (Niu et al., ACCV 2018 / TIP 2020) 데이터로더.

원본 구조 (DATA_PATH 아래):
    pS/vV[/-2]/source{1,2,3}/{video.avi, wave.csv, gt_HR.csv, [time.txt]}
    fold/fold{1..5}.mat   (피험자 배타 5-fold — FOLD_PATH 로 지정)

처리 규칙 (문헌 프로토콜: RhythmNet/PhysFormer 과 동일 취지):
  * VIS 3소스(source1 웹캠·source2 휴대폰·source3 RealSense RGB)만 사용, NIR(source4) 제외 → 2,378 영상.
  * 가변 프레임률: time.txt(ms) 가 있으면 타임스탬프 기준 30 fps 로 최근접 프레임 재표본, 없으면(source2) 30 fps 고정.
  * BVP wave(60 Hz) 는 영상 길이에 선형 재표본(영상·파형 동시 시작·종료 가정), gt_HR.csv 평균은 메타 사이드카에 보존
    ({index}_meta.json → 문헌 호환 GT; 정본 HR 은 라벨 파형에서 toolbox 프로토콜로 재계산).
  * 얼굴 박스: 첫 프레임 Haar(HC) 정적 박스 + large box(yaml CROP_FACE 설정과 동일 계수), 프레임을 읽으면서 즉시 crop/resize
    → 1080p 전체를 메모리에 올리지 않음 (BaseLoader.preprocess 는 skip_crop=True 로 호출).
  * 청크 파일명 index = p{S}_v{V}[-2]_s{src}  (예: p1_v1-2_s3)
FOLD: config.FOLD.FOLD_NAME = 'train3' | 'test3' (fold3 피험자 제외/포함), FOLD_PATH = fold/*.mat 디렉토리.
"""
import glob
import json
import os
import re

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader


class VIPLHRLoader(BaseLoader):
    TARGET_FPS = 30.0

    def __init__(self, name, data_path, config_data, device=None):
        super().__init__(name, data_path, config_data, device)

    # ------------------------------------------------------------------ 목록
    def get_raw_data(self, data_path):
        vids = sorted(glob.glob(os.path.join(data_path, "p*", "v*", "source[123]", "video.avi")))
        if not vids:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = []
        for v in vids:
            d = os.path.dirname(v)
            src = os.path.basename(d)[-1]
            scen = os.path.basename(os.path.dirname(d))
            subj = os.path.basename(os.path.dirname(os.path.dirname(d)))
            m = re.fullmatch(r"p(\d+)", subj)
            if not m or not re.fullmatch(r"v\d(-2)?", scen):
                continue
            dirs.append({"index": f"{subj}_{scen}_s{src}", "path": d, "subject": int(m.group(1))})
        return dirs

    @staticmethod
    def fold_subjects(fold_path, k):
        import scipy.io as sio
        m = sio.loadmat(os.path.join(fold_path, f"fold{k}.mat"))
        return set(int(x) for x in np.asarray(m[f"fold{k}"]).ravel().tolist())

    def split_raw_data(self, data_dirs, begin, end):
        fold = getattr(self.config_data, "FOLD", None)
        if fold is not None and fold.FOLD_NAME and fold.FOLD_PATH:
            mm = re.fullmatch(r"(train|test)([1-5])", fold.FOLD_NAME)
            if not mm:
                raise ValueError("VIPL-HR FOLD_NAME must be train{k}|test{k}, got " + fold.FOLD_NAME)
            subs = self.fold_subjects(fold.FOLD_PATH, int(mm.group(2)))
            keep = (lambda s: s in subs) if mm.group(1) == "test" else (lambda s: s not in subs)
            data_dirs = [d for d in data_dirs if keep(d["subject"])]
        if begin == 0 and end == 1:
            return data_dirs
        n = len(data_dirs)
        return [data_dirs[i] for i in range(int(begin * n), int(end * n))]

    # ------------------------------------------------------------------ 전처리
    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        d = data_dirs[i]
        idx = d["index"]
        try:
            frames, meta = self.read_video_resampled(d["path"], config_preprocess)
            wave = self.read_csv_col(os.path.join(d["path"], "wave.csv"))
            if frames.shape[0] < config_preprocess.CHUNK_LENGTH or len(wave) < 10:
                print(f"[VIPL] skip {idx}: frames={frames.shape[0]} wave={len(wave)}")
                file_list_dict[i] = []
                return
            bvps = BaseLoader.resample_ppg(wave, frames.shape[0])
            try:
                hr = self.read_csv_col(os.path.join(d["path"], "gt_HR.csv"))
                meta["gt_hr_mean"] = float(np.mean(hr)); meta["gt_hr_std"] = float(np.std(hr)); meta["gt_hr_n"] = int(len(hr))
            except Exception:
                meta["gt_hr_mean"] = None
            meta.update(index=idx, subject=d["subject"], scenario=idx.split("_")[1], source=int(idx[-1]),
                        wave_n=int(len(wave)))
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess, skip_crop=True)
            input_name_list, _ = self.save_multi_process(frames_clips, bvps_clips, idx)
            with open(os.path.join(self.cached_path, f"{idx}_meta.json"), "w") as f:
                json.dump(meta, f)
            file_list_dict[i] = input_name_list
        except Exception as e:  # 한 영상 실패가 전체 전처리를 죽이지 않도록
            print(f"[VIPL] FAIL {idx}: {type(e).__name__}: {e}")
            file_list_dict[i] = []

    @staticmethod
    def read_csv_col(path):
        with open(path) as f:
            rows = [r.strip() for r in f.read().splitlines()]
        vals = []
        for r in rows[1:] if rows and not re.fullmatch(r"-?\d+(\.\d+)?", rows[0]) else rows:
            if r:
                vals.append(float(r.split(",")[0]))
        return np.asarray(vals, dtype=np.float64)

    def read_video_resampled(self, vdir, cp):
        """MJPG 영상 → (타임스탬프 기준 30fps 재표본, 정적 Haar 박스 crop, resize) float32 (T,H,W,3) RGB."""
        cap = cv2.VideoCapture(os.path.join(vdir, "video.avi"))
        n_hdr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_hdr = float(cap.get(cv2.CAP_PROP_FPS))
        tpath = os.path.join(vdir, "time.txt")
        if os.path.exists(tpath):
            ts = np.loadtxt(tpath, dtype=np.float64).ravel()
            ts_src = "time.txt"
        else:
            ts = np.arange(n_hdr) * (1000.0 / self.TARGET_FPS)
            ts_src = "fixed30"
        n_ts = len(ts)
        n_use = min(n_ts, n_hdr) if n_hdr > 0 else n_ts
        ts = ts[:n_use]
        if n_use < 2:
            raise ValueError("too few frames")
        tgt = np.arange(ts[0], ts[-1] + 1e-6, 1000.0 / self.TARGET_FPS)
        src_idx = np.searchsorted(ts, tgt)  # 최근접 프레임
        src_idx = np.clip(src_idx, 0, n_use - 1)
        left = np.clip(src_idx - 1, 0, n_use - 1)
        pick_left = np.abs(ts[left] - tgt) < np.abs(ts[src_idx] - tgt)
        src_idx = np.where(pick_left, left, src_idx)
        W, H = cp.RESIZE.W, cp.RESIZE.H
        out = np.zeros((len(tgt), H, W, 3), dtype=np.float32)
        box = None
        j = 0  # next target slot
        fi = 0
        while j < len(tgt) and fi < n_use:
            ok, frame = cap.read()
            if not ok:
                break
            if box is None:
                rgb0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                box = self.face_detection(rgb0, cp.CROP_FACE.BACKEND, cp.CROP_FACE.USE_LARGE_FACE_BOX,
                                          cp.CROP_FACE.LARGE_BOX_COEF) if cp.CROP_FACE.DO_CROP_FACE \
                    else [0, 0, frame.shape[1], frame.shape[0]]
                box = [int(round(float(b))) for b in box]
                y0, y1 = max(box[1], 0), min(box[1] + box[3], frame.shape[0])
                x0, x1 = max(box[0], 0), min(box[0] + box[2], frame.shape[1])
            if j < len(tgt) and src_idx[j] == fi:
                crop = cv2.resize(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), (W, H),
                                  interpolation=cv2.INTER_AREA).astype(np.float32)
                while j < len(tgt) and src_idx[j] == fi:
                    out[j] = crop
                    j += 1
            fi += 1
        cap.release()
        if j < len(tgt):  # 디코드된 프레임이 타임스탬프보다 적음 → 채워진 만큼만
            out = out[:j]
        meta = dict(n_hdr=n_hdr, fps_hdr=fps_hdr, n_ts=n_ts, n_decoded=fi, n_out=int(out.shape[0]),
                    ts_src=ts_src, dur_ms=float(ts[-1] - ts[0]), box=[y0, y1, x0, x1])
        return out, meta
