#!/usr/bin/env python3
"""압축 증강 — 지정 클립의 mpeg4 재인코딩 사본을 만든다(정답 불변, 디노이징 원칙).
MCD 등 실환경 소비자 카메라 영상은 손실 압축을 거치므로, 압축을 노이즈 taxonomy의
한 축으로 학습시킨다. 원본 FFV1는 보존하고 사본 디렉토리를 만든다."""
import argparse, glob, os, shutil, subprocess


def compress_copy(src_dir, dst_dir, q=6):
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.exists(os.path.join(dst_dir, "vid.avi")):
        subprocess.run(["ffmpeg", "-y", "-i", os.path.join(src_dir, "vid.avi"),
                        "-c:v", "mpeg4", "-q:v", str(q),
                        os.path.join(dst_dir, "vid.avi")],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.copy(os.path.join(src_dir, "ground_truth.txt"),
                os.path.join(dst_dir, "ground_truth.txt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="src_dir:dst_dir 목록")
    ap.add_argument("--q", type=int, default=6, help="mpeg4 품질 (2=고화질..31)")
    a = ap.parse_args()
    for pr in a.pairs:
        s, d = pr.split(":")
        compress_copy(s, d, a.q)
        print("compressed:", os.path.basename(d))
