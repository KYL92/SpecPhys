"""[64 문서 §8] 이중 렌더 잡음 정답이 클립 자신의 참조 신호보다 새 정보를 얼마나 주는가 (학습 없음, CPU).

학습 풀(cotrain_allf_hyb4_u8)에서 이중 렌더 정답이 있는 청크 288개(클립 48개 × 6)를 학습과 같은 로더·같은 함수로 읽어
  (a) 정확한 잡음 정답 n(t)            : noise_gt_hyb2 사이드카 (최종 모델에서 교란 슬롯 L_nuis 의 표적)
  (b) 라벨을 끄면 쓰이는 대체 표적       : 크롭 전체 밝기 차분 g_diff (L_nuis, g_std > 1 회색 단계일 때)
                                          배경 밝기 차분, 심박 대역 제거 (L_nu2, 사이드카 없는 청크)
  (c) 출력 배제 마스크 M                 : 배경 R,G,B·얼굴 내 비피부·LK vx,vy (L_excl, 라벨과 무관하게 모든 청크)
를 비교한다. 지표: 스펙트럼 코사인 유사도, n(t) 의 잡음 전력 중 M 이 이미 덮는 비율, 최대 잡음 정점 일치율.
대조: 같은 지표를 다른 클립의 참조와 짝지어 계산(셔플 20회) — 우연 수준.
실행: cd repos/rPPG-Toolbox && RPPG_CACHE_U8=1 RPPG_SKINW=0.2 python ../../scripts/noise_label_redundancy.py
출력: _workspace/data/results/noise_label_redundancy.{json,txt}
"""
import argparse, glob, json, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/SPECPHYS_ROOT"
sys.path.insert(0, os.getcwd())
os.environ.pop("RPPG_TAUG", None)          # 배속 증강 청크는 잡음 정답과 시간축이 다르다(트레이너도 제외)
assert os.environ.get("RPPG_CACHE_U8") == "1" and os.environ.get("RPPG_SKINW") == "0.2", "학습과 같은 입력 경로: RPPG_CACHE_U8=1 RPPG_SKINW=0.2"
from config import get_config
from dataset.data_loader.VIPLHRLoader import VIPLHRLoader
from neural_methods.loss import NFLoss as N

YAML = f"{ROOT}/_workspace/data/physformer/yaml_VIPL/finetune_NF_NSP2_L_s2100.yaml"
NGT = f"{ROOT}/_workspace/data/physformer/noise_gt_hyb2"
OUT = f"{ROOT}/_workspace/data/results/noise_label_redundancy"
FS, W5 = 30.0, 5.0

cfg = get_config(argparse.Namespace(config_file=YAML))
ds = VIPLHRLoader("train", cfg.TRAIN.DATA.DATA_PATH, cfg.TRAIN.DATA)
ngt = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(NGT + "/*.npy")}
sel = []
for i, p in enumerate(ds.inputs):
    m = re.match(r"(subject\d+)_input(\d+)\.npy$", os.path.basename(p))
    if m and m.group(1) in ngt: sel.append(i)
print(f"[nlr] pool chunks {len(ds.inputs)}, dual-render chunks {len(sel)}, sidecars {len(ngt)}", flush=True)
assert len(sel) == 288

def cos(a, b):
    return float(F.cosine_similarity(a, b, dim=-1, eps=1e-12))

rows, keep = [], []
for i in sel:
    data, label, name, cid = ds[i]
    name = str(name).split("|")[0]; cid = int(cid)
    x = torch.as_tensor(np.asarray(data), dtype=torch.float32)[None]          # (1,3,T,H,W) 0..255 (피부 가중 포함)
    x = torch.cat([x, x[:, :, -1:]], 2)                                       # 트레이너와 같이 마지막 프레임 덧붙임
    lab = torch.as_tensor(np.asarray(label), dtype=torch.float32)[None]
    if lab.dim() > 2: lab = lab[..., 0]
    L = lab.shape[-1]
    lab = (lab - lab.mean()) / lab.std()
    hr = N.label_hr(lab, FS).clamp(30.0, 199.0)
    g = x[:, :3].mean(dim=(1, 3, 4)); gd = torch.diff(g, dim=1)               # 모델의 g_diff
    gm = F.avg_pool1d(g.unsqueeze(1), 31, stride=1, padding=15, count_include_pad=False)[:, 0, :g.shape[1]]
    g_std = float((g - gm).std(dim=1))
    seg = ngt[name][cid * L:(cid + 1) * L]; assert len(seg) == L == gd.shape[1], (len(seg), L, gd.shape)
    n = torch.as_tensor(seg, dtype=torch.float32)[None]
    ppath = ds.inputs[i].replace("_input", "_parse")
    trs, bgl, okb, okn = N.ref_traces(x, [ppath])
    vx, vy = N.motion_traces(x)
    gdt = torch.where(okb[:, None], F.pad(bgl, (0, gd.shape[-1] - bgl.shape[-1])), gd)   # L_nu2 표적 trace (트레이너와 같음)
    z = lambda t: (t - t.mean(-1, keepdim=True)) / (t.std(-1, keepdim=True) + 1e-6)
    keep.append(dict(n=n, gd=gd, gdt=gdt, refs=[t.clone() for t in trs] + [vx, vy], hr=hr))
    rows.append(dict(name=name, cid=cid, hr=float(hr), g_std=g_std, ok_bg=bool(okb), ok_ns=bool(okn)))

def metrics(k_n, k_r):
    """k_n: 정답을 줄 청크, k_r: 참조를 줄 청크 (같으면 실제, 다르면 셔플 대조). 심박 대역은 정답 쪽 청크의 라벨 심박으로 지운다."""
    a, b = keep[k_n], keep[k_r]; hr = a["hr"]
    z = lambda t: (t - t.mean(-1, keepdim=True)) / (t.std(-1, keepdim=True) + 1e-6)
    Pn, bpm = N.power(z(a["n"]), FS, 30, 240); Pg, _ = N.power(z(b["gd"]), FS, 30, 240)
    s_ln = cos(Pn, Pg)                                                        # L_nuis 표적: 정답 vs 크롭 밝기
    Ptn = N.nuis_target_spec(a["n"], hr, FS, W5); Ptb = N.nuis_target_spec(b["gdt"], hr, FS, W5)
    s_nu2 = cos(Ptn, Ptb)                                                     # 심박 대역을 지운 표적: 정답 vs 배경 참조
    M = N.nuis_mask(b["refs"], hr, FS, W5)                                    # L_excl 마스크 (참조만)
    cover = float((Ptn * M).sum())                                            # 정답 잡음 전력 중 마스크가 덮는 몫 (M∈[0,1] 가중)
    cover50 = float((Ptn * (M >= 0.5).float()).sum())
    pk_n = int(Ptn.argmax()); Mn = M[0]
    near = float(Mn[max(0, pk_n - 3):pk_n + 4].max())                         # 정답 최대 정점 ±3 bpm 에서 마스크 최댓값
    return dict(sim_nuis=s_ln, sim_nu2=s_nu2, cover=cover, cover50=cover50, peak_hit=float(near >= 0.5))

real = [metrics(k, k) for k in range(len(keep))]
for r, m in zip(rows, real): r.update(m)
rng = np.random.default_rng(0); clip = np.array([r["name"] for r in rows])
shuf = []
for _ in range(20):
    perm = rng.permutation(len(keep))
    for k in range(len(keep)):
        j = int(perm[k])
        if clip[j] == clip[k]: j = int(np.flatnonzero(clip != clip[k])[rng.integers((clip != clip[k]).sum())])
        shuf.append(metrics(k, j))

def summ(ms, key):
    v = np.array([m[key] for m in ms]); return dict(mean=float(v.mean()), median=float(np.median(v)), q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)))

keys = ["sim_nuis", "sim_nu2", "cover", "cover50", "peak_hit"]
res = dict(n_chunks=len(rows), n_clips=len(set(clip)),
           gate_pass=int(sum(r["g_std"] > 1.0 for r in rows)), ok_bg=int(sum(r["ok_bg"] for r in rows)),
           real={k: summ(real, k) for k in keys}, shuffled={k: summ(shuf, k) for k in keys}, rows=rows)
json.dump(res, open(OUT + ".json", "w"), ensure_ascii=False, indent=1)
lab = {"sim_nuis": "L_nuis 표적 유사도: 정답 vs 크롭 밝기 (코사인)", "sim_nu2": "심박 대역 지운 표적 유사도: 정답 vs 배경 참조 (코사인)",
       "cover": "정답 잡음 전력 중 참조 마스크가 덮는 몫 (가중)", "cover50": "정답 잡음 전력 중 마스크 ≥0.5 구간 몫", "peak_hit": "정답 최대 잡음 정점(±3 bpm)이 마스크 ≥0.5 인 비율"}
with open(OUT + ".txt", "w") as f:
    f.write(f"이중 렌더 청크 {res['n_chunks']}개 (클립 {res['n_clips']}개); g_std>1 회색 단계 {res['gate_pass']}개, 배경 참조 사용 가능 {res['ok_bg']}개\n")
    f.write(f"{'지표':<52} {'실제 평균 (중앙값, 사분위)':<34} 셔플 대조 평균\n")
    for k in keys:
        a, b = res["real"][k], res["shuffled"][k]
        f.write(f"{lab[k]:<52} {a['mean']:.3f} ({a['median']:.3f}, {a['q25']:.3f}–{a['q75']:.3f})      {b['mean']:.3f}\n")
print(open(OUT + ".txt").read(), flush=True)
