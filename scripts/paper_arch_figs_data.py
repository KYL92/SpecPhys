#!/usr/bin/env python3
"""[63 문서] 구조 그림(paper_arch_figs.py)에 넣을 실제 예시 자료: 생성 얼굴 클립을 최종 체크포인트에 통과시켜 잡음 참조, 투영 전후 특징 스펙트럼, 출력 스펙트럼을 뽑는다(CPU, 학습 없음).
  python scripts/paper_arch_figs_data.py [SET/subjectN] → results/figs_arch/{arch_data.json, *.png}
얼굴 그림은 생성 얼굴만 쓴다(실제 피험자 영상은 신호만, 얼굴 없음)."""
import glob, json, os, pickle, sys
os.environ.update(RPPG_NF="1", NF_ARCH="nsp", NF_NSP_LAM="0.01", CUDA_VISIBLE_DEVICES="")
import cv2, numpy as np, torch, torch.nn.functional as F
ROOT = "/SPECPHYS_ROOT"; PFA = f"{ROOT}/_workspace/data/physformer"; TB = f"{ROOT}/repos/rPPG-Toolbox"; OUT = f"{ROOT}/_workspace/data/results/figs_arch"; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, TB); os.chdir(TB)
import neural_methods.model.FactorizePhys.FactorizePhys as FP
from neural_methods.model.FactorizePhys.FactorizePhysNF import FactorizePhysNF, noise_refs, nsp_component
from neural_methods.loss import NFLoss as NFL
CLIP = sys.argv[1] if len(sys.argv) > 1 else "STRESS_OFFBAND6/subject2"; HR, NUIS = 64.0, 94.0
CK = f"{PFA}/runs/finetune_NF_NSP2_L_s1600/PreTrainedModels/SYNTH_NF_NSP2_L_s1600_factorizephys_Epoch9.pth"; PUB = f"{TB}/final_model_release/UBFC-rPPG_FactorizePhys_FSAM_Res.pth"
cap = cv2.VideoCapture(f"{PFA}/{CLIP}/vid.avi"); fr = []
while True:
    ok, f = cap.read()
    if not ok: break
    fr.append(f[..., ::-1])
fr = np.stack(fr); print("clip", CLIP, fr.shape)
fc = cv2.CascadeClassifier(f"{TB}/dataset/haarcascade_frontalface_default.xml"); bx = fc.detectMultiScale(fr[0][..., ::-1].astype(np.uint8), minSize=(30, 30), minNeighbors=3); x0, y0, w0, h0 = max(bx, key=lambda b: b[2] * b[3])
c = 1.5; x1, y1 = max(0, int(x0 - (c - 1) / 2 * w0)), max(0, int(y0 - (c - 1) / 2 * h0)); w1, h1 = int(c * w0), int(c * h0); crop = fr[:, y1:y1 + h1, x1:x1 + w1]; print("crop", crop.shape)
L = 161; T0 = 60; seg = crop[T0:T0 + L]
small = np.stack([cv2.resize(f, (72, 72), interpolation=cv2.INTER_AREA) for f in seg]).astype(np.float32)
# 얼굴 부위 분할(청크당 8장 평균, 사이드카와 같은 절차)
from transformers import SegformerForSemanticSegmentation
PM = SegformerForSemanticSegmentation.from_pretrained(f"{ROOT}/_workspace/models/face_parsing").eval()
idx = np.linspace(0, L - 1, 8).round().astype(int); xp = torch.from_numpy(np.stack([cv2.resize(seg[i], (128, 128), interpolation=cv2.INTER_AREA) for i in idx])).permute(0, 3, 1, 2).float() / 255
xp = F.interpolate(xp, size=(384, 384), mode="bilinear", align_corners=False); xp = (xp - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
with torch.no_grad(): lab = PM(pixel_values=xp).logits.argmax(1)
lab = F.interpolate(lab[:, None].float(), size=(64, 64), mode="nearest")[:, 0].long(); oh = lambda ids: sum((lab == i).float() for i in ids).mean(0)
P = torch.stack([oh((1, 2)), oh((0,)), oh((3, 4, 5, 6, 7, 10, 11, 12))]); np.save(f"{OUT}/_parse.npy", (P.numpy() * 255).round().astype(np.uint8)); print("parse mean skin/bg/nonskin", P.mean((1, 2)).tolist())
pskin = cv2.resize(P[0].numpy(), (72, 72), interpolation=cv2.INTER_LINEAR); w = 0.2 + 0.8 * pskin
xw = small * w[None, :, :, None]
X = torch.from_numpy(xw).permute(3, 0, 1, 2)[None].float()
# 최종 모델
m = FactorizePhysNF(frames=160, md_config={**FP.model_config, "MD_FSAM": True, "MD_INFERENCE": True, "MD_RESIDUAL": True}, in_channels=3, dropout=0.1, device=torch.device("cpu"))
sd = torch.load(CK, map_location="cpu"); sd = {k.replace("module.", ""): v for k, v in sd.items()}; print("load", m.load_state_dict(sd, strict=False)); m.eval()
hold = {}; m.rppg_head.conv_block.register_forward_hook(lambda mm, i, o: hold.__setitem__("v", o.detach()))
with torch.no_grad(): out = m(X)[0][0].numpy()
v = hold["v"]; refs = noise_refs(X); comp = nsp_component(v, refs); vp = v - comp; print("v", v.shape, "refs", refs.shape)
def spec(a, fs=30.0, lo=30, hi=200, pad=8):
    a = a - a.mean(-1, keepdims=True); N = a.shape[-1] * pad; A = np.abs(np.fft.rfft(a, n=N, axis=-1)) ** 2; f = np.fft.rfftfreq(N, 1 / fs) * 60; s = (f >= lo) & (f <= hi); return f[s], A[..., s]
Xb = v[0].permute(0, 2, 3, 1).reshape(-1, v.shape[2]).numpy(); Xa = vp[0].permute(0, 2, 3, 1).reshape(-1, v.shape[2]).numpy(); Xc = comp[0].permute(0, 2, 3, 1).reshape(-1, v.shape[2]).numpy()
f, Pb = spec(Xb); _, Pa = spec(Xa); _, Pc = spec(Xc); nz = Xb.std(-1) > 1e-6; print("features", Xb.shape, "active", int(nz.sum()))
tot = Pb[nz].sum(-1, keepdims=True); Sb = (Pb[nz] / tot).mean(0); Sa = (Pa[nz] / tot).mean(0); Sc = (Pc[nz] / tot).mean(0)
pk = lambda S: float(f[np.argmax(S)]); band = lambda S, c, w=6: float(S[(f > c - w) & (f < c + w)].sum() / S.sum())
print("feature spectrum peak before/after", pk(Sb), pk(Sa), "share nuis before/after", band(Sb, NUIS), band(Sa, NUIS), "share HR", band(Sb, HR), band(Sa, HR), "energy kept", float(Sa.sum() / Sb.sum()))
fo, So = spec(out); print("output peak", fo[np.argmax(So)])
# 공개 FactorizePhys(같은 클립, 피부 가중 없음)
pub = FP.FactorizePhys(frames=160, md_config={**FP.model_config, "MD_FSAM": True, "MD_INFERENCE": True, "MD_RESIDUAL": True}, in_channels=3, dropout=0.1, device=torch.device("cpu")); sd = torch.load(PUB, map_location="cpu"); sd = {k.replace("module.", ""): v_ for k, v_ in sd.items()}; print("pub", pub.load_state_dict(sd, strict=False)); pub.eval()
with torch.no_grad(): op = pub(torch.from_numpy(small).permute(3, 0, 1, 2)[None].float())[0][0].numpy()
_, Sp = spec(op); print("public output peak", fo[np.argmax(Sp)])
# 잡음 배제 마스크 M(f): 학습 때와 같은 참조(배경 R,G,B, 비피부, LK)
trs, bgl, okb, okn = NFL.ref_traces(X, [f"{OUT}/_parse.npy"]); vx, vy = NFL.motion_traces(X); hrt = torch.tensor([HR]); print("ref ok", bool(okb[0]), bool(okn[0]))
Mk = NFL.nuis_mask([t.float() for t in trs] + [vx, vy], hrt, 30.0, 5.0); Po, bpm = NFL.power(torch.from_numpy(out)[None], 30.0, 30, 200); Po = NFL._norm(Po)
J = dict(clip=CLIP, hr=HR, nuis=NUIS, t=(np.arange(160) / 30.0).tolist(), refs=dict(R=refs[0, 0].tolist(), G=refs[0, 1].tolist(), B=refs[0, 2].tolist(), vx=refs[0, 6].tolist(), vy=refs[0, 7].tolist()), out=out.tolist(), pub=op.tolist(),
         f=f.tolist(), feat_before=Sb.tolist(), feat_after=Sa.tolist(), feat_removed=Sc.tolist(), so=So.tolist(), sp=Sp.tolist(), bpm=bpm.tolist(), mask=Mk[0].tolist(), p_out=Po[0].tolist(), pskin_mean=float(pskin.mean()))
sel = np.argsort(-Xb[nz].std(-1))[:1]; J["x_before"] = Xb[nz][sel[0]].tolist(); J["x_after"] = Xa[nz][sel[0]].tolist(); J["x_removed"] = Xc[nz][sel[0]].tolist()
# RTNI: 공개 vs 최종 출력 스펙트럼(얼굴 없음, 신호만)
def rtni(pk_path, vid="subject42"):
    d = pickle.load(open(pk_path, "rb")); ch = d["predictions"][vid]; y = np.concatenate([ch[k].numpy().ravel() for k in sorted(ch)]); l = np.concatenate([d["labels"][vid][k].numpy().ravel() for k in sorted(ch)]); return y, l
pa = glob.glob(f"{PFA}/runs/ev_FactorizePhys_RTNI/*/saved_test_outputs/*.pickle")[0]; pb = glob.glob(f"{PFA}/runs/ev_NF_NSP2_L_s1600_RTNI/*/saved_test_outputs/*.pickle")[0]
J["rtni"] = {}
for vid in ("subject42", "subject43", "subject44", "subject45", "subject46", "subject47", "subject48", "subject49"):
    ya, la = rtni(pa, vid); yb, lb = rtni(pb, vid); fa, Sa_ = spec(ya, pad=2); fb, Sb_ = spec(yb, pad=2); _, Sl = spec(lb, pad=2); J["rtni"][vid] = dict(f=fa.tolist(), pub=Sa_.tolist(), ours=Sb_.tolist(), lab=Sl.tolist()); print(vid, "pub peak %.1f ours peak %.1f label peak %.1f" % (fa[np.argmax(Sa_)], fb[np.argmax(Sb_)], fb[np.argmax(Sl)]))
# 히트맵: 특징(세로) × 시간(가로). 투영 전 표준편차로 나눠 투영 전후 진폭을 비교할 수 있게 한다.
import matplotlib; matplotlib.use("Agg"); import matplotlib.cm as mcm
def heat(M, name, vmax=2.5, cmap="RdBu_r", sx=3, sy=3):
    im = (mcm.get_cmap(cmap)((np.clip(M / vmax, -1, 1) + 1) / 2)[..., :3] * 255).astype(np.uint8); im = cv2.resize(im, (M.shape[1] * sx, M.shape[0] * sy), interpolation=cv2.INTER_NEAREST); cv2.imwrite(f"{OUT}/{name}", im[..., ::-1])
Xz = Xb - Xb.mean(-1, keepdims=True); sdv = Xz.std(-1, keepdims=True) + 1e-6; top = np.argsort(-sdv[:, 0])[:120]
u, sv, vt = np.linalg.svd(Xz[top] / sdv[top], full_matrices=False); order = top[np.argsort(u[:, 0])]; print("first PC share", float(sv[0] ** 2 / (sv ** 2).sum()), "PC1 peak bpm", float(f[np.argmax(spec(vt[0])[1])]))
heat(Xz[order] / sdv[order], "X_before.png"); heat((Xc - Xc.mean(-1, keepdims=True))[order] / sdv[order], "X_removed.png"); heat((Xa - Xa.mean(-1, keepdims=True))[order] / sdv[order], "X_after.png")
R = refs[0, [0, 1, 2, 6, 7]].numpy(); amp = R.std(-1, keepdims=True); a0 = np.array([5e-4] * 3 + [5e-2] * 2)[:, None]; Rz = (R - R.mean(-1, keepdims=True)) / (amp + 1e-8) * (amp / (amp + a0))
Rh = np.fft.irfft(np.fft.rfft(Rz, axis=-1) * (-1j) * (np.arange(Rz.shape[-1] // 2 + 1) > 0), n=Rz.shape[-1], axis=-1); RR = np.concatenate([Rz, Rh], 0); heat(RR, "R_matrix.png", vmax=2.0, cmap="PuOr_r", sx=3, sy=24); J["gate"] = (amp / (amp + a0))[:, 0].tolist(); print("amplitude gate", J["gate"])
# 슬롯 신호
aux = m.nf_aux; J["s_p"] = aux["s_p"][0].numpy().tolist(); J["s_n"] = aux["s_n"][0].numpy().tolist(); B_ = aux["bases"][0].numpy(); J["b_p"] = B_[:, 0].tolist(); J["b_n"] = B_[:, 1].tolist(); J["g_diff"] = aux["g_diff"][0].numpy().tolist()
for nm_, v_ in (("s_p", J["s_p"]), ("s_n", J["s_n"]), ("b_p", J["b_p"]), ("b_n", J["b_n"]), ("g_diff", J["g_diff"])):
    a_ = NFL.hp(torch.tensor(v_)[None].float())[0].numpy(); ff, ss = spec(a_); J.setdefault("peaks", {})[nm_] = float(ff[np.argmax(ss)]); print(nm_, "peak bpm", float(ff[np.argmax(ss)]), "std", float(np.std(v_)))
# 합성 쪽 재료: 실 PPG 뱅크 파형, 이중 렌더 잡음 trace, 주입 스크립트로 만든 두 렌더의 차이(프레임 원본은 생성 얼굴)
bk = np.load(f"{PFA}/ppg_bank.npz", allow_pickle=True); seg_i = int(np.argmin(np.abs(bk["f0"] - 72 / 60.0))); J["ppg_wave"] = bk["segments"][seg_i][:160].tolist(); J["ppg_f0"] = float(bk["f0"][seg_i])
nt = np.load(f"{PFA}/noise_gt_hyb2/subject261.npy"); J["n_trace"] = nt[300:460].tolist()
import importlib.util
FD = f"{ROOT}/_workspace/data/faces/diff_female_01_frames"; spec_ = importlib.util.spec_from_file_location("bsd", f"{ROOT}/generator/build_synth_dataset.py"); bsd = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(bsd)
frs = [cv2.resize(cv2.imread(f"{FD}/{i:06d}.png"), (256, 256), interpolation=cv2.INTER_AREA) for i in range(0, 1000, 25)]; skin_ = bsd.feather((bsd.stable_skin_mask(frs, step=1) > 0.5).astype(np.float32), 15)[..., None]
pbv = bsd.PBV_RGB[::-1].reshape(1, 1, 3).astype(np.float32); img = frs[0].astype(np.float32); on = img * (1.0 + skin_ * (0.01 * 1.0 * pbv)); dif = on - img; print("injection: max diff levels", float(dif.max()), "skin fraction", float((skin_ > 0.5).mean()))   # 주입식: build_synth_dataset 와 같다, alpha 0.01, p(t)=1
cv2.imwrite(f"{OUT}/inject_diff.png", cv2.resize(np.clip(dif * 80, 0, 255).astype(np.uint8), (360, 360), interpolation=cv2.INTER_LINEAR)); cv2.imwrite(f"{OUT}/inject_on.png", cv2.resize(np.clip(on, 0, 255).astype(np.uint8), (360, 360), interpolation=cv2.INTER_LINEAR))
for i, t in enumerate((0, 300, 620)): cv2.imwrite(f"{OUT}/anim_{i}.png", cv2.resize(cv2.imread(f"{FD}/{t:06d}.png"), (300, 300), interpolation=cv2.INTER_AREA))
json.dump(J, open(f"{OUT}/arch_data.json", "w"))
# 그림 재료
def save(name, im): cv2.imwrite(f"{OUT}/{name}", im[..., ::-1] if im.ndim == 3 else im)
for i, t in enumerate((0, 20, 40)): save(f"face_{i}.png", cv2.resize(crop[T0 + t], (360, 360), interpolation=cv2.INTER_AREA))
save("face_small.png", cv2.resize(small[0].astype(np.uint8), (288, 288), interpolation=cv2.INTER_NEAREST))
save("skin_weighted.png", cv2.resize(np.clip(xw[0], 0, 255).astype(np.uint8), (288, 288), interpolation=cv2.INTER_NEAREST))
cm = cv2.applyColorMap((np.clip(cv2.resize(P[0].numpy(), (288, 288), interpolation=cv2.INTER_LINEAR), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1]; save("p_skin.png", cm)
print("saved", sorted(os.listdir(OUT)))
