"""SCALE300 합성 config: 300 층화 가상얼굴(scale300_frames) × 1클립(600프레임). 사용: python make_syn_scale_config.py F|P
 F: 고정 Pbv·α(코퍼스와 동일) — 정체성 수 효과만
 P: 피부톤 의존 주입 — ITA로 멜라닌 지수 m∈[0,1] 추정 → α = α0·(1 − 0.45·m), Pbv = (1−0.5m)·Pbv0 + 0.5m·[0.25,0.85,0.45] (정규화)
    (가정: 멜라닌이 높을수록 맥동 진폭이 작고 R 성분 기여가 줄어 G 우세 — 코퍼스 사전이 아닌 모델링 가정이며 결과로 판정)
 공통: HR 42~152 균등, harmonic 코퍼스 기본(a₂ 0~0.75, a₃ 0~0.28), 교란 코퍼스 기본 범위, cumsum 라벨."""
import json, os, sys, numpy as np
V=sys.argv[1]; ROOT="/SPECPHYS_ROOT"; PF=f"{ROOT}/_workspace/data/physformer"
meta=json.load(open(f"{ROOT}/_workspace/data/faces/scale300/meta.json")); FR=f"{ROOT}/_workspace/data/faces/scale300_frames"
# R: 실 데이터셋 구동(실 표정·머리 자세) + 실 얼굴 상자 궤적 병진 이식, 고정 주입
if V in ("R","RA"):
    pairs={q["idx"]:q for q in json.load(open(f"{ROOT}/_workspace/data/faces/scale300r_pairs.json"))}; FR=f"{ROOT}/_workspace/data/faces/scale300r_frames"
PBV0=np.array([0.33,0.77,0.53]); PBVD=np.array([0.25,0.85,0.45])
rs=np.random.default_rng({"F":300,"P":301,"R":302,"RA":302}[V]); base={"F":4000,"P":4500,"R":5000,"RA":5500}[V]; subs=[]; skipped=0
for m in meta:
    fr=f"{FR}/f{m['idx']:03d}"
    if not os.path.isdir(fr) or len(os.listdir(fr))<320: skipped+=1; continue
    i=m["idx"]; u=i/(len(meta)-1)
    nz=dict(illum_drift=float(rs.uniform(0.1,0.22)), flicker_amp=float(rs.uniform(0.02,0.03)), flicker_hz=float(rs.uniform(1.0,1.3)), motion_px=float(rs.uniform(6,20)), rot_deg=float(rs.uniform(0.5,2)), sensor_std=float(rs.uniform(2,3)), color_drift=float(rs.uniform(0.02,0.05)))
    sub=dict(id=base+i+1, frames_dir=os.path.relpath(fr,ROOT), bpm=int(round(42+u*110)), alpha=0.01, seed=base+i+1, spatial="patches" if i%2 else "uniform", coverage=0.6, patch_scale=40,
             wave_a2=float(rs.uniform(0,0.75)), wave_phi2=float(rs.uniform(-0.3,0.3)), wave_a3=float(rs.uniform(0,0.28)), wave_phi3=float(rs.uniform(-0.5,0.5)), noise=nz, ita=m["ita"], tone=m["tone"])
    if V in ("R","RA"):
        if m["idx"] not in pairs: skipped+=1; continue
        q=pairs[m["idx"]]; tj=f"{ROOT}/_workspace/data/faces/driving_real/traj/{q['driving']}.npy"
        if os.path.exists(tj): nz["motion_traj"]=tj; nz["face_frac"]=0.55; nz["motion_px"]=0.0
        sub.update(driving=q["driving"], driving_ds=q["ds"], driving_cond=q["cond"])
        if V=="RA": sub.update(spatial="anatomy", amap_file=f"{ROOT}/_workspace/data/faces/scale300r_amap/f{m['idx']:03d}.npy")
    if V=="P":
        mel=float(np.clip((55-m["ita"])/85,0,1))  # ITA 55°(very light)→0, −30°(dark)→1
        sub["alpha"]=0.01*(1-0.45*mel); p=(1-0.5*mel)*PBV0+0.5*mel*PBVD; sub["pbv_rgb"]=[float(x) for x in p/np.linalg.norm(p)*np.linalg.norm(PBV0)]; sub["melanin"]=mel
    subs.append(sub)
cfg=dict(fps=30,size=256,out=f"_workspace/data/physformer/SYN_S300{V}",subjects=subs,_variant=V)
json.dump(cfg,open(f"{PF}/configs/syn_s300{V.lower()}.json","w"),indent=1); print(f"S300{V}: {len(subs)} clips (skipped {skipped})")
