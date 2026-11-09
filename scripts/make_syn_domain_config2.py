"""16단계 재설계(2요인 분리) 도메인 특화 합성 config. 사용: python make_syn_domain_config2.py MMPD A|B
 A(외형 요인만): 대상 정체성 정지 프레임 애니메이션 + 코퍼스 기본 harmonic(a2≤0.75, a3≤0.28) + 코퍼스 기본 교란 범위
 B(물리 요인만): 가상얼굴 정체성 9종 + 대상 피부 RGB 통계 이식(tint) + 측정 교란 p10~p50×1.5(≤p90) + 기본 harmonic
 공통: HR은 측정 범위 포함(0.8·p10 ~ 1.2·p90), 24클립, cumsum 라벨."""
import json, glob, os, sys, numpy as np
D, V = sys.argv[1], sys.argv[2]; ROOT="/SPECPHYS_ROOT"; PF=f"{ROOT}/_workspace/data/physformer"
st=json.load(open(f"{PF}/../results/domain_stats_{D}.json"))["summary"]
if V=="A": ids=sorted(glob.glob(f"{ROOT}/_workspace/data/faces/{D.lower()}_s*_frames"))
else: ids=sorted(glob.glob(f"{ROOT}/_workspace/data/faces/diff_*_0?_frames")+glob.glob(f"{ROOT}/_workspace/data/faces/sfhq*_frames"))
assert ids
hr_lo=max(40,st["hr"]["p10"]*0.8); hr_hi=min(160,st["hr"]["p90"]*1.2)
def meas(k): return st[k]["p10"], min(st[k]["p90"], st[k]["p50"]*1.5)
if V=="A": drift=(0.1,0.22); mot=(6,20); sens=(2,3); fl_hz=(1.0,1.3); fl_amp=(0.02,0.03); cd=(0.02,0.05)
else: drift=meas("drift"); mot=meas("motion_px"); sens=meas("sensor_std"); fl_hz=(st["flicker_hz"]["p10"],st["flicker_hz"]["p90"]); fl_amp=meas("flicker_amp"); cd=(0.02,0.05)
rs=np.random.default_rng(int(sum(map(ord,D+V)))); base={"MMPD":{"A":3100,"B":3200},"NIRP":{"A":3600,"B":3700}}[D][V]; N=24; subs=[]
for i in range(N):
    fr=ids[i%len(ids)]; u=i/(N-1)
    nz=dict(illum_drift=float(rs.uniform(*drift)), flicker_amp=float(rs.uniform(*fl_amp)), flicker_hz=float(rs.uniform(fl_hz[0],max(fl_hz[1],fl_hz[0]+0.1))), motion_px=float(rs.uniform(*mot)), rot_deg=float(rs.uniform(0.5,2)), sensor_std=float(rs.uniform(*sens)), color_drift=float(rs.uniform(*cd)))
    sub=dict(id=base+i+1, frames_dir=os.path.relpath(fr,ROOT), bpm=int(round(hr_lo+u*(hr_hi-hr_lo))), alpha=0.01, seed=base+i+1, spatial="patches" if i%2 else "uniform", coverage=0.6, patch_scale=40,
             wave_a2=float(rs.uniform(0.0,0.75)), wave_phi2=float(rs.uniform(-0.3,0.3)), wave_a3=float(rs.uniform(0.0,0.28)), wave_phi3=float(rs.uniform(-0.5,0.5)), noise=nz)
    if V=="B": sub["tint_rgb"]=[float(x) for x in st["skin_rgb_mean"]]
    subs.append(sub)
cfg=dict(fps=30,size=256,out=f"_workspace/data/physformer/SYN_{D}_{V}",subjects=subs,_variant=V,_measured=st)
json.dump(cfg,open(f"{PF}/configs/syn_{D.lower()}_{V.lower()}.json","w"),indent=1)
print(f"{D}/{V}: {N} clips, {len(ids)} identities | HR {hr_lo:.0f}-{hr_hi:.0f} | drift {drift[0]:.2f}-{drift[1]:.2f} motion {mot[0]:.1f}-{mot[1]:.1f} sensor {sens[0]:.1f}-{sens[1]:.1f} flicker {fl_hz[0]:.2f}-{fl_hz[1]:.2f}Hz amp {fl_amp[0]:.3f}-{fl_amp[1]:.3f}")
