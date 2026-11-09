"""도메인 특화 합성 config 생성: 측정 통계(domain_stats_<D>.json)의 p10~p90 범위를 '포함'하도록(정합이 아니라 커버) 파라미터를 잡고,
정체성 소스는 대상 셋 학습 분할의 정지 프레임 애니메이션(animations_<d>/*_frames)을 사용. 사용: python make_syn_domain_config.py MMPD"""
import json, glob, os, sys, numpy as np
D=sys.argv[1]; PF="/SPECPHYS_ROOT/_workspace/data/physformer"; st=json.load(open(f"{PF}/../results/domain_stats_{D}.json"))["summary"]
ids=sorted(glob.glob(f"/SPECPHYS_ROOT/_workspace/data/faces/{D.lower()}_s*_frames")); assert ids, "no identity frames"
def rng(k,lo_ext=0.8,hi_ext=1.2): return st[k]["p10"]*lo_ext, st[k]["p90"]*hi_ext
hr_lo,hr_hi=rng("hr"); hr_lo=max(40,hr_lo); hr_hi=min(160,hr_hi)
a2_lo,a2_hi=rng("a2"); a3_lo,a3_hi=rng("a3"); fl_hz=(st["flicker_hz"]["p10"],st["flicker_hz"]["p90"]); fl_amp=rng("flicker_amp"); drift=rng("drift"); mot=rng("motion_px"); sens=rng("sensor_std")
rs=np.random.default_rng(int(sum(map(ord,D)))); subs=[]; base=3000 if D=="MMPD" else 3500
N=len(ids)*2  # 정체성당 2클립(조건 다르게)
for i in range(N):
    fr=ids[i%len(ids)]; u=i/(N-1) if N>1 else 0.5
    nz=dict(illum_drift=float(rs.uniform(*drift)), flicker_amp=float(rs.uniform(*fl_amp)), flicker_hz=float(rs.uniform(fl_hz[0],max(fl_hz[1],fl_hz[0]+0.1))), motion_px=float(rs.uniform(*mot)), rot_deg=float(rs.uniform(0.5,3)), sensor_std=float(rs.uniform(*sens)), color_drift=float(rs.uniform(0.02,0.08)))
    subs.append(dict(id=base+i+1, frames_dir=os.path.relpath(fr,"/SPECPHYS_ROOT"), bpm=int(round(hr_lo+u*(hr_hi-hr_lo))), alpha=0.01, seed=base+i+1, spatial="patches" if i%2 else "uniform", coverage=0.6, patch_scale=40,
                     wave_a2=float(rs.uniform(a2_lo,min(a2_hi,0.9))), wave_phi2=float(rs.uniform(-0.3,0.3)), wave_a3=float(rs.uniform(a3_lo,min(a3_hi,0.5))), wave_phi3=float(rs.uniform(-0.5,0.5)), noise=nz))
cfg=dict(fps=30,size=256,out=f"_workspace/data/physformer/SYN_{D}",subjects=subs,_measured=st)
json.dump(cfg,open(f"{PF}/configs/syn_{D.lower()}.json","w"),indent=1)
print(f"{D}: {N} clips | HR {hr_lo:.0f}-{hr_hi:.0f} | a2 {a2_lo:.2f}-{a2_hi:.2f} a3 {a3_lo:.2f}-{a3_hi:.2f} | flicker {fl_hz[0]:.2f}-{fl_hz[1]:.2f}Hz amp {fl_amp[0]:.3f}-{fl_amp[1]:.3f} | drift {drift[0]:.2f}-{drift[1]:.2f} | motion {mot[0]:.1f}-{mot[1]:.1f}px | sensor {sens[0]:.1f}-{sens[1]:.1f}")
