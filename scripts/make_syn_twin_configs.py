"""24′/25 config: python make_syn_twin_configs.py RAT|NIRPT [N]
 RAT  : syn_s300ra(실 움직임 + A(x,y)) 전체에 실 PPG 뱅크 + 정반사 분리 → SYN_S300RAT (ids 6001~)
 NIRPT: 그중 N(기본 100) 정체성에 MR-NIRP 측정 통계(피부 RGB 이식, σ 3~17, drift 0.5~1.5) → SYN_NIRPT (ids 6501~)"""
import json,sys,numpy as np,os
V=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 100; ROOT="/SPECPHYS_ROOT"; PF=f"{ROOT}/_workspace/data/physformer"
d=json.load(open(f"{PF}/configs/syn_s300ra.json")); bank=np.load(f"{PF}/ppg_bank.npz"); f0=bank["f0"]; rs=np.random.default_rng(24 if V=="RAT" else 25)
subs=d["subjects"] if V=="RAT" else d["subjects"][::2][:N]; base=6000 if V=="RAT" else 6500
st=json.load(open(f"{PF}/../results/domain_stats_NIRP.json"))["summary"] if V=="NIRPT" else None
for i,sub in enumerate(subs):
    sub["id"]=base+i+1; sub["seed"]=base+i+1; tgt=sub["bpm"]/60; cand=np.argsort(np.abs(np.log(f0/tgt)))[:12]
    sub.update(wave_bank=f"{PF}/ppg_bank.npz", wave_bank_idx=int(rs.choice(cand)), spec_exclude=True)
    if V=="NIRPT":
        sub["tint_rgb"]=[float(x) for x in st["skin_rgb_mean"]]; nz=sub["noise"]; nz["sensor_std"]=float(rs.uniform(3,17)); nz["illum_drift"]=float(rs.uniform(0.5,1.5)); nz["flicker_amp"]=float(rs.uniform(0.0,0.01))
d["subjects"]=subs; d["out"]=f"_workspace/data/physformer/SYN_S300{V}" if V=="RAT" else "_workspace/data/physformer/SYN_NIRPT"
json.dump(d,open(f"{PF}/configs/syn_{V.lower()}.json","w"),indent=1); print(V,len(subs),"clips")
