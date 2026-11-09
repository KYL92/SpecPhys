"""디지털 트윈 v1 config: train20h/6hi/6lo의 42→32 가상얼굴 클립을 실 PPG 파형 뱅크 + 정반사 분리로 재렌더(교란·시드·정체성 동일).
뱅크 인덱스는 목표 BPM에 f0가 가까운 세그먼트 중에서 시드로 선택(리샘플 왜곡 최소화)."""
import json, numpy as np
ROOT="/SPECPHYS_ROOT"; PF=f"{ROOT}/_workspace/data/physformer"
bank=np.load(f"{PF}/ppg_bank.npz"); f0=bank["f0"]; rs=np.random.default_rng(23)
for c in ["train20h","train6hi","train6lo"]:
    d=json.load(open(f"{PF}/configs/{c}.json"))
    for sub in d["subjects"]:
        tgt=sub["bpm"]/60.0; cand=np.argsort(np.abs(np.log(f0/tgt)))[:12]
        sub.update(wave_bank=f"{PF}/ppg_bank.npz", wave_bank_idx=int(rs.choice(cand)), spec_exclude=True)
    d["out"]=f"_workspace/data/physformer/{c.upper()}_T"
    json.dump(d,open(f"{PF}/configs/{c}_t.json","w"),indent=1); print(c,len(d["subjects"]),"ok")
