#!/usr/bin/env python3
"""[체인 52] MMPD 전 트라이얼 속성 사전 → _workspace/data/physformer/mmpd_trials.json {sid(7SSTT+700000): subject, trial, light(1~4), motion(1~4), exercise(1/2), skin}
agg_mmpd.py 의 코드 규약(LIGHT 1 LED-low 2 LED-high 3 Incandescent 4 Nature; MOTION 1 Stationary 2 Rotation 3 Talking 4 Walking; exercise 1=yes 2=no)과 동일."""
import glob, json, scipy.io as sio
SRC = "/SPECPHYS_RAW/MMPD"; OUT = "/SPECPHYS_ROOT/_workspace/data/physformer/mmpd_trials.json"
LIGHT = {"LED-low": 1, "LED-high": 2, "Incandescent": 3, "Nature": 4}; MOTION = {"Stationary": 1, "Stationary (after exercise)": 1, "Rotation": 2, "Talking": 3, "Walking": 4, "Watching Videos": 4}  # agg_mmpd.py 의 MO 사전과 동일
d = {}
for mp in sorted(glob.glob(f"{SRC}/subject*/p*_*.mat")):
    s = int(mp.split("/subject")[1].split("/")[0]); t = int(mp.split("_")[-1][:-4])
    m = sio.loadmat(mp, variable_names=["light", "motion", "exercise", "skin_color"])
    d[str(700000 + s * 100 + t)] = dict(subject=s, trial=t, light=LIGHT[str(m["light"][0])], motion=MOTION[str(m["motion"][0])],
                                         exercise=1 if str(m["exercise"][0]) == "True" else 2, skin=int(m["skin_color"][0][0]))
keys = [(v["subject"], v["light"], v["motion"], v["exercise"], v["skin"]) for v in d.values()]
print("trials", len(d), "subjects", len(set(v["subject"] for v in d.values())), "unique (subject,light,motion,exercise,skin) keys", len(set(keys)))
json.dump(d, open(OUT, "w"), indent=0); print("wrote", OUT)
