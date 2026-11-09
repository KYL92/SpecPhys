"""실 데이터셋 학습 분할 영상 → LivePortrait 구동 클립 풀(20초 h264 mp4). held-out/zero-shot 셋(MMPD 7~33, UBFC 42~49, PURE 07~10, MCD 평가 20명, DLCN, MR-NIRP)은 절대 사용하지 않음.
출력: faces/driving_real/{tag}.mp4 + index.json (출처·조건). 사용: python prep_real_driving.py"""
import os, glob, json, subprocess, random
ROOT="/SPECPHYS_ROOT"; PF=f"{ROOT}/_workspace/data/physformer"; OUT=f"{ROOT}/_workspace/data/faces/driving_real"; os.makedirs(OUT,exist_ok=True)
items=[]
for d in sorted(glob.glob(f"{PF}/MMPD_TRAIN/subject*")):
    src=open(f"{d}/source.txt").read().split(); tag=f"mmpd_{os.path.basename(d)[7:]}"; mot=[s for s in src if s.startswith("motion=")][0][7:]
    items.append(dict(tag=tag,src=f"{d}/vid.avi",ds="MMPD",cond=mot))
for d in sorted(glob.glob(f"{PF}/PURE_TRAIN/subject*")): items.append(dict(tag=f"pure_{os.path.basename(d)[7:]}",src=f"{d}/vid.avi",ds="PURE",cond="task"+os.path.basename(d)[-2:]))
for d in sorted(glob.glob(f"{PF}/MCD_TRAIN/subject*")): items.append(dict(tag=f"mcd_{os.path.basename(d)[7:]}",src=f"{d}/vid.avi",ds="MCD",cond="stationary/post-exercise"))
for i in range(1,42):
    p=f"{ROOT}/UBFC_dataset/DATASET_2/subject{i}/vid.avi"
    if os.path.exists(p): items.append(dict(tag=f"ubfc_{i}",src=p,ds="UBFC",cond="talking/game"))
random.seed(7); sel=[]
by={}; [by.setdefault(it["ds"],[]).append(it) for it in items]
quota={"MMPD":120,"PURE":35,"MCD":24,"UBFC":24}
for ds,q in quota.items(): xs=by.get(ds,[]); random.shuffle(xs); sel+=xs[:q]
ok=[]
for it in sel:
    o=f"{OUT}/{it['tag']}.mp4"
    if not os.path.exists(o):
        r=subprocess.run(["ffmpeg","-v","error","-y","-i",it["src"],"-t","20","-r","30","-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2","-c:v","libx264","-crf","18","-pix_fmt","yuv420p","-an",o],capture_output=True)
        if r.returncode: print("FF_FAIL",it["tag"],r.stderr[-200:]); continue
    it["mp4"]=o; ok.append(it)
json.dump(ok,open(f"{OUT}/index.json","w"),indent=1); import collections; print("driving clips",len(ok),collections.Counter(x["ds"] for x in ok),collections.Counter(x["cond"] for x in ok if x["ds"]=="MMPD"))
