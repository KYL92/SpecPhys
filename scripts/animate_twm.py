"""체인 30 TWM: TW 코퍼스 32클립(정체성 = 기존 SFHQ 5종 그대로)에 실 데이터셋 구동 32종(층화: MMPD 4동작×4, PURE 6, MCD 5, UBFC 5)을 1:1 배정해
LivePortrait 애니메이션 → faces/twm_frames/<cfg>_s<id>/ (600프레임) → configs/{cfg}_twm.json (frames_dir만 교체, 파형 뱅크·교란·시드 동일)"""
import json, os, glob, random, subprocess
ROOT="/SPECPHYS_ROOT"; F=f"{ROOT}/_workspace/data/faces"; PF=f"{ROOT}/_workspace/data/physformer"
drv=json.load(open(f"{F}/driving_real/index.json")); random.seed(30)
by={}; [by.setdefault((d["ds"],d["cond"] if d["ds"]=="MMPD" else ""),[]).append(d) for d in drv]
plan=[]
for key,q in [(("MMPD","Stationary"),4),(("MMPD","Rotation"),4),(("MMPD","Talking"),4),(("MMPD","Walking"),4),(("PURE",""),6),(("MCD",""),5),(("UBFC",""),5)]:
    xs=by[key][:]; random.shuffle(xs); plan+=xs[:q]
random.shuffle(plan); OUT=f"{F}/twm_anim"; FR=f"{F}/twm_frames"; os.makedirs(OUT,exist_ok=True); os.makedirs(FR,exist_ok=True); k=0; fail=0
for c in ["train20h","train6hi","train6lo"]:
    d=json.load(open(f"{PF}/configs/{c}_tw.json"))
    for sub in d["subjects"]:
        ident=os.path.basename(sub["frames_dir"]).replace("_frames",""); still=f"{F}/photoreal/{ident}.jpg"
        if not os.path.exists(still): still=f"{F}/photoreal/{ident}.png"
        dr=plan[k%len(plan)]; k+=1; tag=f"{c}_s{sub['id']}"; fd=f"{FR}/{tag}"
        if not (os.path.isdir(fd) and len(os.listdir(fd))>=300):
            m=glob.glob(f"{OUT}/{ident}--{dr['tag']}.mp4")
            if not m:
                subprocess.run(["python","repos/LivePortrait/inference.py","-s",still,"-d",dr["mp4"],"-o",OUT],cwd=ROOT,capture_output=True); m=glob.glob(f"{OUT}/{ident}--{dr['tag']}.mp4")
            if not m: fail+=1; print("TWM_ANIM_FAIL",tag,dr["tag"],flush=True); continue
            os.makedirs(fd,exist_ok=True); subprocess.run(["ffmpeg","-v","error","-y","-i",m[0],"-frames:v","600","-start_number","0",f"{fd}/%06d.png"],capture_output=True)
        sub["frames_dir"]=os.path.relpath(fd,ROOT); sub["driving"]=dr["tag"]; sub["driving_ds"]=dr["ds"]; sub["driving_cond"]=dr["cond"]
    d["out"]=f"_workspace/data/physformer/{c.upper()}_TWM"; json.dump(d,open(f"{PF}/configs/{c}_twm.json","w"),indent=1)
print(f"TWM_ANIM_DONE n={k} fail={fail}",flush=True)
