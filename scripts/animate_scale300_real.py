"""SCALE300-R: 203 정체성(LivePortrait 검출 성공분) × 실 데이터셋 구동 클립 203개(1:1, 시드 고정) → LivePortrait → 프레임(faces/scale300r_frames/f###/).
소스는 Haar 얼굴 상자 1.8배로 재크롭(기존 코퍼스와 같은 얼굴 비율). 사용: python animate_scale300_real.py [limit]"""
import os, sys, json, glob, random, subprocess, cv2, numpy as np
ROOT="/SPECPHYS_ROOT"; F=f"{ROOT}/_workspace/data/faces"; LIM=int(sys.argv[1]) if len(sys.argv)>1 else 10**9
meta=json.load(open(f"{F}/scale300/meta.json")); usable=[m for m in meta if os.path.isdir(f"{F}/scale300_frames/f{m['idx']:03d}")]
drv=json.load(open(f"{F}/driving_real/index.json")); random.seed(11); random.shuffle(drv)
SRC=f"{F}/scale300_crop"; ANIM=f"{F}/scale300r_anim"; FR=f"{F}/scale300r_frames"; [os.makedirs(d,exist_ok=True) for d in (SRC,ANIM,FR)]
haar=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml"); pairs=[]; fail=0
for k,m in enumerate(usable[:LIM]):
    b=f"f{m['idx']:03d}"; d=drv[k%len(drv)]; src=f"{SRC}/{b}.png"
    if not os.path.exists(src):
        im=cv2.imread(m["file"]); h,w=im.shape[:2]; det=haar.detectMultiScale(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),1.1,5,minSize=(80,80))
        if len(det): x,y,bw,bh=max(det,key=lambda r:r[2]*r[3]); cx,cy,s=x+bw/2,y+bh/2,max(bw,bh)*1.8
        else: cx,cy,s=w/2,h*0.45,w*0.7
        x0,y0=int(max(0,cx-s/2)),int(max(0,cy-s/2)); x1,y1=int(min(w,cx+s/2)),int(min(h,cy+s/2)); cv2.imwrite(src,cv2.resize(im[y0:y1,x0:x1],(512,512),interpolation=cv2.INTER_AREA))
    out=glob.glob(f"{ANIM}/{b}--{d['tag']}.mp4")
    if not out:
        subprocess.run(["python","repos/LivePortrait/inference.py","-s",src,"-d",d["mp4"],"-o",ANIM],cwd=ROOT,capture_output=True)
        out=glob.glob(f"{ANIM}/{b}--{d['tag']}.mp4")
    if not out: fail+=1; print("ANIM_FAIL",b,d["tag"],flush=True); continue
    fd=f"{FR}/{b}"
    if not os.path.isdir(fd) or len(os.listdir(fd))<300:
        os.makedirs(fd,exist_ok=True); subprocess.run(["ffmpeg","-v","error","-y","-i",out[0],"-frames:v","600","-start_number","0",f"{fd}/%06d.png"],capture_output=True)
    pairs.append(dict(idx=m["idx"],frames=fd,driving=d["tag"],ds=d["ds"],cond=d["cond"],tone=m["tone"],ita=m["ita"]))
    if (k+1)%25==0: print(f"ANIMR {k+1}/{len(usable)} fail={fail}",flush=True)
json.dump(pairs,open(f"{F}/scale300r_pairs.json","w"),indent=1); print(f"ANIMR_DONE n={len(pairs)} fail={fail}",flush=True)
