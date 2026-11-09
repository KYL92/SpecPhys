"""실 구동 클립의 얼굴 상자 궤적(중심 dx,dy / 얼굴폭) 추출 → driving_real/traj/{tag}.npy (600프레임, 결측은 보간)"""
import os, json, cv2, numpy as np
F="/SPECPHYS_ROOT/_workspace/data/faces/driving_real"; os.makedirs(f"{F}/traj",exist_ok=True)
haar=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml"); stats=[]
for it in json.load(open(f"{F}/index.json")):
    o=f"{F}/traj/{it['tag']}.npy"
    if os.path.exists(o): continue
    cap=cv2.VideoCapture(it["mp4"]); cs=[]; ws=[]
    while True:
        ok,f=cap.read()
        if not ok or len(cs)>=600: break
        g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY); det=haar.detectMultiScale(g,1.1,4,minSize=(40,40))
        if len(det): x,y,w,h=max(det,key=lambda r:r[2]*r[3]); cs.append((x+w/2,y+h/2)); ws.append(w)
        else: cs.append((np.nan,np.nan)); ws.append(np.nan)
    c=np.array(cs,dtype=float); w=np.nanmedian(ws) if np.isfinite(ws).any() else np.nan
    if not np.isfinite(w) or np.isfinite(c[:,0]).sum()<30: print("TRAJ_FAIL",it["tag"]); continue
    t=np.arange(len(c))
    for j in range(2): m=np.isfinite(c[:,j]); c[:,j]=np.interp(t,t[m],c[m,j])
    tr=c/w; tr=np.stack([np.convolve(tr[:,j],np.ones(5)/5,mode="same") for j in range(2)],1); np.save(o,tr.astype(np.float32))
    stats.append((it["tag"],it["ds"],it["cond"],float(np.percentile(np.abs(tr-tr.mean(0)).max(1),90))))
print("done",len(stats)); import collections
for ds in ("MMPD","PURE","MCD","UBFC"): v=[s[3] for s in stats if s[1]==ds]; v and print(ds,"p90 |disp|/facewidth median",round(float(np.median(v)),3))
