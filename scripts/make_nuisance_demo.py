#!/usr/bin/env python3
"""합성 코퍼스 nuisance taxonomy 데모 영상: 동일 프레임에 각 nuisance를 실제 코퍼스 파라미터로 적용해 4×3 타일로 비교.
출력: _workspace/slides/nuisance_demo.mp4 (+ 대표 프레임 png). build_synth_dataset.py의 적용 순서(주입→운동→조명→색온도→센서)와 동일."""
import glob, os, subprocess, sys, tempfile
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/SPECPHYS_ROOT/generator")
sys.path.insert(0, "/SPECPHYS_ROOT/generator")
from inject_pulse import make_waveform, spatial_amplitude_map, feather
from build_synth_dataset import stable_skin_mask, smooth_walk, PBV_RGB
import argparse
_ap=argparse.ArgumentParser(); _ap.add_argument("--identity",default="sfhq04"); _ap.add_argument("--tag",default=""); _ap.add_argument("--start",type=int,default=0); _a=_ap.parse_args()
FR=f"/SPECPHYS_ROOT/_workspace/data/faces/{_a.identity}_frames"; OUT="/SPECPHYS_ROOT/_workspace/slides"; TAG=_a.tag
FPS, N, SZ, BPM, ALPHA = 30, 300, 256, 72, 0.01
paths=sorted(glob.glob(f"{FR}/*.png"))[_a.start:_a.start+N]; frames=[cv2.resize(cv2.imread(p),(SZ,SZ),interpolation=cv2.INTER_AREA) for p in paths]; n=len(frames)
t=np.arange(n)/FPS
ppg,_=make_waveform(n,FPS,BPM,0.0,seed=7,wave_a2=0.3,wave_phi2=0.2,wave_a3=0.1,wave_phi3=0.0)
skin=feather((stable_skin_mask(frames)>0.5).astype(np.float32),15)
pbv=PBV_RGB[::-1].reshape(1,1,3)
def inject(amap,alpha=ALPHA):
    mask=(skin*amap)[...,None]; out=[]
    for i,img in enumerate(frames):
        g=1.0+alpha*ppg[i]*pbv; out.append(img.astype(np.float32)*(1.0+mask*(g-1.0)))
    return out
uni=np.ones((SZ,SZ),np.float32); amap_p=spatial_amplitude_map((SZ,SZ),"patches",0.5,40,1.0,0.35,seed=7)
base=inject(uni)
def apply(imgs,seed=0,illum_drift=0,flicker_amp=0,flicker_hz=1.1,motion_px=0,rot_deg=0,scale_jit=0,pm_amp=0,pm_hz=2.5,color_drift=0,sensor=0):
    rng=np.random.default_rng(seed)
    drift=smooth_walk(n,illum_drift/np.sqrt(n),FPS,rng) if illum_drift>0 else np.zeros(n); drift-=drift.mean()
    flick=flicker_amp*np.sin(2*np.pi*flicker_hz*t) if flicker_amp>0 else np.zeros(n); gain=np.clip(1+drift+flick,0.3,1.7)
    tx=smooth_walk(n,motion_px/np.sqrt(n),FPS//2,rng) if motion_px>0 else np.zeros(n); ty=smooth_walk(n,motion_px/np.sqrt(n),FPS//2,rng) if motion_px>0 else np.zeros(n)
    if pm_amp>0: ty=ty+pm_amp*np.sin(2*np.pi*pm_hz*t)
    rot=smooth_walk(n,rot_deg/np.sqrt(n),FPS//2,rng) if rot_deg>0 else np.zeros(n); sc=1+(smooth_walk(n,scale_jit/np.sqrt(n),FPS//2,rng) if scale_jit>0 else np.zeros(n))
    chg=np.stack([1+smooth_walk(n,color_drift/np.sqrt(n),FPS,rng) for _ in range(3)],1) if color_drift>0 else None
    mv=motion_px>0 or rot_deg>0 or scale_jit>0 or pm_amp>0; out=[]
    for i,img in enumerate(imgs):
        img=img.copy()
        if mv:
            M=cv2.getRotationMatrix2D((SZ/2,SZ/2),float(rot[i]),float(sc[i])); M[:,2]+=(tx[i],ty[i]); img=cv2.warpAffine(img,M,(SZ,SZ),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
        img=img*gain[i]
        if chg is not None: img=img*chg[i].reshape(1,1,3)
        if sensor>0: img=img+rng.normal(0,sensor,img.shape)
        out.append(np.clip(img,0,255))
    return out
def mpeg4(imgs,q=6):
    d=tempfile.mkdtemp(); raw=os.path.join(d,"c.avi")
    p=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s",f"{SZ}x{SZ}","-r",str(FPS),"-i","-","-c:v","mpeg4","-q:v",str(q),raw],stdin=subprocess.PIPE)
    for im in imgs: p.stdin.write(np.ascontiguousarray(im.astype(np.uint8)).tobytes())
    p.stdin.close(); p.wait(); cap=cv2.VideoCapture(raw); out=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        out.append(f.astype(np.float32))
    return out[:n] if len(out)>=n else out+[out[-1]]*(n-len(out))
def amplified(imgs,k=40):  # 주입 변조를 ×k 과장 (원본 대비 차이 증폭)
    return [np.clip(frames[i].astype(np.float32)+k*(imgs[i]-frames[i].astype(np.float32)),0,255) for i in range(n)]
print("rendering tiles...",flush=True)
tiles=[
 ("① clean (주입 ×1, 눈에 안 보임)","α=1%, 72 BPM, harmonic a2=0.3",base),
 ("② 주입 ×40 과장","pulse injection I←I·(1+α·A·p(t)·Pbv)",amplified(base)),
 ("③ perfusion heterogeneity ×40","A(x,y) patches, coverage 0.5",amplified(inject(amap_p))),
 ("④ illumination flicker","1.1 Hz, amp 0.03 (심박 대역 안)",apply(base,1,flicker_amp=0.03,flicker_hz=1.1)),
 ("⑤ illumination drift","random-walk gain, 0.15",apply(base,2,illum_drift=0.15)),
 ("⑥ colour-temperature drift","채널별 저주파 게인, 0.11",apply(base,3,color_drift=0.11)),
 ("⑦ rigid motion (random walk)","10 px, 2.0°",apply(base,4,motion_px=10,rot_deg=2.0)),
 ("⑧ periodic nod","4 px @ 2.5 Hz 정현파 병진",apply(base,5,pm_amp=4,pm_hz=2.5)),
 ("⑨ scale / vibration jitter","scale_jit 0.03 + 6 px",apply(base,6,motion_px=6,scale_jit=0.03)),
 ("⑩ sensor noise","gaussian σ=3 / pixel",apply(base,7,sensor=3)),
 ("⑪ mpeg4 compression","q=6 재인코딩 (블록·링잉)",mpeg4(base,6)),
 ("⑫ driving stress (복합)","DRIVE6: 20px·3.9°·nod·drift·flicker·colour·σ3",apply(base,8,illum_drift=0.22,flicker_amp=0.025,flicker_hz=1.0,motion_px=20,rot_deg=3.9,scale_jit=0.03,pm_amp=4,pm_hz=2.5,color_drift=0.11,sensor=3)),
]
font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",15); font2=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",12); fontH=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",20)
COLS,ROWS,LAB,HDR,PAD=4,3,46,52,6; TW=SZ+PAD; TH=SZ+LAB+PAD; W=COLS*TW+PAD; H=ROWS*TH+HDR+PAD
def compose(i):
    canvas=Image.new("RGB",(W,H),(24,24,26)); dr=ImageDraw.Draw(canvas)
    dr.text((10,12),f"합성 코퍼스 nuisance taxonomy — 동일 정체성·동일 주입(72 BPM)에 각 교란 적용   t = {i/FPS:4.1f} s",font=fontH,fill=(240,240,240))
    for k,(title,sub,imgs) in enumerate(tiles):
        r,c=divmod(k,COLS); x=PAD+c*TW; y=HDR+PAD+r*TH
        im=Image.fromarray(cv2.cvtColor(imgs[i].astype(np.uint8),cv2.COLOR_BGR2RGB)); canvas.paste(im,(x,y))
        dr.rectangle([x,y+SZ,x+SZ,y+SZ+LAB],fill=(40,40,44)); dr.text((x+6,y+SZ+3),title,font=font,fill=(255,255,255)); dr.text((x+6,y+SZ+24),sub,font=font2,fill=(180,180,180))
    return np.array(canvas)
mp4=f"{OUT}/nuisance_demo{TAG}.mp4"
p=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-","-c:v","libx264","-preset","medium","-crf","18","-pix_fmt","yuv420p",mp4],stdin=subprocess.PIPE)
for i in range(n):
    fr=compose(i); p.stdin.write(fr.tobytes())
    if i==int(1.4*FPS): Image.fromarray(fr).save(f"{OUT}/nuisance_demo{TAG}_frame.png")
p.stdin.close(); print("ffmpeg rc",p.wait(), mp4, f"{W}x{H}")
