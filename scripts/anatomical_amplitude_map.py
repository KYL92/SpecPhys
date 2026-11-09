"""해부학 기반 공간 진폭 지도 A(x,y): 얼굴 파싱(SegFormer jonathandinu/face-parsing) → 이마·볼 고관류(1.0), 코(0.7), 턱(0.5), 기타 피부(0.3),
눈·눈썹·입술·안경·머리카락·귀·목·수염(어두운 저채도 피부 텍스처) = 0. 사람마다 다른 유효 맥동 공간: 부위 가중 lognormal 지터, 좌우 비대칭,
평활 랜덤필드 곱, 고관류 영역의 0~40% 무작위 탈락. 평균(비0)=1로 정규화(α 의미 유지).
사용: python anatomical_amplitude_map.py <frame.png> <out.npy> <seed> [preview.png]   또는  --batch <pairs.json> <out_dir>"""
import sys, os, json, numpy as np, cv2, torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
_proc=_model=None
def parse(bgr):
    global _proc,_model
    if _model is None:
        _proc=SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing"); _model=SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").eval().cuda()
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); inp=_proc(images=rgb,return_tensors="pt").to("cuda")
    with torch.no_grad(): lg=_model(**inp).logits
    lg=torch.nn.functional.interpolate(lg,size=bgr.shape[:2],mode="bilinear",align_corners=False)
    return lg.argmax(1)[0].cpu().numpy()
SKIN,NOSE,EYEG,LEYE,REYE,LBROW,RBROW,LEAR,REAR,MOUTH,ULIP,LLIP,HAIR=1,2,3,4,5,6,7,8,9,10,11,12,13
def amap(bgr, seed, size=256):
    lab=parse(bgr); bgr=cv2.resize(bgr,(size,size),interpolation=cv2.INTER_AREA); lab=cv2.resize(lab.astype(np.uint8),(size,size),interpolation=cv2.INTER_NEAREST)
    rng=np.random.default_rng(seed); H,W=lab.shape; rows=np.arange(H)[:,None]*np.ones((1,W)); cols=np.ones((H,1))*np.arange(W)[None,:]
    skin=(lab==SKIN); nose=(lab==NOSE)
    def rng_rows(m): r=np.where(m.any(1))[0]; return (r.min(),r.max()) if len(r) else (None,None)
    bt,_=rng_rows((lab==LBROW)|(lab==RBROW)); _,eb=rng_rows((lab==LEYE)|(lab==REYE)); mt,mb=rng_rows((lab==MOUTH)|(lab==ULIP)|(lab==LLIP)); nr=np.where(nose.any(0))[0]
    if bt is None: bt=int(H*0.35)
    if eb is None: eb=int(H*0.45)
    if mt is None: mt,mb=int(H*0.7),int(H*0.78)
    nl,nrr=(nr.min(),nr.max()) if len(nr) else (int(W*0.42),int(W*0.58))
    W_=dict(forehead=1.0,cheek=1.0,nose=0.7,chin=0.5,other=0.3)
    jit={k:float(v*np.exp(rng.normal(0,0.25))) for k,v in W_.items()}   # 사람별 부위 가중 지터
    A=np.zeros((H,W),np.float32)
    A[skin]=jit["other"]
    A[skin&(rows<bt)]=jit["forehead"]
    cheek=skin&(rows>eb)&(rows<mb)&((cols<nl-2)|(cols>nrr+2)); A[cheek]=jit["cheek"]
    A[skin&(rows>mb)]=jit["chin"]
    A[nose]=jit["nose"]
    # 수염: 입 아래·볼 하단 피부 중 어두운 저채도 고텍스처 픽셀 → 0
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV).astype(np.float32); lum=hsv[...,2]; sat=hsv[...,1]
    ref=np.median(lum[skin&(rows<eb)]) if (skin&(rows<eb)).any() else np.median(lum[skin])
    beard=skin&(rows>eb+(mt-eb)*0.6)&(lum<0.62*ref)&(sat<0.8*np.median(sat[skin])); beard=cv2.morphologyEx(beard.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8)).astype(bool); A[beard]=0
    # 좌우 비대칭
    asym=float(rng.uniform(0.8,1.2)); side=cols<W/2; A[side]*=asym; A[~side]*=2-asym
    # 사람별 평활 랜덤필드 곱 + 고관류 영역 일부 탈락(유효 맥동 공간)
    k=41; f=cv2.GaussianBlur(rng.normal(0,1,(H,W)).astype(np.float32),(k,k),0); f=(f-f.mean())/(f.std()+1e-6)
    sig=float(rng.uniform(0.15,0.5)); A*=np.exp(sig*f-sig*sig/2)
    drop=float(rng.uniform(0.0,0.4)); g=cv2.GaussianBlur(rng.normal(0,1,(H,W)).astype(np.float32),(k,k),0); thr=np.quantile(g,drop); keep=(g>=thr).astype(np.float32)
    keep=cv2.GaussianBlur(keep,(15,15),0); A*=keep
    A=cv2.GaussianBlur(A,(5,5),0); nz=A>0.02
    if nz.any(): A[nz]/=A[nz].mean()
    return A.astype(np.float32), dict(jit=jit,asym=asym,sigma=sig,drop=drop,beard_px=int(beard.sum()))
def preview(bgr,A,path):
    bgr=cv2.resize(bgr,A.shape[::-1]); hm=cv2.applyColorMap(np.clip(A/max(A.max(),1e-6)*255,0,255).astype(np.uint8),cv2.COLORMAP_JET); cv2.imwrite(path,np.hstack([bgr,cv2.addWeighted(bgr,0.5,hm,0.5,0)]))
if __name__=="__main__":
    if sys.argv[1]=="--config":
        # 기존 빌더 config → 각 subject의 frames_dir 첫 프레임에서 A(x,y)(seed=subject seed) 계산, spatial=anatomy 로 바꾼 새 config 저장
        cfg=json.load(open(sys.argv[2])); out_cfg=sys.argv[3]; ad=sys.argv[4]; new_out=sys.argv[5]; os.makedirs(ad,exist_ok=True); ROOT="/SPECPHYS_ROOT"
        for sub in cfg["subjects"]:
            fr=sorted(os.listdir(os.path.join(ROOT,sub["frames_dir"])))[0]; bgr=cv2.imread(os.path.join(ROOT,sub["frames_dir"],fr)); A,info=amap(bgr,int(sub.get("seed",0))+7)
            f=f"{ad}/subject{sub['id']}.npy"; np.save(f,A); sub.update(spatial="anatomy",amap_file=f,anatomy=info); preview(bgr,A,f[:-4]+"_preview.png")
        cfg["out"]=new_out; json.dump(cfg,open(out_cfg,"w"),indent=1); print("CFG_DONE",len(cfg["subjects"]),out_cfg)
    elif sys.argv[1]=="--batch":
        pairs=json.load(open(sys.argv[2])); od=sys.argv[3]; os.makedirs(od,exist_ok=True); meta={}
        for q in pairs:
            fr=sorted(os.listdir(q["frames"]))[0]; bgr=cv2.imread(os.path.join(q["frames"],fr)); A,info=amap(bgr,5000+q["idx"]); np.save(f"{od}/f{q['idx']:03d}.npy",A); meta[q["idx"]]=info
            if q["idx"]%50==0: preview(bgr,A,f"{od}/f{q['idx']:03d}_preview.png")
        json.dump(meta,open(f"{od}/meta.json","w"),indent=1); print("AMAP_DONE",len(meta))
    else:
        bgr=cv2.imread(sys.argv[1]); A,info=amap(bgr,int(sys.argv[3])); np.save(sys.argv[2],A); print(info, "mean nz",float(A[A>0.02].mean()),"area frac",float((A>0.02).mean()))
        if len(sys.argv)>4: preview(bgr,A,sys.argv[4])
