"""SCALE300: SDXL-Turbo로 피부톤 층화 가상얼굴 생성 (6 tone × 2 gender × 5 age × 5 seed = 300) + ITA 측정.
사용: python gen_faces_sdxl.py [N_PER_CELL=5]  → faces/scale300/f{idx:03d}.png, faces/scale300/meta.json"""
import sys, os, json, torch, numpy as np, cv2
from diffusers import AutoPipelineForText2Image
ROOT="/SPECPHYS_ROOT"; OUT=f"{ROOT}/_workspace/data/faces/scale300"; os.makedirs(OUT,exist_ok=True)
K=int(sys.argv[1]) if len(sys.argv)>1 else 5
TONES=[("very fair pale","I"),("fair light","II"),("light olive medium","III"),("tan brown","IV"),("dark brown","V"),("very dark deep black","VI")]
GEND=["woman","man"]; AGES=["young adult 22 year old","30 year old","45 year old","60 year old","75 year old elderly"]
pipe=AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo",torch_dtype=torch.float16,variant="fp16").to("cuda")
def ita(img):
    h,w=img.shape[:2]; c=img[h//2-h//8:h//2+h//8, w//2-w//8:w//2+w//8]
    lab=cv2.cvtColor(c,cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.float64); L=lab[:,0]*100/255; b=lab[:,2]-128
    return float(np.degrees(np.arctan2(np.median(L)-50,np.median(b))))
meta=[]; idx=0
for ti,(tone,fitz) in enumerate(TONES):
    for g in GEND:
        for a in AGES:
            for k in range(K):
                seed=100000+idx; gen=torch.Generator("cuda").manual_seed(seed)
                prompt=f"passport style close-up photo of a {a} {g} with {tone} skin, Fitzpatrick type {fitz}, neutral expression, looking at camera, face centered, plain gray background, soft even studio lighting, sharp focus, photorealistic, 85mm"
                im=pipe(prompt=prompt,num_inference_steps=4,guidance_scale=0.0,generator=gen,height=512,width=512).images[0]
                p=f"{OUT}/f{idx:03d}.png"; im.save(p); bgr=cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)
                meta.append(dict(idx=idx,file=p,tone=fitz,gender=g,age=a,seed=seed,ita=ita(bgr))); idx+=1
        print(f"tone {fitz}: {idx} done, ITA median {np.median([m['ita'] for m in meta if m['tone']==fitz]):.1f}",flush=True)
json.dump(meta,open(f"{OUT}/meta.json","w"),indent=1); print("total",idx)
