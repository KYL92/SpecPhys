"""체인 28: 폴드 k(held 피험자 집합 제외)의 MMPD 학습 클립(피험자당 트라이얼 절반: 짝수 트라이얼)을 풀에 링크. 사용: python mmpd_fold_links.py POOL k"""
import os,sys,glob
P,k=sys.argv[1],int(sys.argv[2]); PF="/SPECPHYS_ROOT/_workspace/data/physformer"; FOLDS={1:range(1,12),2:range(12,23),3:range(23,34)}; held=set(FOLDS[k]); n=0
for d in glob.glob(f"{PF}/MMPD_TRAIN/subject7*")+glob.glob(f"{PF}/MMPD_ALL/subject7*"):
    sid=int(os.path.basename(d)[7:]); S=(sid-700000)//100; T=sid%100
    if S in held or T%2: continue
    os.symlink(os.path.realpath(d),f"{PF}/COTRAIN_{P}/subject{sid}"); n+=1
print(f"fold{k} mmpd train clips {n} (held {min(held)}~{max(held)})")
