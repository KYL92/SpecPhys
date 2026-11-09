"""VIPL 저장 출력에 대한 HR 읽기(readout) 프로토콜 비교 — GPU 불필요.
A 툴박스 정본: cumsum→detrend(100)→butter1 0.6–3.3→periodogram(영상 전체)
B 미분 출력 그대로(적분 없음)→butter→periodogram(영상 전체)
C 문헌형: 10 s 창 3개 각각 A 로 HR → 평균
D 문헌형: 10 s 창 3개 각각 B 로 HR → 평균
"""
import sys, glob, pickle, json, numpy as np
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from agg_model2 import cat_chunks, toolbox_hr, PF
import agg_vipl
from scipy.signal import butter, filtfilt, periodogram
sys.path.insert(0, "/SPECPHYS_ROOT/repos/rPPG-Toolbox")
from evaluation.post_process import _detrend as tb_detrend
FS = 30.0
def detrend(sig, lam=100):
    return np.asarray(tb_detrend(np.asarray(sig, dtype=np.float64), lam)).ravel()
def hr_of(sig, cums, nfft=None):
    x = np.asarray(sig, dtype=np.float64)
    if cums: x = detrend(np.cumsum(x), 100)
    b, a = butter(1, [0.6/(FS/2), 3.3/(FS/2)], btype="bandpass"); x = filtfilt(b, a, x - x.mean())
    N = nfft or int(2**np.ceil(np.log2(len(x))))
    f, p = periodogram(x, fs=FS, nfft=N, detrend=False); m = (f >= 0.6) & (f <= 3.3)
    return 60.0*f[m][np.argmax(p[m])]
def win_hr(sig, cums, w=300):
    n = len(sig)//w
    if n == 0: return hr_of(sig, cums)
    return float(np.mean([hr_of(sig[i*w:(i+1)*w], cums, nfft=4096) for i in range(n)]))
M, F = agg_vipl.meta()
def run(tag_list, name):
    rows = []
    for tag in tag_list:
        pk = sorted(glob.glob(f"{PF}/runs/finetune_{tag}/vipl_u8/saved_test_outputs/*.pickle"))
        d = pickle.load(open(pk[-1], "rb"))
        for vid in d["predictions"]:
            p = np.asarray(cat_chunks(d["predictions"][vid])).ravel(); j = M[str(vid)]
            rows.append(dict(vid=str(vid), scen=j["scenario"][:2], src=j["source"], gt=j["gt_hr_mean"],
                             A=toolbox_hr(p), A2=hr_of(p, True), B=hr_of(p, False), C=win_hr(p, True), D=win_hr(p, False)))
    g = np.array([r["gt"] for r in rows]); sc = np.array([r["scen"] for r in rows]); out = {}
    print(f"\n### {name} n={len(rows)}")
    print(f"{'readout':8s} {'MAE':>6s} {'RMSE':>6s} {'r':>5s} {'bias':>6s} {'>15':>4s} {'GT>=100':>8s} {'v7':>6s} {'v1':>5s} {'60-90':>6s}")
    for k in ["A","A2","B","C","D"]:
        p = np.array([r[k] for r in rows]); e = np.abs(p-g); hi = g>=100; mid=(g>=60)&(g<90)
        out[k] = dict(mae=e.mean(), rmse=np.sqrt((e**2).mean()), r=np.corrcoef(p,g)[0,1], bias=(p-g).mean(), big=int((e>15).sum()), hi=e[hi].mean(), v7=e[sc=="v7"].mean(), v1=e[sc=="v1"].mean(), mid=e[mid].mean())
        o = out[k]; print(f"{k:8s} {o['mae']:6.2f} {o['rmse']:6.2f} {o['r']:5.2f} {o['bias']:+6.2f} {o['big']:4d} {o['hi']:8.1f} {o['v7']:6.1f} {o['v1']:5.1f} {o['mid']:6.2f}")
    return out, rows
res = {}
res["CV_HYB"], rows_h = run([f"VF{k}_HYB_L" for k in range(1,6)], "5-fold HYB (hybrid co-training)")
res["CV_TW"], _ = run([f"VF{k}_TW_L" for k in range(1,6)], "5-fold TW (co-training)")
res["VF1_TWg"], _ = run(["VF1_TWg_L"], "fold1 TWg (device-HR label)")
res["VF1_BASE"], _ = run(["VF1_BASE_L"], "fold1 BASE (VIPL only, collapsed)")
json.dump(dict(summary=res, rows_cv_hyb=rows_h), open(sys.argv[1], "w"), indent=1, default=float)
