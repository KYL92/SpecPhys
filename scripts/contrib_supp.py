#!/usr/bin/env python3
"""[69 문서 §6] 보충 자료 §S9 표와 §S5 검정 행(영문)을 results/contrib_judge.json 에서 생성한다(옮겨 적기 오류 방지).
  python scripts/contrib_supp.py → results/contrib_supp_S9.md, results/contrib_supp_S5rows.md"""
import json
R = "/SPECPHYS_ROOT/_workspace/data/results"; J = json.load(open(f"{R}/contrib_judge.json"))
NM = {"SpecPhys": "SpecPhys (final)", "public FactorizePhys": "public FactorizePhys", "public RhythmFormer": "public RhythmFormer", "public PhysMamba": "public PhysMamba",
      "previous PhysFormer + corpus": "PhysFormer + our corpus", "SpecPhys without synthetic clips": "SpecPhys recipe without generated-face clips", "SpecPhys, clean-mask clips": "SpecPhys recipe, leak-free generated-face clips",
      "quarter real + synthetic (Q)": "quarter of real subjects + generated-face clips", "quarter real only (QN)": "quarter of real subjects only", "synthetic only (SO)": "generated-face clips only"}
def pf(p): return "<.0001" if p < 1e-4 else (f"{p:.4f}".lstrip("0") if p < .001 else f"{p:.3f}".lstrip("0"))
VER = {"지지": "supported", "지지 없음": "not supported", "반대 방향 유의": "significant, reverse"}
S9, S5 = [], []
A = J["A"]; hard = [t for t in A["levels"] if t.startswith("hard")][0]; easy = [t for t in A["levels"] if t.startswith("easy")][0]
S9.append("**Table S9a. Controlled heart-rate sweep (mean absolute error over 40 clips, bpm; share of clips within 5 bpm; within-identity slope).** Four held-out generated identities × ten heart rates (45–180 bpm), scene and noise draws fixed per identity. Weak pulse: modulation 0.003 with illumination drift 0.15, motion 6 px, rotation 1.5° and sensor noise 3; clean pulse: modulation 0.01. Readout: chunk outputs standardised and concatenated, 8× zero-padded FFT peak in 0.7–4.0 Hz; difference-normalised outputs are first integrated per chunk and linearly detrended. Multi-seed models: per-clip error averaged over seeds.\n")
S9.append("| Model | seeds | weak pulse MAE | within 5 bpm | slope | clean pulse MAE | within 5 bpm | slope |"); S9.append("|---|---|---|---|---|---|---|---|")
for k in ["SpecPhys", "public FactorizePhys", "public RhythmFormer", "public PhysMamba", "previous PhysFormer + corpus", "SpecPhys without synthetic clips", "SpecPhys, clean-mask clips", "quarter real + synthetic (Q)", "quarter real only (QN)", "synthetic only (SO)"]:
    h, e = A["levels"][hard].get(k), A["levels"][easy].get(k)
    if not h: continue
    S9.append(f"| {NM[k]} | {h['seeds']} | {h['mae']:.2f} | {h['within5']:.2f} | {h['slope']:.3f} | {e['mae']:.2f} | {e['within5']:.2f} | {e['slope']:.3f} |")
S9.append("\n**Table S9b. Weak-pulse sweep: mean absolute error per injected heart rate (bpm).**\n")
hrs = list(A["levels"][hard]["SpecPhys"]["per_hr"]); S9.append("| Model | " + " | ".join(hrs) + " |"); S9.append("|---|" + "---|" * len(hrs))
for k in ["SpecPhys", "public FactorizePhys", "public RhythmFormer", "public PhysMamba", "previous PhysFormer + corpus", "SpecPhys without synthetic clips"]:
    S9.append(f"| {NM[k]} | " + " | ".join(f"{A['levels'][hard][k]['per_hr'][x]:.1f}" for x in hrs) + " |")
Abp = J["A_bp"]; hb = [t for t in Abp["levels"] if t.startswith("hard")][0]; eb = [t for t in Abp["levels"] if t.startswith("easy")][0]
S9.append(f"\nSensitivity (third-order Butterworth 0.7–4.0 Hz before the FFT, all models): weak pulse SpecPhys {Abp['levels'][hb]['SpecPhys']['mae']:.2f} vs. " + ", ".join(f"{NM[k]} {Abp['levels'][hb][k]['mae']:.2f}" for k in ["public FactorizePhys", "public RhythmFormer", "public PhysMamba", "previous PhysFormer + corpus"]) + f" (all four Holm p ≤ {max(r['holm'] for r in Abp['tests'] if r['set'].startswith('hard')):.4f}); clean pulse SpecPhys {Abp['levels'][eb]['SpecPhys']['mae']:.2f} vs. " + ", ".join(f"{NM[k]} {Abp['levels'][eb][k]['mae']:.2f}" for k in ["public FactorizePhys", "public RhythmFormer", "public PhysMamba", "previous PhysFormer + corpus"]) + ". The filter attenuates 45 bpm, which raises the SpecPhys weak-pulse error at that rate.")
for r in A["tests"]:
    S5.append(f"| controlled sweep ({'weak pulse' if r['set'].startswith('hard') else 'clean pulse'}): SpecPhys vs {NM[r['base']]}, clip MAE | 40 | {r['ref']:.2f} → {r['spec']:.2f} | {r['d']:+.2f} | {pf(r['p'])} | {pf(r['holm'])} | {VER[r['verdict']]} |")
S9.append("\n**Table S9c. Within-subject tracking on real video.** Per subject, predictions and ground truth are centred on the subject's mean; slope b = ΣΔpred·Δgt / ΣΔgt² (1 = full tracking), within-subject correlation and mean absolute within-subject error. VIPL-HR: device heart rate, 2,378 videos of 107 subjects; MMPD: subjects 7–33, common per-video label, 540 videos. Predictions are averaged over seeds before centring.\n")
S9.append("| Model | VIPL-HR slope | r | error (bpm) | MMPD slope | r | error (bpm) |"); S9.append("|---|---|---|---|---|---|---|")
BV, BM = J["B_VIPL-HR"], J["B_MMPD 7-33"]
for k in ["SpecPhys", "public FactorizePhys", "public RhythmFormer", "public PhysMamba", "previous PhysFormer + corpus"]:
    a, b = BV["levels"][k], BM["levels"][k]; S9.append(f"| {NM[k]} | {a['slope']:.3f} | {a['r_within']:.3f} | {a['mae_within']:.2f} | {b['slope']:.3f} | {b['r_within']:.3f} | {b['mae_within']:.2f} |")
S9.append(f"\nPer-seed SpecPhys slopes: VIPL-HR {min(BV['seed_slopes']):.3f}–{max(BV['seed_slopes']):.3f} (all ten above every existing method), MMPD {min(BM['seed_slopes']):.3f}–{max(BM['seed_slopes']):.3f}. On MMPD the exercise label explains 17 % of the within-subject heart-rate variance, and the public RhythmFormer has the higher slope within both exercise strata (0.57 and 0.63 vs. 0.51 and 0.32).")
for tag, B, n in (("VIPL-HR", BV, 107), ("MMPD 7–33", BM, 27)):
    for r in B["tests"]:
        S5.append(f"| within-subject slope, {tag}: SpecPhys vs {NM[r['base']]} (subject-level sign flip) | {n} subjects | {B['levels'][r['base']]['slope']:.3f} → {B['levels']['SpecPhys']['slope']:.3f} | {r['d']:+.3f} | {pf(r['p'])} | {pf(r['holm'])} | {VER[r['verdict']]} |")
C = J["C"]; S9.append("\n**Table S9d. Halved or doubled readings (% of videos; ratio of prediction to ground truth within 2 ± 0.2 or 0.5 ± 0.1; multi-seed models: share of seeds).**\n")
S9.append("| Model | VIPL-HR (doubled / halved) | MMPD 7–33 (doubled / halved) | MR-NIRP still / small / large | MCD-A / MCD-B |"); S9.append("|---|---|---|---|---|")
for k in ["SpecPhys", "public FactorizePhys", "public RhythmFormer", "public PhysMamba", "previous PhysFormer + corpus"]:
    v, m, d = C["levels"]["VIPL-HR"][k], C["levels"]["MMPD 7-33"][k], C["descriptive"][k]
    f_ = lambda x: "–" if x is None else f"{x:.1f}"
    S9.append(f"| {NM[k]} | {v['octave']:.2f} ({v['doubling']:.2f} / {v['halving']:.2f}) | {m['octave']:.2f} ({m['doubling']:.2f} / {m['halving']:.2f}) | {f_(d['MR-NIRP still'])} / {f_(d['MR-NIRP small'])} / {f_(d['MR-NIRP large'])} | {f_(d['MCD-A'])} / {f_(d['MCD-B'])} |")
for r in C["tests"]:
    S5.append(f"| halved/doubled share, {r['set'].replace('MMPD 7-33', 'MMPD 7–33')}: SpecPhys vs {NM[r['base']]} (video-level sign flip) | {C['levels'][r['set']]['SpecPhys']['n']:,} | {C['levels'][r['set']][r['base']]['octave']:.2f} % → {C['levels'][r['set']]['SpecPhys']['octave']:.2f} % | {r['d']:+.2f} pt | {pf(r['p'])} | {pf(r['holm'])} | {VER[r['verdict']]} |")
D = J["D"]; S9.append("\n**Table S9e. Scarce real data (zero-shot MAE, bpm; seeds 2100–2500; same recipe and epochs, all starting from the public FactorizePhys weights).** Quarter of real subjects: ceil(25 %) of the subjects of each real source drawn with a fixed generator before training (UBFC-rPPG 9 of 34 with their noisy and compressed copies, MCD 6 of 24, PURE 2 of 6, MMPD 2 of 6; 960 of 3,125 real chunks). Generated-face clips: the 90 leak-free clips (540 chunks).\n")
S9.append("| Training data | VIPL-HR | MMPD 7–33 | pooled low | pooled high | MR-NIRP small |"); S9.append("|---|---|---|---|---|---|")
LAB = {"all real + synthetic (C)": "all real + generated-face clips", "all real only (NS)": "all real only", "quarter real + synthetic (Q)": "quarter of real subjects + generated-face clips", "quarter real only (QN)": "quarter of real subjects only", "synthetic only (SO)": "generated-face clips only", "public FactorizePhys (start)": "none (public FactorizePhys, the starting point)"}
for k, lab in LAB.items():
    L = D["levels"][k]; S9.append(f"| {lab} | " + " | ".join(f"{L[m]:.2f}" for m in ("P2 VIPL-HR all", "P3 MMPD 7-33 all", "pooled low", "pooled high", "NIRP small")) + " |")
T = D["tests"]; ME = {"P2 VIPL-HR all": "VIPL-HR", "P3 MMPD 7-33 all": "MMPD 7–33", "pooled low": "pooled low rate", "pooled high": "pooled high rate", "NIRP small": "MR-NIRP small"}
mq, ms = T["기전: 어려운 통제 시험 Q vs QN"], T["기전: 어려운 통제 시험 SO vs 공개 FactorizePhys"]
S9.append(f"\nWeak-pulse sweep (descriptive): quarter of real subjects without → with generated-face clips {mq['ref']:.2f} → {mq['cell']:.2f} bpm (clip permutation p = {pf(mq['p'])}); public FactorizePhys → fine-tuned on generated-face clips only {ms['ref']:.2f} → {ms['cell']:.2f} (p = {pf(ms['p'])}).")
for r in T["D1 Q vs QN"]:
    v = "supported" if (r["d"] < 0 and r["holm"] < .05) else ("significant, reverse" if r["holm"] < .05 else "not supported")
    S5.append(f"| scarce real data: quarter of real subjects with vs without generated-face clips, {ME[r['metric']]} | {r['n']:,} | {r['base']:.2f} → {r['cell']:.2f} | {r['d']:+.2f} | {pf(r['perm'])} | {pf(r['holm'])} | {v} |")
for r in T["D2 interaction"]:
    S5.append(f"| scarce real data, interaction (gain with a quarter minus gain with all real subjects), {ME[r['metric']]} | {r['n']:,} | {r['full']:+.2f} (all) vs {r['quarter']:+.2f} (quarter) | {r['d']:+.2f} | {pf(r['p'])} | {pf(r['holm'])} | {VER[r['verdict']]} |")
for r in T["D3 SO vs public FactorizePhys"]:
    v = ("significant, better" if r["d"] < 0 else "significant, worse") if r["holm"] < .05 else "n.s. (two-sided)"
    S5.append(f"| generated-face clips only vs public FactorizePhys (fine-tuning start, two-sided), {ME[r['metric']]} | {r['n']:,} | {r['base']:.2f} → {r['cell']:.2f} | {r['d']:+.2f} | {pf(r['perm'])} | {pf(r['holm'])} | {v} |")
open(f"{R}/contrib_supp_S9.md", "w").write("\n".join(S9) + "\n"); open(f"{R}/contrib_supp_S5rows.md", "w").write("\n".join(S5) + "\n")
sup = sum(("supported" in x.split("|")[-2]) and ("not supported" not in x.split("|")[-2]) for x in S5); adv = sum(("reverse" in x.split("|")[-2]) or ("worse" in x.split("|")[-2]) for x in S5)
print("S5 rows", len(S5), "supported", sup, "adverse", adv)
