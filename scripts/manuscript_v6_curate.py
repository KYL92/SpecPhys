#!/usr/bin/env python3
"""[69 문서 §7] 원고 v6: 선별안(69 §5)대로 본문을 8쪽 구성으로 줄이고, 본문에서 뺀 문단·표·그림은 문장을 바꾸지 않고 보충 §S10 으로 옮긴다(번호·참조만 갱신).
입력: _workspace/58_cvpr_main_v5_full.md·58_cvpr_supp_v5_full.md(v6 이전 동결본, 69 추가분 포함). 출력: _workspace/58_cvpr_main.md(v6), 58_cvpr_supp.md(§S10 추가, 참조 갱신).
  python scripts/manuscript_v6_curate.py"""
import re
W = "/SPECPHYS_ROOT/_workspace"
s = open(f"{W}/58_cvpr_main_v5_full.md").read(); B = s.split("\n\n")
def blk(i, head): assert B[i].lstrip().startswith(head), (i, B[i][:70]); return B[i]
def rep(t, a, b, n=1):
    assert t.count(a) == n, (t.count(a), a[:70]); return t.replace(a, b)
def split_sents(t):
    P = t
    for a in ("vs. ", "i.e. ", "e.g. ", "Sec. ", "Fig. "): P = P.replace(a, a.replace(". ", "\x00 "))
    return [x.replace("\x00", ".") for x in re.split(r"(?<=[.)])\s+(?=[A-Z*])", P)]
def drop(t, idx, starts):
    ss = split_sents(t); out = []
    for i, st in zip(idx, starts): assert ss[i].startswith(st), (i, ss[i][:60])
    return " ".join(x for i, x in enumerate(ss) if i not in idx), [ss[i] for i in idx]
MOVED = []   # (원래 절, 문장)
# ---------- 본문에 남기는 블록(참조 갱신) ----------
b10 = blk(10, "- **SpecPhys, a spectrum-guided")
for a, b_ in (("removes the lock-in to controlled flicker (1.36 vs. 30.3) (Sec. 4.3).", "removes the lock-in to controlled flicker (1.36 vs. 30.3) (Sec. 4.2)."),
              ("significantly worse on no evaluation set (Sec. 4.7).", "significantly worse on no evaluation set (Sec. 4.2)."),
              ("tested under the same rules, do not help (Sec. 4.7).", "tested under the same rules, do not help (Sec. 4.5)."),
              ("each contribute nothing measurable (Sec. 4.5).", "each contribute nothing measurable (Sec. 4.6)."),
              ("rather than lowering that share (Sec. 4.7).", "rather than lowering that share (Sec. 4.5)."),
              ("leaves post-exercise error at 9.7 bpm (Sec. 4.8).", "leaves post-exercise error at 9.7 bpm (Sec. 4.6).")): b10 = rep(b10, a, b_)
b13 = rep(rep(blk(13, "**Attention and factorization"), "does not help on its own (Sec. 4.5)", "does not help on its own (Sec. 4.6)"), "projection on the main path (Sec. 4.7).", "projection on the main path (Sec. 4.5).")
b15 = rep(blk(15, "**Noise references.**"), "and rejected it (Sec. 4.7).", "and rejected it (Sec. 4.5).")
b19 = rep(rep(blk(19, "SpecPhys = FactorizePhys trunk"), "without the additions of Sec. 3.6,", "without the additions of Sec. 3.5,"), "needs a public face parser (Sec. 3.6).", "needs a public face parser (Sec. 3.5).")
b26 = rep(blk(26, "The HR label of a chunk"), "only the output-spectrum terms matter (Sec. 4.5).", "only the output-spectrum terms matter (Sec. 4.6).")
b28, mv = drop(blk(28, "All zero-shot models start"), [1, 2], ["In-domain results use a second stage", "For VIPL-HR the second stage"]); MOVED_INDOMAIN = mv
b40, mv = drop(blk(40, "**Zero-shot evaluation.**"), [3], ["**In-domain evaluation**"]); MOVED_INDOMAIN = mv + MOVED_INDOMAIN
b37 = rep(blk(37, "The layer is differentiable"), "Sec. 4.7 reports how $\\lambda$ was chosen and tested.", "Sec. 4.5 reports how $\\lambda$ was tested and §S10.5 how it was chosen.")
b41 = rep(rep(blk(41, "**Metrics and statistics.**"), "SpecPhys-B in Tables 1–5 uses seeds 100–300.", "SpecPhys-B in Tables 1 and 3 uses seeds 100–300."), "this choice was registered after the second stage of Sec. 4.7.", "this choice was registered after the second stage of the selection (§S10.5).")
b44, mv = drop(blk(44, "FactorizePhys has 52k parameters."), [2], ["The full learning-signal study"]); b44 = "**Implementation.** " + b44; MOVED += [("Sec. 4.1 (implementation)", x) for x in mv]
b46 = rep(blk(46, "Table 1 reports the zero-shot results"), "has the lowest error on walking (Table 2).", "has the lowest error on walking (Table S10a).")
b47 = rep(blk(47, "**Table 1. Zero-shot"), "from the common label of Sec. 4.7)", "from the common label of Table 2)")
b48 = rep(blk(48, "| Model | training data"), "**SpecPhys (ours, final; Sec. 3.6)**", "**SpecPhys (ours, final; Sec. 3.5)**")
gain = ("**Where the gain lies.** Stratified by heart-rate band (Table S10a), the gain of SpecPhys-B over the public FactorizePhys weights lies in the 60–90 bpm band (4.94 vs. 6.30) "
        "and reverses above 120 bpm (24.3 vs. 20.5) and after exercise; the final model recovers both ends on VIPL-HR (<60 bpm 7.88, ≥120 bpm 17.92) at a small cost in the bulk (5.41 vs. 4.94 at 60–90 bpm; §S10.1).")
b79 = rep(blk(79, "**Against existing methods (Table 6, lower part).**"), "**Against existing methods (Table 6, lower part).**", "**Against existing methods (Table 2, lower part).**")
b81 = rep(rep(blk(81, "**Table 6. Final model against existing methods"), "**Table 6. Final model against existing methods", "**Table 2. Final model against existing methods"), "(Tables 1–5 report the three-seed cell)", "(Tables 1 and 3 report the three-seed cell)"); b81 = rep(b81, "SpecPhys-R the paired seeds 2100–2500.", "SpecPhys-R the paired seeds 2100–2500. SpecPhys-R, an earlier configuration, routes each chunk between SpecPhys-L and a variant trained with real MMPD clips replayed at 1.5× speed (§S10.5).")
b54o = blk(54, "**Reading the heart rate across its range (Fig. 3).**"); mv54 = [x for x in split_sents(b54o) if x.startswith("The existing models fail at the ends of the range:")]; assert len(mv54) == 1; MOVED += [("Sec. 4.3", mv54[0])]
b54 = ("Two tests, registered before they were run, ask whether the model reads the pulse frequency rather than a heart-rate prior (Fig. 2). A controlled sweep renders four held-out generated identities at ten heart rates from 45 to 180 bpm with the scene and noise fixed per identity, "
       "with a clean pulse and with a weak pulse (modulation 0.003) under strong illumination drift, motion and sensor noise. With the weak pulse SpecPhys errs by 1.18 bpm on average (each of its ten seeds below 1.6 bpm), against 11.10 for the public FactorizePhys, 34.17 for the public PhysMamba, "
       "36.13 for PhysFormer trained on our corpus and 38.28 for the public RhythmFormer (clip-level permutation, Holm p ≤ .005 for all four); the existing models fail at the ends of the range (Fig. 2a). With the clean pulse SpecPhys (0.52) is better than the public RhythmFormer and PhysMamba (Holm p ≤ .002) "
       "but not significantly better than the public FactorizePhys (2.31) or the corpus PhysFormer (1.47). The advantage comes from the PPG-injected clips: the same recipe trained without them errs by 12.49 bpm with the weak pulse, mostly at 45 and 60 bpm, and the sweep uses the generator of those clips, "
       "which the existing models never saw. On real video, the within-subject slope of predicted on true heart rate (1 = full tracking) is 0.662 for SpecPhys on VIPL-HR, against 0.322–0.356 for the public RhythmFormer and PhysMamba and the corpus PhysFormer (subject-level permutation, Holm p = .0002) "
       "and 0.625 for the public FactorizePhys (n.s.; all ten SpecPhys seeds lie above it); on MMPD the public RhythmFormer follows within-subject changes more steeply (0.570 vs. 0.434, n.s.) with a larger within-subject error (12.42 vs. 8.44 bpm). "
       "Halved or doubled readings on VIPL-HR are fewest for SpecPhys (0.70 % vs. 5.00 % for the public RhythmFormer and 1.93 % for PhysMamba, Holm p ≤ .001; §S9).")
b55 = rep(blk(55, "**Figure 3.** Reading the heart rate"), "**Figure 3.** Reading the heart rate", "**Figure 2.** Reading the heart rate")
b59 = blk(59, "**Does the synthetic part of the pool contribute?**")
i0 = b59.index("Two earlier controls on PhysFormer"); i1 = b59.index("In the final model the synthetic clips therefore"); moved59 = b59[i0:i1].strip(); b59 = b59[:i0] + b59[i1:]
assert moved59.endswith("Holm p < .0001)."), moved59[-40:]
bridge = ("SpecPhys-B is weak at both ends of the heart-rate range (Sec. 4.2, Table S10a). Training alone did not hold both ends with one model (§S10.5); the recipe of Sec. 3.5 keeps the two terms that helped, "
          "the doubling penalty at the low end and the noise-reference exclusion loss at the high end.")
b75 = rep(blk(75, "**Why the structure did not help"), "The trunk explains Sec. 4.5: its temporal", "The trunk explains why the two-slot attention adds nothing on its own (Sec. 4.6): its temporal"); b75 = rep(b75, "and the projection of Sec. 3.6.", "and the projection of Sec. 3.5.")
b76 = blk(76, "**Selection and confirmation (Table 7).**"); k = b76.index("Two tests on fresh seeds followed."); moved76 = b76[:k].strip()
b76k = "**Confirmation (Table 4).** The projection and its strength $\\lambda$ = 0.01 were selected on earlier seeds (§S10.5). " + b76[k:]
b57, mv = drop(blk(57, "The public baselines saw different training data"), [7], ["The pool-trained PhysMamba, in turn"]); MOVED += [("Sec. 4.4", x) for x in mv]
b58, mv = drop(blk(58, "**Does the objective transfer?**"), [3, 5], ["A fixed learning-rate or epoch budget", "A frequency term in the native loss"]); MOVED += [("Sec. 4.4", x) for x in mv]
b77 = rep(rep(blk(77, "**Which part of the structure.**"), "(Table 7, lower part)", "(Table 4, lower part)"), "in line with Sec. 4.5.", "in line with Sec. 4.6.")
b77, mv = drop(b77, [6, 7], ["Two slots here means", "The original attention with projection has"]); MOVED += [("Sec. 4.5", x) for x in mv]
b80 = blk(80, "**What it does not fix.**"); k = b80.index("It has costs against the original attention")
b80k = b80[:k] + "Its costs on single sets and the comparison above 110 bpm with the high-rate-biased public RhythmFormer are given in §S10.5."
b84 = rep(rep(blk(84, "**Table 7. Confirmatory tests"), "**Table 7. Confirmatory tests", "**Table 4. Confirmatory tests"), "Guards in Sec. 4.7 and §S6d.", "Guards in §S6d and §S10.5.")
whatnot = ("**Base-recipe ablation (Table S10b).** Dropping the nuisance-slot supervision and decorrelation, reducing the factorization to rank 1 or restoring the original attention layout differs from SpecPhys-B in no paired test (VIPL-HR p ≥ .43), "
           "whereas removing the spectral losses costs 2.6 bpm on VIPL-HR (p < .0001); trunk, hybrid corpus and output-spectrum losses carry the base recipe (§S10.3). "
           "**Failure taxonomy and label-free separation (Table S10c).** In the weak regimes, lock-in onto a measurable nuisance peak exceeds a shuffled-prediction null by only 9–19 points, and none of four pre-registered label-free separation signals improves the weak regimes; "
           "one of them, a local proxy, worsens MMPD walking (17.11 → 18.57, Holm p = .046; §S10.2, §S10.4). "
           "**Recipes for the high end.** Synthetic high-rate clips, speed augmentation, rate-dependent loss weights and balanced sampling each moved the other end or the bulk in the opposite direction (§S6b, §S10.5). "
           "**Target labels (Table S10d).** Two-stage adaptation of SpecPhys-B with target-domain labels reaches 5.31 bpm on VIPL-HR, 0.90 below the best in-domain recipe, but leaves the post-exercise error at 9.7 bpm (§S10.6).")
# 한계: 문장 단위로 남길 것 선택(문장 원문 그대로)
L95 = blk(95, "**Limitations.**"); P = L95
for a in ("vs. ", "i.e. ", "e.g. ", "Sec. ", "Fig. "): P = P.replace(a, a.replace(". ", "\x00 "))
sents = [x.replace("\x00", ".") for x in re.split(r"(?<=[.)])\s+(?=[A-Z*])", P)]; assert len(sents) == 19 and sents[0].startswith("**Limitations.**"), len(sents)
KEEP = [0, 2, 3, 4, 5, 6, 7, 9, 10, 16, 17]
kept = [sents[i] for i in KEEP]
kept[KEEP.index(2)] = rep(kept[KEEP.index(2)], "is not significant (Sec. 4.7).", "is not significant (Sec. 4.2).")
kept[KEEP.index(16)] = rep(kept[KEEP.index(16)], "(an exploratory analysis, Table 6)", "(an exploratory analysis, Table 2)")
lim = " ".join(kept) + " Further limitations, among them the backbone-specific effect of the spectral objective, the pulse-mask leak of the generator, the exploratory separation designs and the fixed weights of the separation signals, are listed in §S10.8."
note = ("*CVPR main paper draft v6 (2026-11-09): the experiments selected for the eight-page paper stay in the main text (selection in the accompanying plan); every paragraph, table and figure taken out of v5 is reproduced unchanged in supplementary §S10, with updated numbers and cross-references only, "
        "and the full v5 text is kept as a separate snapshot. Final configuration as in v5: SpecPhys is the single model with the noise-subspace projection layer, SpecPhys-B the earlier base recipe. "
        "Structure: 8-page double-blind submission, teaser figure, CVPR section order, limitations inside the conclusion, mandatory LLM Usage Statement before the references, supplementary material split out (58_cvpr_supp.md). "
        "Anonymous submission: no author names, repository paths, internal cell identifiers or share links appear in the text. Every number traces to the generated tables (supplementary §S1).*")
M = [B[0], note, "BUDGET", B[3], B[4], B[5], B[6], B[7], B[8], B[9], b10, B[11], B[12], b13, B[14], b15, B[16], B[17], B[18], b19, B[20], B[21], B[22], B[23], B[24], B[25], b26,
     "### 3.4 Training and readout", b28, rep(blk(32, "### 3.6 Final configuration"), "### 3.6 Final configuration", "### 3.5 Final configuration"), B[33], B[34], B[35], B[36], b37,
     B[38], rep(blk(39, "### 4.1 Datasets and protocol"), "### 4.1 Datasets and protocol", "### 4.1 Datasets, protocol and implementation"), b40, b41, B[42], b44,
     "### 4.2 Comparison with existing methods", b46, b47, b48, B[49], gain, b79, b81, B[82], B[83],
     "### 4.3 Reading the heart rate across its range", b54, b55,
     blk(56, "### 4.4 Is the gain the data or the objective?"), b57, b58, b59, B[60], B[61],
     "### 4.5 Model structure: a projection on the main path", bridge, b75, b76k, b77, b80k, b84, B[85], B[86],
     "### 4.6 What did not help", whatnot,
     blk(93, "## 5. Conclusion"), B[94], lim, B[96], B[97], B[98], B[99], B[100], B[101]]
main = "\n\n".join(M)
STALE = re.compile(r"(Sec\. 4\.[78]|Sec\. 3\.6|Tables? [5-8]\b|Fig\. [34]|Figure [34]|Tables 1–5)")
bad = [m.group(0) for m in STALE.finditer(main)]; assert not bad, bad
t = main.split("\n"); on = False; txt = tab = 0
for l in t:
    if l.startswith("## 1. Introduction"): on = True; continue
    if l.startswith("## References"): on = False
    if not on or l.startswith("#"): continue
    n = len(re.findall(r"\S+", l)); tab += n if l.startswith("|") else 0; txt += 0 if l.startswith("|") else n
nfig = len(re.findall(r"^\*\*Figure \d", main, re.M)); ntab = len(re.findall(r"^\*\*Table \d", main, re.M))
budget = (f"*Budget check (2026-11-09, v6): abstract 463 words; main text {txt:,} words + tables {tab:,} words, {nfig} figures, {ntab} tables (v5 with the contribution evidence: 9,646 + 2,293 words, 4 figures, 8 tables). "
          "References and the LLM statement are outside the page limit. Shortening the abstract to about 250 words would free roughly a quarter page; that is the authors' decision.*")
main = main.replace("BUDGET", budget); open(f"{W}/58_cvpr_main.md", "w").write(main); print("v6 main:", txt, "words +", tab, "table words,", nfig, "figures,", ntab, "tables")
# ---------- 보충: §S10 + 참조 갱신 ----------
def sup(t):  # 옮긴 블록 안의 번호·참조 갱신
    for a, b_ in (("**Table 2. Stratified", "**Table S10a. Stratified"), ("Table 2 stratifies", "Table S10a stratifies"), ("Table 2 leaves", "Table S10a leaves"), ("(Sec. 3.6, Sec. 4.7)", "(main Sec. 3.5, Sec. 4.5)"),
                  ("Table 4 removes", "Table S10b removes"), ("**Table 4. Ablation", "**Table S10b. Ablation"), ("(Sec. 4.7)", "(main Sec. 4.5)"), ("(Fig. 4c;", "(Fig. S10b, panel c;"),
                  ("Table 5 and Fig. 4d report the four additions of Sec. 3.5", "Table S10c and Fig. S10b (panel d) report the four additions of §S10.2"), ("**Table 5. Label-free", "**Table S10c. Label-free"), ("(last row of Table 5)", "(last row of Table S10c)"),
                  ("**Figure 4.**", "**Figure S10b.**"), ("Sec. 4.6 evaluates four pre-registered additions to SpecPhys-B that use no new labels (Fig. 2)", "§S10.4 evaluates four pre-registered additions to SpecPhys-B that use no new labels (Fig. S10a)"),
                  ("**Figure 2.**", "**Figure S10a.**"), ("(Table 7)", "(main Table 4)"), ("Table 8 reports", "Table S10d reports"), ("**Table 8. In-domain", "**Table S10d. In-domain"),
                  ("(Sec. 4.4)", "(main Sec. 4.4)"), ("Table 6)", "main Table 2)"), ("(Sec. 4.5)", "(§S10.3)"),
                  ("against the common label of Sec. 4.7)", "against the common label of main Table 2)"), ("the public RhythmFormer is not significant (main Sec. 4.5).", "the public RhythmFormer is not significant (main Sec. 4.2).")):
        t = t.replace(a, b_)
    return t
S10 = ["## S10. Material moved from the main text (v6)",
       "The eight-page main text (v6) keeps the experiments selected for the paper (Sec. 4.1–4.6). The paragraphs, tables and figures below were part of the main text before; they are reproduced unchanged except for table, figure and section numbers. "
       "Numbering map (before → v6): main Table 2 → Table S10a, Table 4 → Table S10b, Table 5 → Table S10c, Table 8 → Table S10d, Table 6 → main Table 2, Table 7 → main Table 4; Figure 2 → Figure S10a, the results figure (Figure 3, later Figure 4) → Figure S10b, the contribution-evidence figure → main Figure 2; "
       "Sec. 3.5 → §S10.2, Sec. 3.6 → Sec. 3.5, Sec. 4.3 → Sec. 4.2, Sec. 4.5 → §S10.3, Sec. 4.6 → §S10.4, Sec. 4.7 → Sec. 4.5 and §S10.5, Sec. 4.8 → §S10.6.",
       "### S10.1 Where the VIPL-HR gain comes from, and where it stops (from Sec. 4.3)", sup(B[50]), sup(B[51]), B[52], B[53],
       "### S10.2 Label-free separation signals under test (from Sec. 3.5)", sup(B[30]), sup(B[31]),
       "### S10.3 Ablation of the base recipe: the two-slot attention is not the gain (from Sec. 4.5)", sup(B[63]), sup(B[64]), B[65],
       "### S10.4 Analysis: failure taxonomy and label-free separation signals (from Sec. 4.6)", sup(B[67]), sup(B[68]), sup(B[69]), B[70], sup(B[71]),
       "### S10.5 From training recipe to model structure: selection history and decision (from Sec. 4.7)", sup(B[73]), sup(B[74]), sup(moved76), sup(B[78]), sup(B[80]),
       "### S10.6 In-domain adaptation with target labels (from Sec. 4.8)", "Protocol (from Sec. 3.4 and Sec. 4.1): " + " ".join(MOVED_INDOMAIN), sup(B[88]), sup(B[89]), B[90], B[91], B[92],
       "### S10.7 Earlier synthetic-data controls on PhysFormer (from Sec. 4.4)", moved59,
       "### S10.8 Full limitations paragraph (from Sec. 5)", sup(L95),
       "### S10.9 Single sentences moved from the main text", "\n".join(f"- From {src}: {sup(x)}" for src, x in MOVED)]
sp = open(f"{W}/58_cvpr_supp_v5_full.md").read()
for a, b_ in (("main Table 6, S1-T12 and S1-T13 rescore", "main Table 2, S1-T12 and S1-T13 rescore"), ("**Final configuration (Sec. 3.6;", "**Final configuration (Sec. 3.5;"),
              ("## S3. The four label-free separation signals (Sec. 3.5)", "## S3. The four label-free separation signals (§S10.2)"), ("changed no result (Table 5, last row).", "changed no result (Table S10c, last row)."),
              ("The failure taxonomy (Sec. 4.6) showed", "The failure taxonomy (§S10.4) showed"), ("the four existing methods of Table 6 and the share", "the four existing methods of Table 2 and the share"),
              ("(Sec. 4.3, Sec. 4.4, Fig. 3).", "(Sec. 4.3, Sec. 4.4, Fig. 2)."), ("S9 contribution evidence (controlled heart-rate sweep, within-person tracking, halved or doubled readings, scarce real data)", "S9 contribution evidence (controlled heart-rate sweep, within-person tracking, halved or doubled readings, scarce real data), S10 material moved from the main text in v6")):
    sp = rep(sp, a, b_)
assert "## S10." not in sp
sp = sp.rstrip("\n") + "\n\n" + "\n\n".join(S10) + "\n"; open(f"{W}/58_cvpr_supp.md", "w").write(sp)
left = [m.group(0) for m in re.finditer(r"\(Sec\. 4\.[78]\)|Sec\. 3\.6\b", sp[: sp.index('## S10.')])]; print("supp: S10 appended; stale refs before S10:", left)
