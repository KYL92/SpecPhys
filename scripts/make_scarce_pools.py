#!/usr/bin/env python3
"""[69 문서, 사전 등록] 실제 학습 영상이 적을 때 합성 클립의 몫: 실제 출처별로 사람의 4분의 1(올림)만 남긴 풀.
사람 = idhr_pool_coupling.py 와 같은 정의(UBFC 1~41, 잡음 사본 309~341·코덱 사본 501~510 은 원본 UBFC 사람으로, MCD n%100, PURE 9SSTT → SS, MMPD 700ABB → A).
출처마다 사람을 정렬한 뒤 np.random.default_rng(0) 으로 ceil(0.25 × 사람 수) 명을 뽑는다(UBFC → MCD → PURE → MMPD 순서, 결과 전 고정).
  --select (Alpha): results/scarce_pool_selection.json 에 뽑힌 사람과 세 갈래의 클립 이름을 쓴다.
     Q  = 뽑힌 사람의 실제 클립 + 깨끗한 합성 클립 90개, QN = 뽑힌 사람의 실제 클립만, SO = 깨끗한 합성 클립 90개만
  --build (pro6000): 깨끗한 마스크 풀 cotrain_allf_hyb4c_u8_pool.csv 를 걸러 cotrain_scarce_{q,qn,so}_u8_pool.csv 를 만든다."""
import csv, json, math, os, re, sys
import numpy as np
ROOT = "/SPECPHYS_ROOT"; PF = f"{ROOT}/_workspace/data/physformer"; R = f"{ROOT}/_workspace/data/results"; L = f"{PF}/cache/DataFileLists"
SEL = f"{R}/scarce_pool_selection.json"
def clip(p): return re.search(r"(subject\d+)_input", p).group(1)
if "--select" in sys.argv:
    src = open(f"{ROOT}/scripts/idhr_pool_coupling.py").read(); src = src[:src.index("from scipy.signal import butter")]
    g = {}; exec(compile(src, "ipc", "exec"), g); ident = g["ident"]
    rows = list(csv.reader(open(f"{L}/cotrain_allf_hyb4_u8_pool.csv")))[1:]
    clips = sorted({clip(r[1]) for r in rows}, key=lambda s: int(s[7:])); kind = {c: ident(int(c[7:])) for c in clips}
    assert sum(k[0] == "synthetic" for k in kind.values()) == 90 and sum(k[0] == "real" for k in kind.values()) == 276
    rng = np.random.default_rng(0); chosen = {}
    for ds in ("ubfc", "mcd", "pure", "mmpd"):
        ppl = sorted({k[1] for k in kind.values() if k[0] == "real" and k[1].startswith(ds + "_")}, key=lambda s: int(s.split("_")[1]))
        k = math.ceil(0.25 * len(ppl)); chosen[ds] = dict(persons=len(ppl), chosen=sorted(rng.choice(ppl, k, replace=False).tolist(), key=lambda s: int(s.split("_")[1])))
    keep = {p for v in chosen.values() for p in v["chosen"]}
    real_q = [c for c in clips if kind[c][0] == "real" and kind[c][1] in keep]; syn = [c for c in clips if kind[c][0] == "synthetic"]
    nch = {}
    for r in rows: nch[clip(r[1])] = nch.get(clip(r[1]), 0) + 1
    out = dict(rule="ceil(0.25 x persons) per real source, rng=np.random.default_rng(0), order ubfc->mcd->pure->mmpd", chosen=chosen,
               arms=dict(Q=real_q + syn, QN=real_q, SO=syn),
               counts={a: dict(clips=len(v), chunks=sum(nch[c] for c in v), real_chunks=sum(nch[c] for c in v if kind[c][0] == "real")) for a, v in dict(Q=real_q + syn, QN=real_q, SO=syn, FULL=clips).items()})
    json.dump(out, open(SEL, "w"), indent=1); print(json.dumps(dict(chosen=chosen, counts=out["counts"]), indent=1))
if "--build" in sys.argv:
    S = json.load(open(SEL)); rows = list(csv.reader(open(f"{L}/cotrain_allf_hyb4c_u8_pool.csv")))
    for a, names in S["arms"].items():
        keep = set(names); out = [rows[0]]; n = 0
        for r in rows[1:]:
            if clip(r[1]) in keep: out.append([str(n), r[1]]); n += 1
        assert n == S["counts"][a]["chunks"], (a, n, S["counts"][a]["chunks"]); assert len({clip(r[1]) for r in out[1:]}) == len(keep)
        if a != "QN": assert sum("hyb4c_syn_u8" in r[1] for r in out[1:]) == 540, a   # 깨끗한 합성 청크
        f = f"{L}/cotrain_scarce_{a.lower()}_u8_pool.csv"; tmp = f + ".tmp"; csv.writer(open(tmp, "w", newline="")).writerows(out); os.replace(tmp, f); print("built", f, n, "chunks", len(keep), "clips")
