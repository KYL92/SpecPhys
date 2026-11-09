"""[Virtual-Face patch 35] 도메인 일반화(DG) 기준선 3종 — PhysFormerTrainer 에서 env `RPPG_DG` 로 선택 (40 문서, 체인 55).
  RPPG_DG=gdro  GroupDRO (Sagawa et al. ICLR 2020): 도메인별 위험의 온라인 지수 가중(η=RPPG_DG_ETA, 기본 0.01) → 최악 도메인 가중 손실.
  RPPG_DG=vrex  V-REx (Krueger et al. ICML 2021): 배치 내 도메인별 위험의 평균 + β·분산(β=RPPG_DG_BETA, 기본 1.0).
  RPPG_DG=dann  DANN (Ganin et al. JMLR 2016): transformer3 출력 토큰 평균(64-d) 위 도메인 분류기 + 기울기 반전(λ = RPPG_DG_LAMBDA(0.1)·(2/(1+e^{-10p})−1), p = 진행률).
도메인 라벨: RPPG_DG_DOMAIN_MAP(json {chunk_name: domain_name}, scripts/dg_domain_map.py 가 풀의 심볼릭 링크 출처로 생성). 맵에 없는 이름은 'other'.
DG 가중은 감독 항(a·NegPearson + b·(fre+kl))의 샘플별 값에만 적용하고, 하이브리드·SiNC 등 추가 항은 그대로 더해진다.
epoch 끝에 `DG stats epoch k: {...}` 를 출력한다(체인이 이 줄의 존재로 적용 여부를 검증)."""
import json, math, os
import torch
import torch.nn as nn


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DG:
    def __init__(self, method, model, device, lr, total_iters):
        self.method = method; self.device = device; self.total_iters = total_iters; self.it = 0
        mp = os.environ.get("RPPG_DG_DOMAIN_MAP", "")
        if mp and not os.path.exists(mp): raise FileNotFoundError(f"RPPG_DG_DOMAIN_MAP not found: {mp}")
        self.map = json.load(open(mp)) if mp else {}
        names = sorted(set(self.map.values())) or ["other"]
        if "other" not in names: names.append("other")
        self.dom_idx = {n: i for i, n in enumerate(names)}; self.K = len(names)
        self.eta = float(os.environ.get("RPPG_DG_ETA", "0.01")); self.beta = float(os.environ.get("RPPG_DG_BETA", "1.0"))
        self.lam_max = float(os.environ.get("RPPG_DG_LAMBDA", "0.1"))
        self.q = torch.ones(self.K, device=device) / self.K
        fw = os.environ.get("RPPG_DG_WEIGHTS", "")
        self.fixed_w = {}
        if method == "fixed":
            if not fw: raise ValueError("RPPG_DG=fixed needs RPPG_DG_WEIGHTS json {domain: weight}")
            dw = json.loads(fw); self.fixed_w = {n: float(dw.get(self.map.get(n, "other"), 1.0)) for n in self.map}
        self.feat = None; self.head = None; self.head_opt = None
        if method == "dann":
            m = model.module if hasattr(model, "module") else model
            m.transformer3.register_forward_hook(lambda mod, inp, out: setattr(self, "feat", out[0] if isinstance(out, tuple) else out))
            self.head = nn.Sequential(nn.Linear(m.dim, m.dim), nn.ReLU(), nn.Linear(m.dim, self.K)).to(device)
            self.head_opt = torch.optim.Adam(self.head.parameters(), lr=lr, weight_decay=0.00005)
        self.stats = {"n": {}, "miss": 0, "dom_acc": [0, 0], "pen": []}
        print(f"DG init method={method} K={self.K} domains={names} map_entries={len(self.map)} eta={self.eta} beta={self.beta} lam={self.lam_max}", flush=True)

    def domains(self, names):
        out = []
        for n in names:
            d = self.map.get(str(n))
            if d is None: self.stats["miss"] += 1; d = "other"
            self.stats["n"][d] = self.stats["n"].get(d, 0) + 1
            out.append(self.dom_idx[d])
        return torch.tensor(out, device=self.device)

    def loss(self, per_sample, names):
        """per_sample: (B,) 샘플별 감독 손실. 반환: DG 결합 손실(스칼라)."""
        d = self.domains(names); self.it += 1
        present = torch.unique(d)
        risks = torch.stack([per_sample[d == k].mean() for k in present])
        if self.method == "gdro":
            with torch.no_grad():
                self.q[present] = self.q[present] * torch.exp(self.eta * risks.detach()); self.q = self.q / self.q.sum()
            return (self.q[present] * risks).sum()
        if self.method == "vrex":
            pen = risks.var(unbiased=False) if len(present) > 1 else per_sample.sum() * 0.0
            self.stats["pen"].append(float(pen.detach())); return risks.mean() + self.beta * pen
        if self.method == "fixed":  # [41 문서, 체인 56] 도메인 고정 가중: RPPG_DG_WEIGHTS json {domain: w}(없으면 1), 배치 내 가중 평균
            w = torch.tensor([self.fixed_w.get(n, 1.0) for n in names], device=self.device)
            return (w * per_sample).sum() / w.sum()
        if self.method == "dann":
            p = min(1.0, self.it / max(1, self.total_iters)); lam = self.lam_max * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)
            f = self.feat.mean(1)  # (B, dim)
            logits = self.head(_GRL.apply(f, lam))
            adv = nn.functional.cross_entropy(logits, d)
            self.stats["dom_acc"][0] += int((logits.argmax(1) == d).sum()); self.stats["dom_acc"][1] += len(d); self.stats["pen"].append(float(adv.detach()))
            self.head_opt.zero_grad()
            return per_sample.mean() + adv
        raise ValueError(self.method)

    def step(self):
        if self.head_opt is not None: self.head_opt.step()

    def epoch_report(self, epoch):
        s = self.stats; extra = {}
        if self.method == "gdro": extra["q"] = {n: round(float(self.q[i]), 3) for n, i in self.dom_idx.items()}
        if self.method == "vrex": extra["var_mean"] = round(sum(s["pen"]) / max(1, len(s["pen"])), 4)
        if self.method == "fixed": extra["weights"] = json.loads(os.environ.get("RPPG_DG_WEIGHTS", "{}"))
        if self.method == "dann": extra["dom_acc"] = round(s["dom_acc"][0] / max(1, s["dom_acc"][1]), 3); extra["adv_mean"] = round(sum(s["pen"]) / max(1, len(s["pen"])), 4)
        print(f"DG stats epoch {epoch}: method={self.method} n={s['n']} miss={s['miss']} {extra}", flush=True)
        self.stats = {"n": {}, "miss": 0, "dom_acc": [0, 0], "pen": []}
