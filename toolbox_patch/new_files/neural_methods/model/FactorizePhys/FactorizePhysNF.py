"""[Virtual-Face patch 37, 47·48 문서] NF-Phys — FactorizePhys 트렁크 + rank-2 잡음 인수분해 FSAM(NF-FSAM).

FactorizePhys(NeurIPS 2024)의 FSAM 은 voxel 임베딩을 rank-1 비음수 행렬 인수분해(NMF)로 "지배 성분" 하나로 근사해 어텐션을 만든다.
NF-FSAM 은 시간 기저를 둘로 둔다: 맥파 기저 s_p(초기값 상수) + 교란 기저 s_n(초기값 = 입력 클립 자신의 전역 밝기 trace, 추론 시에도 가용).
  - 행렬 배치: 원 코드의 T_KAB view 는 (B,C,T,H,W) 를 permute 없이 (B,T,C·H·W) 로 보므로 행이 (채널, 8프레임 블록) 쌍이 된다.
    NF-FSAM 은 permute 로 행 = 시간(T) 을 정확히 맞춘다(시간 trace 감독이 가능해야 하므로).
  - 어텐션 마스크는 맥파 슬롯(기저 0)의 rank-1 재구성만으로 만든다(교란 슬롯 제외).
  - 슬롯 시간 프로파일 s_k = M·c_k/‖c_k‖² (미분 가능한 최소제곱 투영) 을 트레이너가 감독한다(스펙트럼 NLL·교란 trace·탈상관, NFLoss.py).
  - 트렁크·헤드·FSAM 1×1 conv 의 모듈 이름을 원본과 같게 두어 공개 체크포인트(UBFC-rPPG_FactorizePhys_FSAM_Res.pth)를 strict=False 로 그대로 적재한다.
환경 변수: RPPG_NF=1(트레이너가 이 모델을 생성), NF_RANK1=1(절제: rank-1·시간 배치만), NF_NO_DIFF=1(절제: 내부 프레임 차분 생략).
[패치 38, 53 문서] nf_aux 에 슬롯 계수 coef (B, A·H·W, R) 와 격자 att_hw=(A,H,W) 를 추가 — 맥파/교란 슬롯의 공간 적재(어텐션 공간 정칙화 손실의 입력).
[패치 55, 62 문서] NF_ARCH=nsg|nsp|snmf: conv_block 출력 v(주 경로)를 그 클립의 잡음 참조로 시간축 전역 조절 — 학습·추론 모두 적용(RPPG_NF=1 과 함께).
  nsg  잡음 스펙트럼 안내 게이트(NF_NSG_REFS=0 이면 참조 입력 없음), nsp 잡음 부분공간 능선 투영, snmf 지도 스펙트럼 NMF + Wiener 층. 트레이너가 [nf-arch] 와 NSG stats 를 찍는다.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys, BVP_Head, nf
from neural_methods.model.FactorizePhys.FSAM import ConvBNReLU


def motion_similarity(f):
    """[패치 55b, 62 §8] 전역 유사 변환(이동 tx·ty, 배율 s, 회전 r) LK 최소제곱 해 (B,4,L−1). s·r 은 프레임 가장자리 기준 px/frame 로 환산. 보행·회전의 원근 변화까지 잡는다."""
    g = f.mean(1); B, L, H, W = g.shape
    Ix = 0.5 * (g[:, :-1, 1:-1, 2:] - g[:, :-1, 1:-1, :-2]); Iy = 0.5 * (g[:, :-1, 2:, 1:-1] - g[:, :-1, :-2, 1:-1]); It = (g[:, 1:] - g[:, :-1])[:, :, 1:-1, 1:-1]
    ys = torch.linspace(-1, 1, H - 2, device=f.device, dtype=f.dtype); xs = torch.linspace(-1, 1, W - 2, device=f.device, dtype=f.dtype); yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    A = torch.stack([Ix, Iy, xx * Ix + yy * Iy, xx * Iy - yy * Ix], dim=-1).reshape(B, L - 1, -1, 4)   # (B,T,P,4)
    ATA = A.transpose(2, 3).matmul(A) + 1e-3 * torch.eye(4, device=f.device, dtype=f.dtype)
    ATb = A.transpose(2, 3).matmul(-It.reshape(B, L - 1, -1, 1))
    return torch.linalg.solve(ATA, ATb)[..., 0].permute(0, 2, 1)                                     # (B,4,T)


def noise_refs(x):
    """[패치 55] 프레임 x (B,3,L,H,W) 0~255 → 추론 때도 가용한 잡음 참조 (B,8,L−1): 측면 띠 R·G·B, 전역 R·G·B 차분 정규화 trace, LK 전역 이동 vx, vy.
    측면 띠(좌우 바깥 1/9)의 피부 확률은 0.4~2.3 %(62 문서 §3)이므로 파서 없이 맥박이 거의 없는 조명 참조가 된다."""
    from neural_methods.loss.NFLoss import motion_traces
    B, C, L, H, W = x.shape
    f = x[:, :3].float(); m = max(1, W // 9)
    side = torch.cat([f[..., :m], f[..., -m:]], dim=-1).mean((-1, -2))   # (B,3,L)
    glob = f.mean((-1, -2))                                              # (B,3,L)
    dn = lambda v: (v[..., 1:] - v[..., :-1]) / (v[..., 1:] + v[..., :-1] + 1e-6)
    if os.environ.get("NF_NSP_AFF", "0") == "1":   # [패치 55b] 이동 2 → 유사 변환 4 (nsp 전용; nsg 는 8 참조 고정)
        return torch.cat([dn(side), dn(glob), motion_similarity(f)], dim=1).detach()
    vx, vy = motion_traces(f)
    return torch.cat([dn(side), dn(glob), vx.unsqueeze(1), vy.unsqueeze(1)], dim=1).detach()


def ref_logamp(refs):
    """참조 (B,8,T) → 로그 진폭 스펙트럼 (B,8,F): 차분 정규화 trace 는 τ=1e-4, 이동 속도는 τ=0.01 px/frame 기준(진폭 정보를 남긴다)."""
    T = refs.shape[-1]; A = torch.fft.rfft(refs - refs.mean(-1, keepdim=True), dim=-1).abs() / T
    tau = torch.tensor([1e-4] * 6 + [1e-2] * 2, device=refs.device, dtype=refs.dtype)[None, :, None]
    return torch.log1p(A / tau)


class NS_Gate(nn.Module):
    """[패치 55] 잡음 스펙트럼 안내 게이트: 채널·주파수별 G(f) ∈ (0,1) 을 작은 1D conv(채널 간 공유)로 예측해 v 의 시간 rFFT 에 곱한다. 시작 시 G≈0.98 (항등에 가까움)."""

    def __init__(self, use_refs=True, hid=8):
        super().__init__()
        self.use_refs = bool(use_refs)
        self.net = nn.Sequential(nn.Conv1d((8 if use_refs else 0) + 2, hid, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv1d(hid, hid, 3, padding=1), nn.ReLU(inplace=True), nn.Conv1d(hid, 1, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.scale = float(os.environ.get("NF_NSG_SCALE", "20"))

    def forward(self, v, refs):
        B, C, T, H, W = v.shape
        V = torch.fft.rfft(v, dim=2); Fn = V.shape[2]
        mag = V.abs().mean((-1, -2)).detach()                                # (B,C,F)
        ms = torch.log1p(mag / (mag[:, :, 1:].mean(-1, keepdim=True) + 1e-6)).reshape(B * C, 1, Fn)
        pos = (torch.arange(Fn, device=v.device, dtype=v.dtype) / max(Fn - 1, 1))[None, None, :].expand(B * C, 1, Fn)
        parts = [ms, pos]
        if self.use_refs:
            parts.append(ref_logamp(refs).unsqueeze(1).expand(B, C, 8, Fn).reshape(B * C, 8, Fn))
        d = self.scale * self.net(torch.cat(parts, dim=1)).reshape(B, C, Fn)   # 배율: lr 1e-4 로도 게이트가 닫힐 수 있게(배율 없이는 10 에폭 동안 G≈0.98 고정, 첫 기동에서 확인)
        G = torch.exp(-F.softplus(d - 4.0))
        G = torch.cat([torch.ones_like(G[:, :, :1]), G[:, :, 1:]], dim=2)    # DC 는 건드리지 않는다
        return torch.fft.irfft(V * G[:, :, :, None, None], n=T, dim=2), G


def _hilbert(R):
    """시간축 힐베르트 변환(직교 성분). 합성곱 특징의 잡음 위상은 참조와 어긋나므로 동상·직교 성분을 함께 쓴다."""
    X = torch.fft.rfft(R, dim=-1); X = X * (-1j); X[..., 0] = 0
    return torch.fft.irfft(X, n=R.shape[-1], dim=-1)


def side_pcs(v, k):
    """[패치 55b] 특징 수준 잡음 참조: v 의 좌우 가장자리 열 특징 trace 의 상위 k 시간 주성분 (B,k,T), 설명 분산 비의 제곱근으로 가중(공통 잡음이 뚜렷할수록 큼), 노름 √T 기준. 기울기 없음."""
    B, C, T, H, W = v.shape
    r = torch.cat([v[..., 0], v[..., -1]], dim=-1).permute(0, 1, 3, 2).reshape(B, C * 2 * H, T).detach(); r = r - r.mean(-1, keepdim=True)
    e, U = torch.linalg.eigh(r.transpose(1, 2).bmm(r))                                       # 오름차순
    frac = (e[:, -k:].clamp(min=0.0) / (e.clamp(min=0.0).sum(-1, keepdim=True) + 1e-8)).sqrt()
    return U[:, :, -k:].transpose(1, 2) * (T ** 0.5) * frac[:, :, None]


def nsp_component(v, refs, lam=None):
    """[패치 55] 잡음 부분공간 능선 투영(파라미터 없음)의 제거 성분: 측면 R·G·B + 움직임 참조의 동상·직교 성분(+ NF_NSP_FK>0 이면 특징 수준 주성분)을 v 의 시간축에서 회귀. 참조 진폭으로 소프트 게이트 — 움직임·조명 변화가 작으면 제거하지 않는다
    (정지 영상의 참 맥박은 LK 이동과 결맞으므로, 61 §8.4)."""
    lam = float(os.environ.get("NF_NSP_LAM", "0.1")) if lam is None else lam
    B, C, T, H, W = v.shape
    R = torch.cat([refs[:, :3], refs[:, 6:]], dim=1).to(v.dtype); R = R - R.mean(-1, keepdim=True); Km = R.shape[1] - 3
    amp = R.std(-1, keepdim=True); a0 = torch.tensor([5e-4] * 3 + [5e-2] * Km, device=v.device, dtype=v.dtype)[None, :, None]
    Rz = R / (amp + 1e-8) * (amp / (amp + a0))
    fk = int(os.environ.get("NF_NSP_FK", "0"))
    if fk > 0: Rz = torch.cat([Rz, side_pcs(v, fk).to(v.dtype)], dim=1)
    Rz = torch.cat([Rz, _hilbert(Rz)], dim=1)                                                 # (B,2K,T)
    A = Rz.bmm(Rz.transpose(1, 2)) + lam * T * torch.eye(Rz.shape[1], device=v.device, dtype=v.dtype)[None]
    x = v.permute(0, 1, 3, 4, 2).reshape(B, C * H * W, T); xz = x - x.mean(-1, keepdim=True)   # 행 = 원소, 열 = 시간
    beta = torch.linalg.solve(A, Rz.bmm(xz.transpose(1, 2))).transpose(1, 2)                  # (B,N,2K)
    return beta.bmm(Rz).reshape(B, C, H, W, T).permute(0, 1, 4, 2, 3)


def nsp_project(v, refs, lam=None):
    return v - nsp_component(v, refs, lam)


def snmf_gate(v, refs, steps=6, eps=1e-6):
    """[패치 55] 지도 스펙트럼 NMF + Wiener 층(파라미터 없음): M = |V(f)| (행 = 주파수, DC 제외) 를 맥박 기저 1개 + 잡음 사전 K개로 분해하고 W = 맥박 재구성 / (맥박 + 잡음 재구성) 을 V 에 곱한다.
    잡음 사전 = 측면·이동 진폭 스펙트럼의 봉우리(국소 기저선 초과분)만, 진폭 게이트 w_k 를 곱해 고정. 맥박 기저 초기값 = M 평균 × (1 − 잡음 봉우리) — 곱셈 갱신은 0 을 되살리지 않으므로 맥박 기저는 참조 봉우리를 차지하지 못한다.
    참조에 뚜렷한 봉우리가 없거나 진폭이 작으면 W≈1 (항등). 마스크로는 기울기를 흘리지 않는다."""
    B, C, T, H, W_ = v.shape
    V = torch.fft.rfft(v, dim=2); Fn = V.shape[2] - 1; N = C * H * W_
    with torch.no_grad():
        Mg = V[:, :, 1:].abs().permute(0, 2, 1, 3, 4).reshape(B, Fn, N)                       # (B,F,N) ≥ 0
        R = torch.cat([refs[:, :3], refs[:, 6:]], dim=1).to(v.dtype); R = R - R.mean(-1, keepdim=True)
        Dr = torch.fft.rfft(R, dim=-1).abs()[:, :, 1:]                                        # (B,K,F)
        base = F.avg_pool1d(Dr, 9, stride=1, padding=4, count_include_pad=False)
        D = F.relu(Dr - 1.5 * base)                                                           # 봉우리만
        D = (D / (D.max(-1, keepdim=True)[0] + eps)).permute(0, 2, 1)                         # (B,F,K), 사전마다 최대 1
        amp = R.std(-1); a0 = torch.tensor([5e-4] * 3 + [5e-2] * 2, device=v.device, dtype=v.dtype)[None]
        wk = amp / (amp + a0)                                                                 # (B,K)
        Dw = D * wk[:, None, :]                                                               # 진폭 게이트 적용 사전
        peak = Dw.max(-1, keepdim=True)[0]                                                    # (B,F,1)
        bp = Mg.mean(-1, keepdim=True) * (1.0 - peak).clamp(min=0.0) + eps
        bp = bp / (bp.norm(dim=1, keepdim=True) + eps)
        Dn = Dw / (Dw.norm(dim=1, keepdim=True) + eps)
        bases = torch.cat([bp, Dn], dim=-1)                                                   # (B,F,1+K)
        coef = F.softmax(torch.bmm(Mg.transpose(1, 2), bases), dim=-1)                        # (B,N,1+K)
        for _ in range(steps):
            coef = coef * torch.bmm(Mg.transpose(1, 2), bases) / (coef.bmm(bases.transpose(1, 2).bmm(bases)) + eps)
            num = torch.bmm(Mg, coef); den = bases.bmm(coef.transpose(1, 2).bmm(coef))
            bases = torch.cat([bases[:, :, :1] * num[:, :, :1] / (den[:, :, :1] + eps), bases[:, :, 1:]], dim=-1)   # 맥박 기저만 갱신
        rp = torch.bmm(bases[:, :, :1], coef[:, :, :1].transpose(1, 2))
        rn = torch.bmm(bases[:, :, 1:], coef[:, :, 1:].transpose(1, 2))
        Wm = (rp + eps) / (rp + rn + eps)                                                     # (B,F,N)
        G = torch.ones(B, Fn + 1, N, device=v.device, dtype=v.dtype); G[:, 1:] = Wm
        gate = G.reshape(B, Fn + 1, C, H, W_).permute(0, 2, 1, 3, 4)                          # (B,C,F+1,H,W)
    out = torch.fft.irfft(V * gate, n=T, dim=2)
    return out, gate.mean((-1, -2))                                                           # (B,C,F+1)


class BgRef(nn.Module):
    """[Virtual-Face patch 57, 66 문서] 입력 단계 배경 기준 조명 보정(학습된 화소별 게이트). 안정 배경(부위 분할 배경 확률 ≥ 0.875, 크롭의 ≥ 3 %)의
    채널별 로그 밝기 변화 ρ_c(t) 를 화소·채널별 적재 a_c(x) 만큼 빼서 x̃ = x·exp(−a·ρ) — 얼굴과 배경이 공유하는 조명 변화를 같은 주파수에서도 지운다(a = 1 이면 패치 51 의 나눗셈).
    a 는 화소의 부위 확률·시간 평균 색과 청크 단위 신뢰도 7개(채널별 피부–배경 1 s 고역 결합 3, 배경 로그 밝기, 배경 면적, 배경·피부 고역 변동 크기)를 입력으로 1×1 합성곱 두 층이 정한다.
    화소 자신의 시간 신호에 a 를 맞추지 않으므로 맥박을 흡수하지 않는다. 마지막 층 0 초기화 → 시작 시 항등. 입력의 피부 가중(RPPG_SKINW)은 통계 계산 때만 되돌린다."""

    def __init__(self, hid=16):
        super().__init__()
        self.f1 = nn.Conv2d(3 + 3 + 7, hid, 1)
        self.f2 = nn.Conv2d(hid, 3, 1)
        nn.init.zeros_(self.f2.weight); nn.init.zeros_(self.f2.bias)
        self.last_a = self.last_ok = self.last_P = None

    def forward(self, x, P):                       # x (B,3,L,H,W) 0..255 (피부 가중된 원 프레임), P (B,3,H,W) 부위 확률 [피부, 배경, 비피부]
        B, C, L, H, W = x.shape
        with torch.no_grad():
            skw = os.environ.get("RPPG_SKINW", "")
            w = (float(skw) + (1.0 - float(skw)) * P[:, 0]) if skw else torch.ones_like(P[:, 0])
            xr = x / w.clamp(min=1e-3)[:, None, None]                                   # 피부 가중을 되돌린 프레임
            bg = (P[:, 1] >= 0.875).to(x.dtype); area = bg.mean((1, 2)); ok = (area >= 0.03).to(x.dtype)
            b = (xr * bg[:, None, None]).sum((-1, -2)) / bg.sum((1, 2)).clamp(min=1.0)[:, None, None]   # (B,3,L) 배경 밝기
            lb = torch.log(b.clamp(min=1.0)); rho = (lb - lb.mean(-1, keepdim=True)) * ok[:, None, None]
            sk = (P[:, 0] >= 0.5).to(x.dtype)
            sm = (xr * sk[:, None, None]).sum((-1, -2)) / sk.sum((1, 2)).clamp(min=1.0)[:, None, None]
            ls = torch.log(sm.clamp(min=1.0)); ls = ls - ls.mean(-1, keepdim=True)
            hp = lambda v: v - F.avg_pool1d(v, 31, stride=1, padding=15, count_include_pad=False)[..., :v.shape[-1]]
            hb, hs = hp(lb), hp(ls)
            kap = (hb * hs).sum(-1) / (hb.norm(dim=-1) * hs.norm(dim=-1) + 1e-8)                        # (B,3) 피부–배경 결합
            feats = torch.cat([kap, (torch.log(b.mean((1, 2)).clamp(min=1.0)) / 5.0)[:, None], area[:, None],
                               (torch.log(hb.std(-1).mean(-1) + 1e-5) / 10.0)[:, None], (torch.log(hs.std(-1).mean(-1) + 1e-5) / 10.0)[:, None]], dim=1)
            mrgb = torch.log(xr.mean(2).clamp(min=1.0)) / 5.0                                               # (B,3,H,W) 화소 시간 평균 색
            inp = torch.cat([P, mrgb, feats[:, :, None, None].expand(B, feats.shape[1], H, W)], dim=1)
        if os.environ.get("NF_BGREF_MODE", "learn") == "gate":   # [패치 57b, 66 문서 §2.2] 물리 규칙 게이트(학습 없음): 배경이 피부와 조명 변화를 실제로 공유할 때만 채널별로 a = 1(배경으로 나눔)
            with torch.no_grad():
                q = hb.std(-1) / (hs.std(-1) + 1e-8)                                                    # (B,3) 배경/피부 고역 변동 비
                bright = (b.mean(-1)).clamp(min=0.0)                                                     # (B,3) 피부 가중을 되돌린 배경 밝기
                gate = ((kap >= float(os.environ.get("NF_BGREF_KAPPA", "0.6"))) & (q >= float(os.environ.get("NF_BGREF_QLO", "0.5"))) & (q <= float(os.environ.get("NF_BGREF_QHI", "2.0")))
                        & (bright >= float(os.environ.get("NF_BGREF_MINB", "25"))) & (ok[:, None] > 0)).to(x.dtype)
                a = gate[:, :, None, None].expand(B, 3, H, W).contiguous()
                self.last_gate, self.last_kap, self.last_q = gate, kap, q
        else:
            a = 2.0 * torch.tanh(0.5 * self.f2(F.gelu(self.f1(inp)))) * ok[:, None, None, None]         # (B,3,H,W)
        self.last_a, self.last_ok, self.last_P = a, ok, P
        return x * torch.exp(-a[:, :, None] * rho[..., None, None])


class LearnedNoiseSub(nn.Module):
    """[Virtual-Face patch 56, 65 문서] 학습된 잡음 빼기: 주 경로 v (B,C,T,H,W) 와 잡음 참조 8개(측면 R·G·B, 전역 R·G·B, LK vx·vy)를 보고
    잡음 성분 s (v 와 같은 모양, 시간 평균 0)를 예측해 v 에서 뺀다. 공간 3×3·시간 5 합성곱 두 층이라 위치·색 채널 관계로 같은 주파수의 잡음도 가를 수 있다
    (시간축 능선 투영은 참조와 같은 주파수의 성분을 통째로 지운다). 마지막 층 0 초기화 → 시작 시 항등. 입력은 표본·채널별 표준편차로 정규화하고 출력은 같은 척도로 되돌린다.
    참조는 투영 층과 같은 진폭 게이트(색 5e-4, 움직임 0.05 px/frame)로 약하면 꺼진다. 감독(NF_LSUB_SUP=1)은 트레이너가 이중 렌더 잡음 영상의 특징으로 준다."""

    def __init__(self, C, R=8, hid=16, kt=5):
        super().__init__()
        self.c1 = nn.Conv3d(C + R, hid, (kt, 3, 3), padding=(kt // 2, 1, 1))
        self.c2 = nn.Conv3d(hid, C, (kt, 3, 3), padding=(kt // 2, 1, 1))
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, v, refs):
        B, C, T, H, W = v.shape
        mu = v.mean(dim=(2, 3, 4), keepdim=True); sd = v.std(dim=(2, 3, 4), keepdim=True).detach() + 1e-6
        R = refs.to(v.dtype); R = R - R.mean(-1, keepdim=True); amp = R.std(-1, keepdim=True)
        a0 = torch.tensor([5e-4] * 6 + [5e-2] * (R.shape[1] - 6), device=v.device, dtype=v.dtype)[None, :, None]
        Rz = (R / (amp + 1e-8) * (amp / (amp + a0)))[..., None, None].expand(B, R.shape[1], T, H, W)
        s = self.c2(F.gelu(self.c1(torch.cat([(v - mu) / sd, Rz], dim=1)))) * sd
        return s - s.mean(dim=2, keepdim=True)


class NF_FSAM(nn.Module):
    """rank-2 NMF over the temporal layout (rows = time, cols = channel×H×W)."""

    def __init__(self, inC, align_C, steps=4, rank1=False):
        super().__init__()
        self.pre_conv_block = nn.Sequential(nn.Conv3d(inC, align_C, (1, 1, 1)), nn.ReLU(inplace=True))
        self.post_conv_block = nn.Sequential(ConvBNReLU(align_C, align_C, dim="3D", kernel_size=1),
                                             nn.Conv3d(align_C, inC, 1, bias=False))
        self.steps = int(steps)
        self.rank1 = bool(rank1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / N))

    @staticmethod
    def _mu_step(M, bases, coef):
        # M (B,T,N), bases (B,T,R), coef (B,N,R): NMF multiplicative updates (same as FSAM.NMF.local_step)
        num = torch.bmm(M.transpose(1, 2), bases)
        den = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * num / (den + 1e-6)
        num = torch.bmm(M, coef)
        den = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * num / (den + 1e-6)
        return bases, coef

    def forward(self, x, g):
        """x: (B,C,T,H,W) non-negative-shifted embeddings; g: (B,T) nuisance init trace (any sign)."""
        z = self.pre_conv_block(x)                                   # (B,A,T,H,W) >= 0
        B, A, T, H, W = z.shape
        M = z.permute(0, 2, 1, 3, 4).reshape(B, T, A * H * W)        # rows = time
        bp = torch.ones(B, T, 1, device=z.device, dtype=z.dtype) / math.sqrt(T)
        if self.rank1:
            bases = bp
        else:
            gn = g.to(z.dtype) - g.to(z.dtype).min(dim=1, keepdim=True)[0] + 1e-3
            bn = F.normalize(gn, dim=1).unsqueeze(-1)                # non-negative, unit norm
            bases = torch.cat([bp, bn], dim=-1)                      # (B,T,2): slot 0 = pulse, slot 1 = nuisance
        with torch.no_grad():
            coef = F.softmax(torch.bmm(M.transpose(1, 2), bases), dim=-1)   # (B,N,R)
            for _ in range(self.steps):
                bases, coef = self._mu_step(M, bases, coef)
        # differentiable one-step coefficient update (as FSAM.NMF.compute_coef)
        num = torch.bmm(M.transpose(1, 2), bases)
        den = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * num / (den + 1e-6)
        recon = torch.bmm(bases, coef.transpose(1, 2))               # (B,T,N) full-rank reconstruction
        dist = torch.dist(M, recon)
        cn = coef / (coef.pow(2).sum(dim=1, keepdim=True) + 1e-6)    # (B,N,R)
        S = torch.bmm(M, cn)                                         # (B,T,R) slot temporal profiles
        s_p = S[:, :, 0]
        s_n = None if self.rank1 else S[:, :, 1]
        att = torch.bmm(bases[:, :, :1], coef[:, :, :1].transpose(1, 2))     # pulse slot only
        att = att.reshape(B, T, A, H, W).permute(0, 2, 1, 3, 4).contiguous()
        att = self.post_conv_block(att)
        self.last_hw = (A, H, W)
        return att, dist, s_p, s_n, bases, coef


class NF_BVP_Head(BVP_Head):
    """BVP_Head with NF-FSAM; conv_block / fsam_norm / bias1 / final_layer keep the original names (checkpoint reuse)."""

    def __init__(self, md_config, device, dropout_rate=0.1, debug=False, rank1=False):
        cfg = dict(md_config)
        cfg["MD_FSAM"] = True
        super().__init__(cfg, device, dropout_rate=dropout_rate, debug=debug)
        self.fsam = NF_FSAM(nf[2], cfg["align_channels"], steps=cfg["MD_STEPS"], rank1=rank1)
        self.md_res = cfg["MD_RESIDUAL"]
        self.arch = os.environ.get("NF_ARCH", "")      # [패치 55, 62 문서] 주 경로 v 의 잡음 스펙트럼 조절
        self.last_gate = None
        if self.arch == "nsg":
            self.nsg = NS_Gate(use_refs=os.environ.get("NF_NSG_REFS", "1") == "1")
        elif self.arch not in ("", "nsp", "snmf"):
            raise ValueError(f"unknown NF_ARCH {self.arch}")
        self.nsp_a = nn.Parameter(torch.full((nf[2],), 2.0)) if (self.arch == "nsp" and os.environ.get("NF_NSP_LEARN", "0") == "1") else None   # [패치 55b] 채널별 제거 강도(시작 시그모이드 0.88)
        self.lsub = LearnedNoiseSub(nf[2], R=8, hid=int(os.environ.get("NF_LSUB_HID", "16"))) if os.environ.get("NF_LSUB", "0") == "1" else None   # [패치 56, 65 문서] 학습된 잡음 빼기
        self.last_v = self.last_s = self.last_refs = None

    def forward(self, voxel_embeddings, g, length, refs=None):
        v = self.conv_block(voxel_embeddings)
        if self.lsub is not None:   # [패치 56] 예측한 잡음 성분을 주 경로에서 뺀 뒤 기존 투영을 적용
            if refs is None: raise ValueError("NF_LSUB=1 needs noise references (set NF_ARCH=nsp)")
            s = self.lsub(v, refs); self.last_v, self.last_s, self.last_refs = v, s, refs; v = v - s
        if self.arch == "nsg":
            v, self.last_gate = self.nsg(v, refs)
        elif self.arch == "nsp":
            v = nsp_project(v, refs) if self.nsp_a is None else v - torch.sigmoid(self.nsp_a)[None, :, None, None, None] * nsp_component(v, refs)
        elif self.arch == "snmf":
            v, self.last_gate = snmf_gate(v, refs)
        att, appx_error, s_p, s_n, bases, coef = self.fsam(v - v.min(), g)
        x = torch.mul(v - v.min() + self.bias1, att - att.min() + self.bias1)
        fe = self.fsam_norm(x)
        if self.md_res:
            fe = v + fe
        out = self.final_layer(fe).view(-1, length)
        return out, fe, appx_error, s_p, s_n, bases, coef


class ORIG_Proj_Head(BVP_Head):
    """[패치 55c, 62 §14] 원본 FactorizePhys 어텐션(rank-1, T_KAB) + 주 경로 잡음 부분공간 투영 — 두 슬롯 구조와 투영의 2×2 아블레이션용."""

    def forward(self, voxel_embeddings, batch, length, refs=None):
        v = nsp_project(self.conv_block(voxel_embeddings), refs)
        att, appx_error = self.fsam(v - v.min())
        x = torch.mul(v - v.min() + self.bias1, att - att.min() + self.bias1)
        fe = self.fsam_norm(x)
        if self.md_res: fe = v + fe
        return self.final_layer(fe).view(-1, length), fe, appx_error


class FactorizePhysNF(FactorizePhys):
    def __init__(self, frames, md_config, in_channels=3, dropout=0.1, device=torch.device("cpu"), debug=False):
        super().__init__(frames, md_config, in_channels=in_channels, dropout=dropout, device=device, debug=debug)
        assert self.in_channels == 3, "NF-Phys supports RGB input only"
        self.use_fsam = True
        self.md_infer = True                      # NF-FSAM is part of the forward path at train and test time
        self.rank1 = os.environ.get("NF_RANK1", "0") == "1"
        self.no_diff = os.environ.get("NF_NO_DIFF", "0") == "1"
        self.orig_layout = os.environ.get("NF_ORIGLAYOUT", "0") == "1"   # [10-28 절제] 원본 FactorizePhys FSAM(rank-1, 원 T_KAB view) 유지 + 스펙트럼 손실만
        if self.orig_layout:
            cfg = dict(md_config); cfg["MD_FSAM"] = True; cfg["MD_INFERENCE"] = True
            self.rppg_head = (ORIG_Proj_Head if os.environ.get("NF_ARCH", "") == "nsp" else BVP_Head)(cfg, device=device, dropout_rate=dropout, debug=debug)
        else:
            self.rppg_head = NF_BVP_Head(md_config, device=device, dropout_rate=dropout, debug=debug, rank1=self.rank1)
        self.nf_aux = None
        self.arch = os.environ.get("NF_ARCH", "")
        self.bgref = BgRef(hid=int(os.environ.get("NF_BGREF_HID", "16"))) if os.environ.get("NF_BGREF", "0") == "1" else None   # [패치 57] 입력 단계 배경 기준 보정
        if self.bgref is not None:
            print(f"[bgref] module active: mode={os.environ.get('NF_BGREF_MODE', 'learn')} params={sum(p.numel() for p in self.bgref.parameters())} kappa={os.environ.get('NF_BGREF_KAPPA', '0.6')} q=[{os.environ.get('NF_BGREF_QLO', '0.5')},{os.environ.get('NF_BGREF_QHI', '2.0')}] minb={os.environ.get('NF_BGREF_MINB', '25')} (needs RPPG_PARSE_CH=1 input)", flush=True)
        if getattr(self.rppg_head, "lsub", None) is not None:
            print(f"[lsub] module active: params={sum(p.numel() for p in self.rppg_head.lsub.parameters())} total={sum(p.numel() for p in self.parameters())}", flush=True)
        if self.arch:
            print(f"[nf-arch] arch={self.arch} refs={os.environ.get('NF_NSG_REFS', '1')} lam={os.environ.get('NF_NSP_LAM', '0.1')} aff={os.environ.get('NF_NSP_AFF', '0')} fk={os.environ.get('NF_NSP_FK', '0')} learn={os.environ.get('NF_NSP_LEARN', '0')} params={sum(p.numel() for p in self.parameters())}", flush=True)

    def feat_v(self, x):
        """[패치 56] 주 경로 특징 v = conv_block(트렁크(프레임)) — 빼기·투영 전. 트레이너가 이중 렌더 잡음 영상의 특징을 같은 가중치로 얻을 때 쓴다."""
        x = x[:, :3]
        xd = x[:, :, :-1] if self.no_diff else torch.diff(x, dim=2)
        return self.rppg_head.conv_block(self.rppg_feature_extractor(self.norm(xd[:, :3])))

    def forward(self, x):                         # x: (B,3,L,H,W) raw 0..255 frames (trainer appends the last frame)
        if x.shape[1] > 3:                        # [패치 57] RPPG_PARSE_CH=1 이면 채널 3~5 = 부위 확률(시간 불변)
            P = x[:, 3:6, 0].float(); x = x[:, :3]
            if self.bgref is not None: x = self.bgref(x.float(), P)
        elif self.bgref is not None: raise ValueError("NF_BGREF=1 needs RPPG_PARSE_CH=1 (parse channels missing)")
        B, C, L, H, W = x.shape
        g = x[:, :3].float().mean(dim=(1, 3, 4))  # (B,L) global intensity trace, gray levels (pre-diff)
        if self.no_diff:
            xd = x[:, :, :-1]
            gd = g[:, :-1]
        else:
            xd = torch.diff(x, dim=2)
            gd = torch.diff(g, dim=1)             # diff-domain trace, matches the trunk input
        length = L - 1
        xd = self.norm(xd[:, :3])
        vox = self.rppg_feature_extractor(xd)
        if self.orig_layout:
            out, fe, err = self.rppg_head(vox, B, length, noise_refs(x)) if self.arch == "nsp" else self.rppg_head(vox, B, length)     # 원본 헤드: (rPPG, factorized_embeddings, appx_error)
            s_p = s_n = bases = coef = None
        else:
            out, fe, err, s_p, s_n, bases, coef = self.rppg_head(vox, gd, length, noise_refs(x) if self.arch else None)
        k = 31
        gm = F.avg_pool1d(g.unsqueeze(1), k, stride=1, padding=k // 2, count_include_pad=False)[:, 0, :L]
        g_std = (g - gm).std(dim=1)               # gate statistic: 1-s high-passed global trace std (gray levels)
        self.nf_aux = {"s_p": s_p, "s_n": s_n, "g_diff": gd, "g_std": g_std.detach(), "bases": None if bases is None else bases.detach(),
                       "coef": coef, "att_hw": None if coef is None else self.rppg_head.fsam.last_hw,   # [패치 38] 슬롯 공간 적재(미분 가능)
                       "gate": getattr(self.rppg_head, "last_gate", None)}   # [패치 55] 주파수 게이트 (B,C,F)
        return out, vox, fe, err
