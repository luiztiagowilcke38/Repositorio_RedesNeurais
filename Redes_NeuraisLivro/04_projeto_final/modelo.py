import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        posicao = torch.arange(max_seq_len).type_as(inv_freq)
        frequencias = torch.einsum('i,j->ij', posicao, inv_freq)
        self.register_buffer("cos", frequencias.cos())
        self.register_buffer("sin", frequencias.sin())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        c = self.cos[:n, :].unsqueeze(0).unsqueeze(0)
        s = self.sin[:n, :].unsqueeze(0).unsqueeze(0)
        return (x * c) + (self._rot(x) * s)
    def _rot(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

class AtencaoSelfMultiCabeca(nn.Module):
    def __init__(self, d_m: int, n_h: int, drop: float = 0.1):
        super().__init__()
        self.n_h = n_h
        self.d_k = d_m // n_h
        self.qkv = nn.Linear(d_m, d_m * 3, bias=False)
        self.rope = RotaryPositionalEmbedding(self.d_k)
        self.out = nn.Linear(d_m, d_m)
        self.drop = nn.Dropout(drop)
        self.esc = math.sqrt(self.d_k)
    def forward(self, x: torch.Tensor, m: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_h, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rope(q), self.rope(k)
        s = torch.matmul(q, k.transpose(-2, -1)) / self.esc
        if m is not None: s = s.masked_fill(m == 0, float("-inf"))
        a = self.drop(F.softmax(s, dim=-1))
        r = torch.matmul(a, v).transpose(1, 2).reshape(b, n, d)
        return self.out(r)

class DropPathStochastic(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0: return x
        k = 1 - self.p
        f = (x.shape[0],) + (1,) * (x.ndim - 1)
        m = torch.empty(f, device=x.device).bernoulli_(k)
        if k > 0.0: m.div_(k)
        return x * m

class SqueezeExciteBlock(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.ex = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.sq(x).view(b, c)
        y = self.ex(y).view(b, c, 1, 1)
        return x * y

class InvertedResidual(nn.Module):
    def __init__(self, ci: int, co: int, e: int = 4, p: float = 0.1):
        super().__init__()
        ch = ci * e
        self.res = (ci == co)
        self.net = nn.Sequential(
            nn.Conv2d(ci, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU6(inplace=True),
            SqueezeExciteBlock(ch),
            nn.Conv2d(ch, co, 1, bias=False),
            nn.BatchNorm2d(co)
        )
        self.dp = DropPathStochastic(p) if self.res else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res: return x + self.dp(self.net(x))
        return self.net(x)

class ExpertMoE(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d), nn.Dropout(0.1))
    def forward(self, x): return self.net(x)

class MixtureExpertsAvancado(nn.Module):
    def __init__(self, d: int, n: int = 8, k: int = 1):
        super().__init__()
        self.k = k
        self.exps = nn.ModuleList([ExpertMoE(d) for _ in range(n)])
        self.gate = nn.Linear(d, n)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d_ = x.shape
        xf = x.view(-1, d_)
        lt = self.gate(xf)
        pb = F.softmax(lt, dim=-1)
        val, idx = torch.topk(pb, self.k, dim=-1)
        out = torch.zeros_like(xf)
        for i, exp in enumerate(self.exps):
            m = (idx == i).any(dim=-1)
            if m.any(): out[m] += (pb[m, i:i+1] * exp(xf[m]))
        return out.view(b, n, d_)

class FusaoCrossModal(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
    def forward(self, x1, x2):
        q, k, v = self.q(x1), self.k(x2), self.v(x2)
        at = F.softmax(torch.matmul(q, k.transpose(-2,-1))/math.sqrt(d), dim=-1)
        return self.norm(x1 + torch.matmul(at, v))

class VisionCore(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            InvertedResidual(32, 64), InvertedResidual(64, 128),
            InvertedResidual(128, 256), InvertedResidual(256, d),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, x): return self.net(x).flatten(1)

class TabularCore(nn.Module):
    def __init__(self, ni, d):
        super().__init__()
        self.lstm = nn.LSTM(ni, d//2, 2, bidirectional=True, batch_first=True)
    def forward(self, x):
        _, (h, _) = self.lstm(x.unsqueeze(1))
        return torch.cat((h[-2], h[-1]), dim=-1)

class CustomLayerNormLearnable(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(d)); self.b = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        u, s = x.mean(-1, keepdim=True), x.std(-1, keepdim=True)
        return self.a * (x - u) / (s + 1e-6) + self.b

class MultiModalFusionElite(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.dimensao_modelo
        self.text_emb = nn.Embedding(config.vocab_tamanho, d)
        self.text_blocks = nn.ModuleList([
            nn.ModuleList([CustomLayerNormLearnable(d), AtencaoSelfMultiCabeca(d, 8),
                          CustomLayerNormLearnable(d), MixtureExpertsAvancado(d)])
            for _ in range(config.num_camadas)
        ])
        self.vision = VisionCore(d)
        self.tabular = TabularCore(config.num_features_tabulares, d)
        self.fusion = FusaoCrossModal(d)
        self.head = nn.Sequential(nn.Linear(d, d//2), nn.ReLU(), nn.Linear(d//2, 2))
    def forward(self, t, i, b):
        x = self.text_emb(t)
        for ln1, at, ln2, moe in self.text_blocks:
            x = x + at(ln1(x))
            x = x + moe(ln2(x))
        xt = x.mean(1)
        xv = self.vision(i)
        xb = self.tabular(b)
        f = self.fusion(xt.unsqueeze(1), xv.unsqueeze(1))
        f = self.fusion(f, xb.unsqueeze(1))
        return self.head(f.squeeze(1))

class CustomModuleRefiner:
    def __init__(self, m): self.m = m
    def reset_params(self):
        for p in self.m.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

class AuxiliaryHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Linear(d, 1)
    def forward(self, x): return self.net(x)

class DeepStateInspector:
    def __init__(self, m): self.m = m
    def report(self):
        for n, p in self.m.named_parameters(): print(f"{n}: {p.shape}")

class AdaptiveWeightFusion(nn.Module):
    def __init__(self, n: int = 3):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n))
    def forward(self, l: List[torch.Tensor]):
        s = F.softmax(self.w, dim=0)
        return sum(f * s[i] for i, f in enumerate(l))

class FeatureSpaceProjector(nn.Module):
    def __init__(self, di, do):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(di, do), nn.LayerNorm(do), nn.ReLU())
    def forward(self, x): return self.net(x)

class AttentionGating(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
    def forward(self, x): return x * self.g(x)

class BackboneRefinementUnit(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Conv2d(d, d, 3, 1, 1, groups=d)
        self.norm = nn.BatchNorm2d(d)
    def forward(self, x): return self.norm(self.conv(x))

class GlobalAttentionPooling(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.at = nn.Sequential(nn.Linear(d, 1), nn.Softmax(dim=1))
    def forward(self, x):
        w = self.at(x)
        return (x * w).sum(1)

class ResidualCrossModalBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = FusaoCrossModal(d)
    def forward(self, x1, x2): return x1 + self.f(x1, x2)

class HybridTransformerCNN_RNN(MultiModalFusionElite):
    def __init__(self, c): super().__init__(c)

class ModelDiagnostics:
    @staticmethod
    def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

class FinalEliteModel(HybridTransformerCNN_RNN):
    def __init__(self, c): super().__init__(c)
