"""
두 가지 개선 방법:

Approach 1: SVDD + Reconstruction Error (Hybrid)
  - inference 시 encoder + decoder 둘 다 사용
  - score = α * svdd_dist + (1-α) * recon_error

Approach 2: SVDD + Pointwise Features (Feature-augmented)
  - 구간별 통계(std, diff_std)를 CNN latent에 concat
  - inference 시 encoder만 사용 (decoder 불필요)
  - 가장 빠른 방법
"""

import numpy as np
import torch
import torch.nn as nn

# ── 공통 Encoder/Decoder ─────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16), nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(64, latent_dim, bias=False)

    def forward(self, x):
        return self.fc(self.net(x))


class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 19)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(64, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16, 2,  4, stride=2, padding=1, bias=False),
            nn.AdaptiveAvgPool1d(300),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 64, 19))


# ── Approach 1: Hybrid SVDD + Reconstruction ─────────────────────────────────
class HybridSVDD(nn.Module):
    """
    score = α * ||z - c||²  +  (1-α) * MSE(x, x_hat)
    inference: encoder + decoder 모두 필요
    """
    def __init__(self, latent_dim=32, nu=0.05, alpha=0.5):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.nu = nu
        self.alpha = alpha
        self.register_buffer('c', torch.zeros(latent_dim))
        self.R = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        z    = self.encoder(x)
        xhat = self.decoder(z)
        return z, xhat

    @torch.no_grad()
    def anomaly_score(self, x):
        z    = self.encoder(x)
        xhat = self.decoder(z)
        dist = torch.sum((z - self.c) ** 2, dim=1)
        recon = torch.mean((x - xhat) ** 2, dim=(1, 2))
        # 각각 정규화 후 결합
        return self.alpha * dist + (1 - self.alpha) * recon

    def svdd_loss(self, x):
        z, xhat = self.forward(x)
        dist  = torch.sum((z - self.c) ** 2, dim=1)
        R2    = self.R ** 2
        svdd  = R2 + (1/(self.nu * x.size(0))) * torch.sum(torch.clamp(dist - R2, min=0))
        recon = nn.MSELoss()(xhat, x)
        return svdd + (1 - self.alpha) * recon

    @torch.no_grad()
    def init_center(self, loader, device):
        zs = [self.encoder(x.to(device)) for x, in loader]
        c  = torch.cat(zs).mean(dim=0)
        c[(c.abs() < 0.01) & (c >= 0)] =  0.01
        c[(c.abs() < 0.01) & (c <  0)] = -0.01
        self.c.copy_(c)
        dists = torch.cat([torch.sum((self.encoder(x.to(device)) - self.c)**2, dim=1)
                           for x, in loader])
        self.R.data = torch.sqrt(torch.quantile(dists, 1 - self.nu))


# ── Approach 2: Feature-augmented SVDD ───────────────────────────────────────
# 구간 정의 (300pt 기준)
ZONES = {'flat': (35,160), 'transition': (160,200), 'zero': (200,215),
         'spike': (215,245), 'tail': (245,300)}

def extract_features(x_np: np.ndarray) -> np.ndarray:
    """
    x_np: (N, 2, 300) numpy
    각 채널 × 구간 × [mean, std, diff_std] → 2 × 5 × 3 = 30 features
    계산 비용: O(N × 300) numpy 연산, 매우 빠름
    """
    feats = []
    for ch in range(2):
        for s, e in ZONES.values():
            seg = x_np[:, ch, s:e]
            feats.append(seg.mean(axis=1))
            feats.append(seg.std(axis=1))
            feats.append(np.diff(seg, axis=1).std(axis=1))
    return np.stack(feats, axis=1).astype(np.float32)   # (N, 30)


class FeatEncoder(nn.Module):
    """CNN latent (32) + 구간 통계 features (30) → concat → fc → latent"""
    def __init__(self, latent_dim=32, n_feats=30):
        super().__init__()
        self.cnn = Encoder(latent_dim)
        # 통계 features 처리 (작은 MLP)
        self.feat_fc = nn.Sequential(
            nn.Linear(n_feats, 16, bias=False),
            nn.LeakyReLU(0.2),
        )
        # 합쳐서 최종 latent
        self.merge = nn.Linear(latent_dim + 16, latent_dim, bias=False)

    def forward(self, x, feats):
        z_cnn  = self.cnn(x)
        z_feat = self.feat_fc(feats)
        return self.merge(torch.cat([z_cnn, z_feat], dim=1))


class FeatSVDD(nn.Module):
    """
    inference: encoder(CNN + feat_fc + merge) 만 사용 → 빠름
    """
    def __init__(self, latent_dim=32, n_feats=30, nu=0.05):
        super().__init__()
        self.encoder = FeatEncoder(latent_dim, n_feats)
        self.nu = nu
        self.register_buffer('c', torch.zeros(latent_dim))
        self.R = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, feats):
        return self.encoder(x, feats)

    @torch.no_grad()
    def anomaly_score(self, x, feats):
        z = self.encoder(x, feats)
        return torch.sum((z - self.c) ** 2, dim=1)

    def svdd_loss(self, x, feats):
        dist = self.anomaly_score(x, feats)
        R2   = self.R ** 2
        return R2 + (1/(self.nu * x.size(0))) * torch.sum(torch.clamp(dist - R2, min=0))

    @torch.no_grad()
    def init_center(self, loader, device):
        zs = [self.encoder(x.to(device), f.to(device)) for x, f in loader]
        c  = torch.cat(zs).mean(dim=0)
        c[(c.abs() < 0.01) & (c >= 0)] =  0.01
        c[(c.abs() < 0.01) & (c <  0)] = -0.01
        self.c.copy_(c)
        dists = torch.cat([self.anomaly_score(x.to(device), f.to(device))
                           for x, f in loader])
        self.R.data = torch.sqrt(torch.quantile(dists, 1 - self.nu))


# ── Multi-center SVDD (K=2) ───────────────────────────────────────────────────
class MultiCenterSVDD(nn.Module):
    """
    K=2 중심 (c_0, c_1), 각각 반경 (R_0, R_1) 학습.

    anomaly_score = min_k( ||z - c_k||² / R_k² )
    threshold     = 1.0   (정규화된 거리이므로 고정)

    → score < 1.0 : 어느 구에든 속함 = 정상
    → score ≥ 1.0 : 모든 구 밖 = 이상

    inference: encoder forward pass + 거리 계산만 (decoder 불필요)
    """
    def __init__(self, latent_dim=32, n_feats=30, K=2, nu=0.05):
        super().__init__()
        self.K          = K
        self.nu         = nu
        self.latent_dim = latent_dim
        self.encoder    = FeatEncoder(latent_dim, n_feats)
        for k in range(K):
            self.register_buffer(f'c_{k}', torch.zeros(latent_dim))
            self.register_parameter(f'R_{k}', nn.Parameter(torch.tensor(1.0)))

    def _centers(self):
        return [getattr(self, f'c_{k}') for k in range(self.K)]

    def _radii(self):
        return [getattr(self, f'R_{k}') for k in range(self.K)]

    def _encode(self, x, feats):
        return self.encoder(x, feats)

    def _norm_dists(self, z):
        """정규화 거리 행렬 (N, K):  ||z-c_k||² / R_k²"""
        return torch.stack(
            [torch.sum((z - c)**2, dim=1) / (R**2 + 1e-8)
             for c, R in zip(self._centers(), self._radii())],
            dim=1
        )

    @torch.no_grad()
    def anomaly_score(self, x, feats):
        """score < 1.0 = 정상,  score ≥ 1.0 = 이상"""
        return self._norm_dists(self._encode(x, feats)).min(dim=1).values

    def svdd_loss(self, x, feats):
        z         = self._encode(x, feats)
        raw_dists = torch.stack(
            [torch.sum((z - c)**2, dim=1) for c in self._centers()], dim=1
        )  # (N, K)
        assignments = raw_dists.detach().argmin(dim=1)

        loss = torch.tensor(0.0, device=x.device)
        for k in range(self.K):
            mask = (assignments == k)
            if mask.sum() == 0:
                continue
            R_k  = self._radii()[k]
            d_k  = raw_dists[mask, k]
            loss = loss + (R_k**2 + (1/(self.nu * mask.sum())) *
                           torch.sum(torch.clamp(d_k - R_k**2, min=0))) / self.K
        return loss

    @torch.no_grad()
    def init_centers(self, loader, device, n_iter=50):
        """k-means로 중심 초기화 후 각 클러스터의 (1-nu) 분위수로 R 초기화"""
        self.encoder.eval()
        zs = torch.cat([self._encode(x.to(device), f.to(device)) for x, f in loader])

        # k-means
        cents = [zs[torch.randperm(len(zs))[k]].clone() for k in range(self.K)]
        for _ in range(n_iter):
            dists   = torch.stack([torch.sum((zs-c)**2, dim=1) for c in cents], dim=1)
            assigns = dists.argmin(dim=1)
            cents   = [zs[assigns==k].mean(0) if (assigns==k).sum()>0 else cents[k]
                       for k in range(self.K)]

        dists   = torch.stack([torch.sum((zs-c)**2, dim=1) for c in cents], dim=1)
        assigns = dists.argmin(dim=1)

        for k in range(self.K):
            c = cents[k]
            c[(c.abs()<0.01)&(c>=0)] =  0.01
            c[(c.abs()<0.01)&(c< 0)] = -0.01
            getattr(self, f'c_{k}').copy_(c)
            mask   = (assigns == k)
            d_k    = torch.sum((zs[mask] - c)**2, dim=1)
            R_init = torch.sqrt(torch.quantile(d_k, 1 - self.nu))
            getattr(self, f'R_{k}').data = R_init
            print(f"  Center {k}: n={mask.sum().item()}  R={R_init:.4f}")
