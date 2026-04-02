"""
1D-CNN Encoder + Deep SVDD
- Input: (batch, 2, 300)  → 2채널 (Profile A, Profile B)
- Pretrain: Autoencoder (reconstruction)
- Fine-tune: SVDD loss (hypersphere)
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 2, 300) -> (B, 16, 150)
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16), nn.LeakyReLU(0.2),
            # -> (B, 32, 75)
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            # -> (B, 64, 38)
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            # -> (B, 64, 19)
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),   # -> (B, 64, 1)
            nn.Flatten(),              # -> (B, 64)
        )
        self.fc = nn.Linear(64, latent_dim, bias=False)

    def forward(self, x):
        return self.fc(self.net(x))


class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 19)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16, 2,  kernel_size=4, stride=2, padding=1, bias=False),
            nn.AdaptiveAvgPool1d(300),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 64, 19)
        return self.net(x)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DeepSVDD(nn.Module):
    def __init__(self, latent_dim=32, nu=0.05):
        super().__init__()
        self.encoder  = Encoder(latent_dim)
        self.latent_dim = latent_dim
        self.nu       = nu
        self.register_buffer('c', torch.zeros(latent_dim))
        self.R        = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.encoder(x)

    def anomaly_score(self, x):
        z    = self.encoder(x)
        dist = torch.sum((z - self.c) ** 2, dim=1)
        return dist

    def svdd_loss(self, x):
        dist = self.anomaly_score(x)
        R2   = self.R ** 2
        loss = R2 + (1 / (self.nu * x.size(0))) * torch.sum(torch.clamp(dist - R2, min=0))
        return loss

    @torch.no_grad()
    def init_center(self, loader, device):
        """정상 데이터로 중심 c 초기화"""
        zs = []
        self.encoder.eval()
        for x, in loader:
            zs.append(self.encoder(x.to(device)))
        c = torch.cat(zs).mean(dim=0)
        # 중심이 0 근처면 collapse 위험 → 최소값 보장
        c[(c.abs() < 0.01) & (c >= 0)] =  0.01
        c[(c.abs() < 0.01) & (c <  0)] = -0.01
        self.c.copy_(c)
        # R 초기화: 정상 거리의 nu 분위수
        dists = torch.cat([
            torch.sum((self.encoder(x.to(device)) - self.c) ** 2, dim=1)
            for x, in loader
        ])
        self.R.data = torch.sqrt(torch.quantile(dists, 1 - self.nu))
