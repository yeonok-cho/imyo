"""
ACT.RESI / ACT.TEMP 실측 통계 기반 가상 데이터로 Deep SVDD 학습.

입력: RESI, TEMP 두 신호를 2채널로 stack (기존 model.py/model_v2.py의
      Profile A / Profile B 2채널 구조를 그대로 재사용)
정상 데이터만 사용 (anomaly_ratio=0) — One-class 학습.

실행:
    python train_resi_temp.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import Encoder, Decoder            # pretrain (AE)
from model_v2 import FeatSVDD, extract_features
from resi_temp_datagen import (
    RESI_CONFIG, TEMP_CONFIG,
    generate_paired_dataset,
)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 32
BATCH_SIZE = 64
LR         = 1e-3
N_TRAIN    = 2000
ROOT       = os.path.dirname(os.path.dirname(__file__))   # svdd_project/
print(f"Device: {DEVICE}")


def load_data(n_profiles=N_TRAIN, random_state=2026):
    """정상 RESI/TEMP 프로파일 생성 → min-max 정규화 → (N,2,300) stack"""
    resi, temp, _ = generate_paired_dataset(
        n_profiles=n_profiles,
        anomaly_ratio=0.0,
        random_state=random_state,
    )

    resi_n = (resi - RESI_CONFIG.raw_min) / (RESI_CONFIG.raw_max - RESI_CONFIG.raw_min)
    temp_n = (temp - TEMP_CONFIG.raw_min) / (TEMP_CONFIG.raw_max - TEMP_CONFIG.raw_min)

    X = np.stack([resi_n, temp_n], axis=1).astype(np.float32)

    norm = {
        'resi_min': RESI_CONFIG.raw_min, 'resi_max': RESI_CONFIG.raw_max,
        'temp_min': TEMP_CONFIG.raw_min, 'temp_max': TEMP_CONFIG.raw_max,
    }
    np.save(f'{ROOT}/resi_temp_norm.npy', norm)

    return X


def pretrain_encoder(X_np, epochs=30):
    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = Encoder(LATENT_DIM)
            self.dec = Decoder(LATENT_DIM)
        def forward(self, x): return self.dec(self.enc(x))

    X   = torch.tensor(X_np)
    ae  = AE().to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=LR)
    dl  = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=True)

    print("\n[Pretrain] Autoencoder")
    for ep in range(1, epochs + 1):
        ae.train()
        loss_sum = 0.0
        for x, in dl:
            x = x.to(DEVICE)
            loss = nn.MSELoss()(ae(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()
        if ep % 10 == 0:
            print(f"  Epoch {ep:3d}/{epochs}  loss={loss_sum/len(dl):.5f}")
    return ae.enc


def train_feat_svdd(X_np, pretrained_enc, epochs=50):
    print("\n[Deep SVDD] Feature-augmented single-center")
    F_np = extract_features(X_np)
    f_mean = F_np.mean(axis=0); f_std = F_np.std(axis=0) + 1e-8
    F_norm = (F_np - f_mean) / f_std
    np.save(f'{ROOT}/resi_temp_feat_norm.npy', {'mean': f_mean, 'std': f_std})

    X = torch.tensor(X_np)
    F = torch.tensor(F_norm)
    dl  = DataLoader(TensorDataset(X, F), batch_size=BATCH_SIZE, shuffle=True)
    dl0 = DataLoader(TensorDataset(X, F), batch_size=BATCH_SIZE, shuffle=False)

    model = FeatSVDD(LATENT_DIM, n_feats=F.shape[1], nu=0.05).to(DEVICE)
    model.encoder.cnn.load_state_dict(pretrained_enc.state_dict())
    model.init_center(dl0, DEVICE)
    print(f"  R init: {model.R.item():.4f}")

    opt = torch.optim.Adam(model.parameters(), lr=LR * 0.1)
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, f in dl:
            x, f = x.to(DEVICE), f.to(DEVICE)
            loss = model.svdd_loss(x, f)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if ep % 10 == 0:
            print(f"  Epoch {ep:3d}/{epochs}  loss={total/len(dl):.5f}  R={model.R.item():.4f}")

    torch.save({
        'encoder': model.encoder.state_dict(),
        'c': model.c, 'R': model.R.item(),
        'latent_dim': LATENT_DIM, 'n_feats': F.shape[1],
    }, f'{ROOT}/resi_temp_svdd_model.pt')
    print("  저장 완료: resi_temp_svdd_model.pt")
    return model


if __name__ == '__main__':
    X_np = load_data()
    print(f"학습 데이터: {X_np.shape}  (RESI+TEMP 정상 프로파일)")

    pretrained = pretrain_encoder(X_np, epochs=30)
    train_feat_svdd(X_np, pretrained, epochs=50)

    print("\n모든 학습 완료! → evaluate_resi_temp.py 로 평가하세요.")
