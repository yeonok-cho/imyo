"""
학습 파이프라인
Step 1: Autoencoder pretrain (reconstruction)
Step 2: SVDD fine-tune (encoder only)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import Autoencoder, DeepSVDD

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 16
BATCH_SIZE = 64
LR         = 1e-3

print(f"Device: {DEVICE}")


def load_data():
    A = np.loadtxt('/home/imyo/svdd_project/data/profile_A.csv', delimiter=',').astype(np.float32)
    B = np.loadtxt('/home/imyo/svdd_project/data/profile_B.csv', delimiter=',').astype(np.float32)
    # 정규화 파라미터 저장
    norm_params = {
        'A_min': float(A.min()), 'A_max': float(A.max()),
        'B_min': float(B.min()), 'B_max': float(B.max()),
    }
    np.save('/home/imyo/svdd_project/norm_params.npy', norm_params)
    A = (A - norm_params['A_min']) / (norm_params['A_max'] - norm_params['A_min'])
    B = (B - norm_params['B_min']) / (norm_params['B_max'] - norm_params['B_min'])
    X = np.stack([A, B], axis=1)
    return torch.tensor(X)


def make_loader(X, shuffle=True):
    return DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=shuffle)


# ── Step 1: Pretrain ──────────────────────────────────────────────────────────
def pretrain(X, epochs=30):
    print("\n[Step 1] Autoencoder Pretrain")
    model  = Autoencoder(LATENT_DIM).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    loader = make_loader(X)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for x, in loader:
            x = x.to(DEVICE)
            loss = nn.MSELoss()(model(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={total_loss/len(loader):.5f}")

    torch.save(model.encoder.state_dict(), '/home/imyo/svdd_project/encoder_pretrained.pt')
    print("  pretrained encoder 저장 완료")
    return model.encoder


# ── Step 2: SVDD Fine-tune ────────────────────────────────────────────────────
def finetune_svdd(X, pretrained_encoder, epochs=50):
    print("\n[Step 2] SVDD Fine-tune")
    model = DeepSVDD(LATENT_DIM).to(DEVICE)
    model.encoder.load_state_dict(pretrained_encoder.state_dict())

    loader = make_loader(X, shuffle=False)
    model.init_center(loader, DEVICE)
    print(f"  center norm: {model.c.norm():.4f},  R init: {model.R.item():.4f}")

    loader = make_loader(X, shuffle=True)
    # R은 고정 (init값 유지), encoder만 fine-tune
    opt = torch.optim.Adam(model.encoder.parameters(), lr=LR * 0.1)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for x, in loader:
            x = x.to(DEVICE)
            loss = model.svdd_loss(x)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={total_loss/len(loader):.5f}  R={model.R.item():.4f}")

    torch.save({
        'encoder': model.encoder.state_dict(),
        'c':       model.c,
        'R':       model.R.item(),
        'latent_dim': LATENT_DIM,
    }, '/home/imyo/svdd_project/svdd_model.pt')
    print("  SVDD 모델 저장 완료")
    return model


if __name__ == '__main__':
    X = load_data()
    print(f"학습 데이터: {X.shape}  (samples x channels x time)")

    pretrained_enc = pretrain(X, epochs=30)
    svdd_model     = finetune_svdd(X, pretrained_enc, epochs=50)
    print("\n학습 완료!")
