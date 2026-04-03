"""
Approach 1 & 2 학습
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import Encoder, Decoder   # pretrain용
from model_v2 import HybridSVDD, FeatSVDD, MultiCenterSVDD, extract_features

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 32
BATCH_SIZE = 64
LR         = 1e-3
ROOT       = os.path.dirname(os.path.dirname(__file__))   # svdd_project/
print(f"Device: {DEVICE}")


def load_data():
    A = np.loadtxt(f'{ROOT}/data/profile_A.csv', delimiter=',').astype(np.float32)
    B = np.loadtxt(f'{ROOT}/data/profile_B.csv', delimiter=',').astype(np.float32)
    norm = np.load(f'{ROOT}/norm_params.npy', allow_pickle=True).item()
    A = (A - norm['A_min']) / (norm['A_max'] - norm['A_min'])
    B = (B - norm['B_min']) / (norm['B_max'] - norm['B_min'])
    X = np.stack([A, B], axis=1)
    return X


def pretrain_encoder(X_np, epochs=30):
    """공통 pretrain (기존과 동일)"""
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
    for ep in range(1, epochs+1):
        ae.train()
        loss_sum = sum(
            (lambda loss: (opt.zero_grad(), loss.backward(), opt.step(), loss.item())[-1])(
                nn.MSELoss()(ae(x.to(DEVICE)), x.to(DEVICE))
            ) for x, in dl
        )
        if ep % 10 == 0:
            print(f"  Epoch {ep:3d}/{epochs}  loss={loss_sum/len(dl):.5f}")
    return ae.enc


# ── Approach 1: Hybrid ────────────────────────────────────────────────────────
def train_hybrid(X_np, pretrained_enc, epochs=50):
    print("\n[Approach 1] Hybrid SVDD + Reconstruction")
    X   = torch.tensor(X_np)
    dl  = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=True)
    dl0 = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=False)

    model = HybridSVDD(LATENT_DIM, nu=0.05, alpha=0.6).to(DEVICE)
    model.encoder.load_state_dict(pretrained_enc.state_dict())
    model.init_center(dl0, DEVICE)
    print(f"  R init: {model.R.item():.4f}")

    opt = torch.optim.Adam(model.parameters(), lr=LR * 0.1)
    for ep in range(1, epochs+1):
        model.train()
        total = 0
        for x, in dl:
            x = x.to(DEVICE)
            loss = model.svdd_loss(x)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if ep % 10 == 0:
            print(f"  Epoch {ep:3d}/{epochs}  loss={total/len(dl):.5f}  R={model.R.item():.4f}")

    torch.save({'encoder': model.encoder.state_dict(),
                'decoder': model.decoder.state_dict(),
                'c': model.c, 'R': model.R.item(),
                'alpha': model.alpha, 'latent_dim': LATENT_DIM},
               f'{ROOT}/hybrid_model.pt')
    print("  저장 완료: hybrid_model.pt")
    return model


# ── Approach 2: Feature-augmented ─────────────────────────────────────────────
def train_feat(X_np, pretrained_enc, epochs=50):
    print("\n[Approach 2] Feature-augmented SVDD")
    F_np = extract_features(X_np)
    # feature 정규화
    f_mean = F_np.mean(axis=0); f_std = F_np.std(axis=0) + 1e-8
    F_norm = (F_np - f_mean) / f_std
    np.save(f'{ROOT}/feat_norm.npy', {'mean': f_mean, 'std': f_std})

    X = torch.tensor(X_np)
    F = torch.tensor(F_norm)
    dl  = DataLoader(TensorDataset(X, F), batch_size=BATCH_SIZE, shuffle=True)
    dl0 = DataLoader(TensorDataset(X, F), batch_size=BATCH_SIZE, shuffle=False)

    model = FeatSVDD(LATENT_DIM, n_feats=F.shape[1], nu=0.05).to(DEVICE)
    model.encoder.cnn.load_state_dict(pretrained_enc.state_dict())
    model.init_center(dl0, DEVICE)
    print(f"  R init: {model.R.item():.4f}")

    opt = torch.optim.Adam(model.parameters(), lr=LR * 0.1)
    for ep in range(1, epochs+1):
        model.train()
        total = 0
        for x, f in dl:
            x, f = x.to(DEVICE), f.to(DEVICE)
            loss = model.svdd_loss(x, f)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if ep % 10 == 0:
            print(f"  Epoch {ep:3d}/{epochs}  loss={total/len(dl):.5f}  R={model.R.item():.4f}")

    torch.save({'encoder': model.encoder.state_dict(),
                'c': model.c, 'R': model.R.item(),
                'latent_dim': LATENT_DIM, 'n_feats': F.shape[1]},
               f'{ROOT}/svdd_model_feat.pt')
    print("  저장 완료: feat_model.pt")
    return model


# ── Multi-center SVDD (K=2) ───────────────────────────────────────────────────
def train_multicenter(X_np, pretrained_enc, epochs=50, K=2):
    print(f"\n[Approach 3] Multi-center SVDD  (K={K})")
    F_np   = extract_features(X_np)
    fnorm  = np.load(f'{ROOT}/feat_norm.npy', allow_pickle=True).item()
    F_norm = ((F_np - fnorm['mean']) / fnorm['std']).astype(np.float32)

    X  = torch.tensor(X_np)
    F  = torch.tensor(F_norm)
    dl  = DataLoader(TensorDataset(X, F), batch_size=BATCH_SIZE, shuffle=True)
    dl0 = DataLoader(TensorDataset(X, F), batch_size=BATCH_SIZE, shuffle=False)

    model = MultiCenterSVDD(LATENT_DIM, n_feats=F.shape[1], K=K, nu=0.05).to(DEVICE)
    model.encoder.cnn.load_state_dict(pretrained_enc.state_dict())

    print("  k-means 중심 초기화 중...")
    model.init_centers(dl0, DEVICE, n_iter=100)

    opt = torch.optim.Adam(model.parameters(), lr=LR * 0.1)
    for ep in range(1, epochs+1):
        model.train()
        total = 0
        for x, f in dl:
            x, f = x.to(DEVICE), f.to(DEVICE)
            loss = model.svdd_loss(x, f)
            opt.zero_grad(); loss.backward(); opt.step()
            # R이 음수가 되지 않도록
            for k in range(K):
                getattr(model, f'R_{k}').data.clamp_(min=1e-4)
            total += loss.item()
        if ep % 10 == 0:
            Rs = [f"R{k}={getattr(model,f'R_{k}').item():.4f}" for k in range(K)]
            print(f"  Epoch {ep:3d}/{epochs}  loss={total/len(dl):.5f}  {', '.join(Rs)}")

    save_dict = {
        'encoder'   : model.encoder.state_dict(),
        'K'         : K,
        'latent_dim': LATENT_DIM,
        'n_feats'   : F.shape[1],
        'nu'        : model.nu,
    }
    for k in range(K):
        save_dict[f'c_{k}'] = getattr(model, f'c_{k}')
        save_dict[f'R_{k}'] = getattr(model, f'R_{k}').item()

    torch.save(save_dict, f'{ROOT}/multicenter_model.pt')
    print("  저장 완료: multicenter_model.pt")
    return model


if __name__ == '__main__':
    X_np       = load_data()
    pretrained = pretrain_encoder(X_np, epochs=30)
    train_feat(X_np, pretrained, epochs=50)         # feat_norm.npy 생성
    train_multicenter(X_np, pretrained, epochs=50)
    print("\n모든 학습 완료!")
