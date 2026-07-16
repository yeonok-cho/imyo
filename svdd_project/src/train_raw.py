"""
실제 raw RESI/TEMP 데이터로 새로 학습을 시도하는 스크립트.

기존 resi_temp_datagen.py는 요약 통계(mean/std shape)만으로 raw 데이터를
재구성했는데, 개별 profile의 hold 구간 타이밍 편차가 평균에 뭉개지는 등
실제 데이터를 충분히 반영하지 못했다 (MODEL_ARCHITECTURE.md §5 참조).

이 스크립트는 합성 데이터 대신 **실제 raw CSV**를 그대로 읽어서 학습한다.
모델 구조(Encoder → zone 통계 feature → FeatSVDD)는 그대로 재사용하되,
zone 위치는 하드코딩하지 않고 raw 데이터에서 직접 검출한다
(ramp/hold 구간을 |diff| 기반으로 자동 분할).

── 사용법 ─────────────────────────────────────────────────────────────────
1. svdd_project/data/ 밑에 아래 두 파일을 준비 (N행 x 300열, 콤마 구분, 정상 데이터만):
     resi_raw.csv
     temp_raw.csv
2. python train_raw.py
   → zone 자동 검출 결과를 출력하고, 이상해 보이면 FLAT_PCT/MIN_RUN을
     조정해서 다시 실행 (아래 하이퍼파라미터 섹션 참조).
3. 학습 완료 후 raw_svdd_model.pt / raw_norm.npy / raw_feat_norm.npy /
   raw_zones.npy 가 생성됨 (raw_zones.npy로 검출된 zone을 확인할 수 있음).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import Encoder, Decoder
from model_v2 import FeatSVDD
from resi_temp_datagen import z_normalize, linear_detrend, normalized_shape  # 범용 유틸

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 32
BATCH_SIZE = 64
LR         = 1e-3
ROOT       = os.path.dirname(os.path.dirname(__file__))
RESI_CSV   = f'{ROOT}/data/resi_raw.csv'
TEMP_CSV   = f'{ROOT}/data/temp_raw.csv'

# ── zone 자동 검출 하이퍼파라미터 ────────────────────────────────────────────
ZONE_SMOOTH_WIN = 7     # |diff|를 스무딩할 이동평균 창 크기
FLAT_PERCENTILE = 30    # 이 percentile 이하 |diff|를 '평탄(hold)'으로 간주
MIN_RUN_LEN     = 8     # 이보다 짧은 구간은 인접 구간에 합침 (너무 잘게 쪼개지는 것 방지)


# ============================================================
# 1. 데이터 로드
# ============================================================

def load_raw(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  raw 데이터가 없습니다: {path}\n"
            f"  svdd_project/data/ 아래에 정상 profile CSV(N x 300)를 넣어주세요.\n"
        )
    x = np.loadtxt(path, delimiter=',').astype(np.float64)
    if x.ndim == 1:
        x = x[None, :]
    assert x.shape[1] == 300, f"{path}: 300 타점이어야 함 (현재 {x.shape[1]})"
    return x


# ============================================================
# 2. Zone 자동 검출 — raw 데이터 자체의 구조에서 ramp/hold 구간을 찾는다
# ============================================================

def compute_channel_template(raw_profiles: np.ndarray) -> np.ndarray:
    """개별 profile을 detrend + z-normalize한 뒤 평균 → 대표 shape."""
    normed = np.array([normalized_shape(p) for p in raw_profiles])
    return normed.mean(axis=0)


def detect_zones(mean_shape: np.ndarray,
                  win: int = ZONE_SMOOTH_WIN,
                  flat_pct: float = FLAT_PERCENTILE,
                  min_run: int = MIN_RUN_LEN) -> dict:
    """
    대표 shape의 |diff|를 스무딩해서 '평탄(hold)' vs '변화(ramp)' 구간을
    자동으로 분할한다. 짧은 구간은 인접 구간에 병합해 300 전체를 커버하는
    contiguous zone dict를 만든다. 반환: {'ramp_0': (s,e), 'hold_0': (s,e), ...}
    """
    n = len(mean_shape)
    d = np.diff(mean_shape)
    k = np.ones(win) / win
    abs_d_smooth = np.convolve(np.abs(d), k, mode='same')
    thresh = np.percentile(abs_d_smooth, flat_pct)
    flat_mask = abs_d_smooth <= thresh   # length n-1

    # 1) 1차 run 분할 (diff-index 구간 [i,j)를 그대로 shape-index 구간으로 사용)
    raw_runs = []   # (start, end, is_flat)  end는 exclusive, shape-index 기준
    i = 0
    m = len(flat_mask)
    while i < m:
        j = i
        cur = flat_mask[i]
        while j < m and flat_mask[j] == cur:
            j += 1
        raw_runs.append([i, j, bool(cur)])
        i = j
    raw_runs[-1][1] = n   # 마지막 run만 끝까지(shape 마지막 포인트까지) 확장

    # 2) 너무 짧은 run은 다음 run과 합침 (같은 타입으로 흡수)
    merged = [raw_runs[0]]
    for s, e, is_flat in raw_runs[1:]:
        if e - s < min_run and merged:
            merged[-1][1] = e   # 이전 run에 흡수 (타입은 이전 run 유지)
        else:
            merged.append([s, e, is_flat])

    # 2b) 흡수 후 인접한 같은 타입 run은 하나로 합침
    collapsed = [merged[0]]
    for s, e, is_flat in merged[1:]:
        if is_flat == collapsed[-1][2]:
            collapsed[-1][1] = e
        else:
            collapsed.append([s, e, is_flat])
    merged = collapsed

    # 3) 이름 붙이기
    zones = {}
    hold_i, ramp_i = 0, 0
    for s, e, is_flat in merged:
        if is_flat:
            zones[f'hold_{hold_i}'] = (s, e)
            hold_i += 1
        else:
            zones[f'ramp_{ramp_i}'] = (s, e)
            ramp_i += 1
    return zones


# ============================================================
# 3. Zone 기반 feature 추출 (범용 — zone 개수/이름이 채널마다 달라도 됨)
# ============================================================

def extract_features(x_np: np.ndarray, zones_per_channel: list[dict]) -> np.ndarray:
    """
    x_np: (N, C, 300)
    zones_per_channel: 채널별 zone dict 리스트 (len == C)
    각 zone마다 [mean, std, diff_std] 추출
    """
    feats = []
    for ch, zones in enumerate(zones_per_channel):
        seg_all = x_np[:, ch, :]
        for s, e in zones.values():
            seg = seg_all[:, s:e]
            feats.append(seg.mean(axis=1))
            feats.append(seg.std(axis=1))
            feats.append(np.diff(seg, axis=1).std(axis=1) if e - s > 1 else np.zeros(len(seg)))
    return np.stack(feats, axis=1).astype(np.float32)


# ============================================================
# 4. 학습 (기존과 동일한 2단계: AE pretrain → FeatSVDD fine-tune)
# ============================================================

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


def train_feat_svdd(X_np, F_np, pretrained_enc, epochs=50, nu=0.05):
    print("\n[Deep SVDD] Feature-augmented single-center")
    f_mean = F_np.mean(axis=0); f_std = F_np.std(axis=0) + 1e-8
    F_norm = (F_np - f_mean) / f_std
    np.save(f'{ROOT}/raw_feat_norm.npy', {'mean': f_mean, 'std': f_std})

    X = torch.tensor(X_np)
    F = torch.tensor(F_norm)
    dl  = DataLoader(TensorDataset(X, F), batch_size=BATCH_SIZE, shuffle=True)
    dl0 = DataLoader(TensorDataset(X, F), batch_size=BATCH_SIZE, shuffle=False)

    model = FeatSVDD(LATENT_DIM, n_feats=F.shape[1], nu=nu).to(DEVICE)
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
    }, f'{ROOT}/raw_svdd_model.pt')
    print("  저장 완료: raw_svdd_model.pt")
    return model


# ============================================================
# 5. 메인
# ============================================================

if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    resi = load_raw(RESI_CSV)
    temp = load_raw(TEMP_CSV)
    assert len(resi) == len(temp), "RESI/TEMP 샘플 수가 같아야 합니다 (같은 시점 페어)"
    print(f"RESI: {resi.shape}   TEMP: {temp.shape}")

    # ── zone 자동 검출 (raw 데이터 자체에서) ────────────────────────────────
    resi_template = compute_channel_template(resi)
    temp_template  = compute_channel_template(temp)
    zones_resi = detect_zones(resi_template)
    zones_temp = detect_zones(temp_template)

    print("\n검출된 RESI zone:")
    for name, (s, e) in zones_resi.items():
        print(f"  {name:<10} [{s:3d}:{e:3d}]  len={e-s}")
    print("검출된 TEMP zone:")
    for name, (s, e) in zones_temp.items():
        print(f"  {name:<10} [{s:3d}:{e:3d}]  len={e-s}")
    print("\n  (zone이 이상하면 ZONE_SMOOTH_WIN / FLAT_PERCENTILE / MIN_RUN_LEN 조정 후 재실행)")

    np.save(f'{ROOT}/raw_zones.npy', {'resi': zones_resi, 'temp': zones_temp})

    # ── 정규화 (raw 데이터 자체의 min/max 사용) ────────────────────────────
    resi_min, resi_max = resi.min(), resi.max()
    temp_min, temp_max = temp.min(), temp.max()
    np.save(f'{ROOT}/raw_norm.npy', {
        'resi_min': resi_min, 'resi_max': resi_max,
        'temp_min': temp_min, 'temp_max': temp_max,
    })

    resi_n = (resi - resi_min) / (resi_max - resi_min)
    temp_n = (temp - temp_min) / (temp_max - temp_min)
    X_np = np.stack([resi_n, temp_n], axis=1).astype(np.float32)

    F_np = extract_features(X_np, [zones_resi, zones_temp])
    print(f"\n학습 데이터: X={X_np.shape}  F(zone feature)={F_np.shape}")

    pretrained = pretrain_encoder(X_np, epochs=30)
    train_feat_svdd(X_np, F_np, pretrained, epochs=50)

    print("\n완료! raw_svdd_model.pt / raw_norm.npy / raw_feat_norm.npy / raw_zones.npy 생성됨")
