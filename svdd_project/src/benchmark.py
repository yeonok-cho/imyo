import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time, tracemalloc
import numpy as np
import torch
from model import DeepSVDD

DEVICE = torch.device('cpu')

# 모델 로드
ckpt  = torch.load('/home/imyo/svdd_project/svdd_model.pt', map_location=DEVICE)
model = DeepSVDD(ckpt['latent_dim']).to(DEVICE)
model.encoder.load_state_dict(ckpt['encoder'])
model.c.copy_(ckpt['c'])
model.R.data = torch.tensor(ckpt['R'])
model.eval()

norm = np.load('/home/imyo/svdd_project/norm_params.npy', allow_pickle=True).item()

def make_input(n):
    A = np.random.randn(n, 300).astype(np.float32)
    B = np.random.randn(n, 300).astype(np.float32)
    A = (A - norm['A_min']) / (norm['A_max'] - norm['A_min'])
    B = (B - norm['B_min']) / (norm['B_max'] - norm['B_min'])
    return torch.tensor(np.stack([A, B], axis=1))

WARMUP = 20
REPEAT = 200

print("=" * 50)
print("  Inference Benchmark (CPU)")
print("=" * 50)

for n in [1, 10, 100, 500]:
    x = make_input(n)

    # warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model.anomaly_score(x)

    # 시간 측정
    times = []
    with torch.no_grad():
        for _ in range(REPEAT):
            t0 = time.perf_counter()
            score = model.anomaly_score(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    times = np.array(times)
    per_sample = times.mean() / n

    print(f"\n  배치 크기: {n:>4}개")
    print(f"  평균 총 시간:      {times.mean():.3f} ms")
    print(f"  1개당 시간:        {per_sample:.4f} ms")
    print(f"  min / max:         {times.min():.3f} / {times.max():.3f} ms")

# 메모리 측정 (1개)
print("\n" + "=" * 50)
print("  메모리 사용량 (1개 프로파일)")
print("=" * 50)
x1 = make_input(1)
tracemalloc.start()
with torch.no_grad():
    _ = model.anomaly_score(x1)
cur, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"  현재 메모리:  {cur  / 1024:.1f} KB")
print(f"  피크 메모리:  {peak / 1024:.1f} KB")

# 모델 크기
param_count = sum(p.numel() for p in model.encoder.parameters())
param_mb    = sum(p.numel() * 4 for p in model.encoder.parameters()) / 1024 / 1024
print(f"\n  Encoder 파라미터 수: {param_count:,}")
print(f"  Encoder 모델 크기:   {param_mb:.2f} MB")
print("=" * 50)
