"""Multi-center SVDD 평가 + 단일 중심 FeatSVDD 비교"""
import sys, os, time, tracemalloc
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report
from model_v2 import FeatSVDD, MultiCenterSVDD, extract_features
from anomaly_generator import (make_anomaly_batch, ANOMALY_TYPES, TEMPLATE_A, TEMPLATE_B,
    _add_excess_noise, _add_spike_anomaly, _add_mean_shift, _add_local_burst, _add_shape_distortion)

DEVICE = torch.device('cpu')
ROOT   = os.path.dirname(os.path.dirname(__file__))

norm  = np.load(f'{ROOT}/norm_params.npy',  allow_pickle=True).item()
fnorm = np.load(f'{ROOT}/feat_norm.npy',    allow_pickle=True).item()

def normalize(A, B):
    An = (A - norm['A_min']) / (norm['A_max'] - norm['A_min'])
    Bn = (B - norm['B_min']) / (norm['B_max'] - norm['B_min'])
    return np.stack([An, Bn], axis=1).astype(np.float32)

def norm_feats(X):
    return ((extract_features(X) - fnorm['mean']) / fnorm['std']).astype(np.float32)

def load_feat():
    ck = torch.load(f'{ROOT}/svdd_model_feat.pt', map_location=DEVICE)
    m  = FeatSVDD(ck['latent_dim'], ck['n_feats']).to(DEVICE)
    m.encoder.load_state_dict(ck['encoder'])
    m.c.copy_(ck['c']); m.R.data = torch.tensor(ck['R']); m.eval()
    return m, ck['R']**2

def load_mc():
    ck = torch.load(f'{ROOT}/multicenter_model.pt', map_location=DEVICE)
    m  = MultiCenterSVDD(ck['latent_dim'], ck['n_feats'], K=ck['K']).to(DEVICE)
    m.encoder.load_state_dict(ck['encoder'])
    for k in range(ck['K']):
        getattr(m, f'c_{k}').copy_(ck[f'c_{k}'])
        getattr(m, f'R_{k}').data = torch.tensor(ck[f'R_{k}'])
    m.eval()
    return m, 1.0   # threshold=1.0 (정규화 거리)

def gen_samples(atype, n=100):
    As, Bs = [], []
    for _ in range(n):
        a = TEMPLATE_A + np.random.normal(0, 0.05, 300)
        b = TEMPLATE_B + np.random.normal(0, 0.05, 300)
        if   atype=='excess_noise':     a=_add_excess_noise(a);               b=_add_excess_noise(b)
        elif atype=='spike_low':        a=_add_spike_anomaly(a,TEMPLATE_A,'low');   b=_add_spike_anomaly(b,TEMPLATE_B,'low')
        elif atype=='spike_high':       a=_add_spike_anomaly(a,TEMPLATE_A,'high');  b=_add_spike_anomaly(b,TEMPLATE_B,'high')
        elif atype=='spike_missing':    a=_add_spike_anomaly(a,TEMPLATE_A,'missing');b=_add_spike_anomaly(b,TEMPLATE_B,'missing')
        elif atype=='mean_shift':       a=_add_mean_shift(a);                 b=_add_mean_shift(b)
        elif atype=='local_burst':      a=_add_local_burst(a);                b=_add_local_burst(b)
        elif atype=='shape_distortion': a=_add_shape_distortion(a,TEMPLATE_A);b=_add_shape_distortion(b,TEMPLATE_B)
        As.append(a); Bs.append(b)
    return np.array(As,dtype=np.float32), np.array(Bs,dtype=np.float32)

@torch.no_grad()
def det_rate(model, thresh, atype, is_mc=False):
    As, Bs = gen_samples(atype, 100)
    X  = torch.tensor(normalize(As, Bs))
    F  = torch.tensor(norm_feats(normalize(As, Bs)))
    s  = model.anomaly_score(X, F).numpy()
    return (s > thresh).mean()

def benchmark_one(model, thresh, is_mc, n_rep=300):
    a = np.random.randn(1,300).astype(np.float32)
    b = np.random.randn(1,300).astype(np.float32)
    X = torch.tensor(normalize(a, b))
    F = torch.tensor(norm_feats(normalize(a, b)))
    fn = lambda: model.anomaly_score(X, F)
    with torch.no_grad():
        for _ in range(50): fn()
    times = []
    with torch.no_grad():
        for _ in range(n_rep):
            t0=time.perf_counter(); fn(); times.append((time.perf_counter()-t0)*1000)
    tracemalloc.start()
    with torch.no_grad(): fn()
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return np.mean(times), np.std(times), peak/1024, sum(p.numel() for p in model.parameters())

if __name__ == '__main__':
    feat_m, feat_th = load_feat()
    mc_m,   mc_th   = load_mc()

    normal_A = np.loadtxt(f'{ROOT}/data/profile_A.csv', delimiter=',').astype(np.float32)
    normal_B = np.loadtxt(f'{ROOT}/data/profile_B.csv', delimiter=',').astype(np.float32)

    print("=" * 62)
    print("  Detection Rate:  FeatSVDD (단일) vs MultiCenter SVDD (K=2)")
    print("=" * 62)
    print(f"  {'Type':<22} {'Single':>10} {'Multi-K2':>10}")
    print("  " + "-"*46)

    results = {}
    for atype in ANOMALY_TYPES:
        r1 = det_rate(feat_m, feat_th, atype)
        r2 = det_rate(mc_m,   mc_th,   atype, is_mc=True)
        results[atype] = (r1, r2)
        f1 = '✅' if r1>=0.5 else '❌'
        f2 = '✅' if r2>=0.5 else '❌'
        print(f"  {atype:<22} {f1}{r1:>7.0%}   {f2}{r2:>7.0%}")

    print("\n" + "=" * 62)
    print("  Computational Cost (1 profile, CPU)")
    print("=" * 62)
    for label, model, th, is_mc in [
        ('Single FeatSVDD', feat_m, feat_th, False),
        ('Multi-center (K=2)', mc_m, mc_th,  True),
    ]:
        mu, sd, mem, params = benchmark_one(model, th, is_mc)
        print(f"\n  [{label}]")
        print(f"    Inference time : {mu:.3f} ± {sd:.3f} ms")
        print(f"    Peak memory    : {mem:.1f} KB")
        print(f"    Parameters     : {params:,}")
        print(f"    Model size     : {params*4/1024/1024:.2f} MB")

    # ── 시각화: latent space (PCA 2D) ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Multi-center SVDD (K=2) vs Single FeatSVDD', fontsize=13, fontweight='bold')

    # 탐지율 비교
    x   = np.arange(len(ANOMALY_TYPES))
    w   = 0.35
    axes[0].bar(x-w/2, [results[t][0] for t in ANOMALY_TYPES], w, label='Single', color='#2196F3', alpha=0.8)
    axes[0].bar(x+w/2, [results[t][1] for t in ANOMALY_TYPES], w, label='Multi-K2', color='#FF5722', alpha=0.8)
    axes[0].axhline(0.8, color='red', linestyle='--', linewidth=1.2)
    axes[0].set_xticks(x); axes[0].set_xticklabels(ANOMALY_TYPES, rotation=30, ha='right', fontsize=8)
    axes[0].set_ylim(0,1.1); axes[0].set_ylabel('Detection Rate')
    axes[0].set_title('Detection Rate by Type'); axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

    # 정상 score 분포: 두 모델 비교
    idx  = np.random.choice(len(normal_A), 300)
    X_n  = torch.tensor(normalize(normal_A[idx], normal_B[idx]))
    F_n  = torch.tensor(norm_feats(normalize(normal_A[idx], normal_B[idx])))
    with torch.no_grad():
        s_feat = feat_m.anomaly_score(X_n, F_n).numpy()
        s_mc   = mc_m.anomaly_score(X_n, F_n).numpy()

    axes[1].hist(s_feat, bins=40, color='#2196F3', alpha=0.7, density=True, label='Single')
    axes[1].axvline(feat_th, color='#2196F3', linestyle='--', linewidth=1.5, label=f'thresh={feat_th:.2f}')
    axes[1].set_title('Normal Score — Single FeatSVDD'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].hist(s_mc, bins=40, color='#FF5722', alpha=0.7, density=True, label='Multi-K2')
    axes[2].axvline(mc_th, color='#FF5722', linestyle='--', linewidth=1.5, label=f'thresh={mc_th:.2f}')
    axes[2].set_title('Normal Score — Multi-center (K=2)'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{ROOT}/multicenter_eval.png', dpi=140, bbox_inches='tight')
    print("\n이미지 저장: multicenter_eval.png")
