import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time, tracemalloc
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, classification_report
from model_v2 import HybridSVDD, FeatSVDD, extract_features, ZONES
from anomaly_generator import (make_anomaly_batch, ANOMALY_TYPES, TEMPLATE_A, TEMPLATE_B,
    _add_excess_noise, _add_spike_anomaly, _add_mean_shift, _add_local_burst, _add_shape_distortion)

DEVICE = torch.device('cpu')
norm   = np.load('/home/imyo/svdd_project/norm_params.npy', allow_pickle=True).item()
fnorm  = np.load('/home/imyo/svdd_project/feat_norm.npy',   allow_pickle=True).item()

def normalize(A, B):
    A = (A - norm['A_min']) / (norm['A_max'] - norm['A_min'])
    B = (B - norm['B_min']) / (norm['B_max'] - norm['B_min'])
    return np.stack([A, B], axis=1).astype(np.float32)

def norm_feats(F):
    return ((F - fnorm['mean']) / fnorm['std']).astype(np.float32)


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_hybrid():
    ck = torch.load('/home/imyo/svdd_project/hybrid_model.pt', map_location=DEVICE)
    m  = HybridSVDD(ck['latent_dim'], alpha=ck['alpha']).to(DEVICE)
    m.encoder.load_state_dict(ck['encoder'])
    m.decoder.load_state_dict(ck['decoder'])
    m.c.copy_(ck['c']); m.R.data = torch.tensor(ck['R']); m.eval()
    return m, ck['R']

def load_feat():
    ck = torch.load('/home/imyo/svdd_project/feat_model.pt', map_location=DEVICE)
    m  = FeatSVDD(ck['latent_dim'], ck['n_feats']).to(DEVICE)
    m.encoder.load_state_dict(ck['encoder'])
    m.c.copy_(ck['c']); m.R.data = torch.tensor(ck['R']); m.eval()
    return m, ck['R']


# ── anomaly 샘플 생성 ─────────────────────────────────────────────────────────
def gen_samples(atype, n=100):
    As, Bs = [], []
    for _ in range(n):
        a = TEMPLATE_A + np.random.normal(0, 0.05, 300)
        b = TEMPLATE_B + np.random.normal(0, 0.05, 300)
        if   atype == 'excess_noise':    a=_add_excess_noise(a); b=_add_excess_noise(b)
        elif atype == 'spike_low':       a=_add_spike_anomaly(a,TEMPLATE_A,'low');   b=_add_spike_anomaly(b,TEMPLATE_B,'low')
        elif atype == 'spike_high':      a=_add_spike_anomaly(a,TEMPLATE_A,'high');  b=_add_spike_anomaly(b,TEMPLATE_B,'high')
        elif atype == 'spike_missing':   a=_add_spike_anomaly(a,TEMPLATE_A,'missing');b=_add_spike_anomaly(b,TEMPLATE_B,'missing')
        elif atype == 'mean_shift':      a=_add_mean_shift(a); b=_add_mean_shift(b)
        elif atype == 'local_burst':     a=_add_local_burst(a); b=_add_local_burst(b)
        elif atype == 'shape_distortion':a=_add_shape_distortion(a,TEMPLATE_A); b=_add_shape_distortion(b,TEMPLATE_B)
        As.append(a); Bs.append(b)
    return np.array(As,dtype=np.float32), np.array(Bs,dtype=np.float32)


# ── 탐지율 계산 ───────────────────────────────────────────────────────────────
@torch.no_grad()
def detection_rate(model, R, atype, is_feat=False):
    As, Bs = gen_samples(atype, 100)
    X   = torch.tensor(normalize(As, Bs))
    if is_feat:
        F = torch.tensor(norm_feats(extract_features(normalize(As, Bs))))
        s = model.anomaly_score(X, F).numpy()
    else:
        s = model.anomaly_score(X).numpy()
    return (s > R**2).mean()


# ── 벤치마크 ─────────────────────────────────────────────────────────────────
def benchmark(model, R, label, is_feat=False, n_repeat=300):
    dummy_A = np.random.randn(1, 300).astype(np.float32)
    dummy_B = np.random.randn(1, 300).astype(np.float32)
    X1  = torch.tensor(normalize(dummy_A, dummy_B))

    if is_feat:
        F1 = torch.tensor(norm_feats(extract_features(normalize(dummy_A, dummy_B))))
        fn = lambda: model.anomaly_score(X1, F1)
    else:
        fn = lambda: model.anomaly_score(X1)

    # warmup
    with torch.no_grad():
        for _ in range(50): fn()

    times = []
    with torch.no_grad():
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter()-t0)*1000)

    # 피크 메모리
    tracemalloc.start()
    with torch.no_grad(): fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    params = sum(p.numel() for p in model.parameters())
    return {
        'label'     : label,
        'mean_ms'   : np.mean(times),
        'std_ms'    : np.std(times),
        'peak_kb'   : peak / 1024,
        'params'    : params,
        'model_mb'  : params * 4 / 1024 / 1024,
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    normal_A = np.loadtxt('/home/imyo/svdd_project/data/profile_A.csv', delimiter=',').astype(np.float32)
    normal_B = np.loadtxt('/home/imyo/svdd_project/data/profile_B.csv', delimiter=',').astype(np.float32)

    hybrid_m, hybrid_R = load_hybrid()
    feat_m,   feat_R   = load_feat()

    print("=" * 60)
    print("  Detection Rate by Anomaly Type")
    print("=" * 60)
    print(f"{'Type':<22} {'Hybrid':>8} {'Feat':>8}")
    print("-" * 42)
    results = {}
    for atype in ANOMALY_TYPES:
        r1 = detection_rate(hybrid_m, hybrid_R, atype, is_feat=False)
        r2 = detection_rate(feat_m,   feat_R,   atype, is_feat=True)
        results[atype] = (r1, r2)
        flag1 = '✅' if r1 >= 0.5 else '❌'
        flag2 = '✅' if r2 >= 0.5 else '❌'
        print(f"  {atype:<20} {flag1}{r1:>5.0%}   {flag2}{r2:>5.0%}")

    print("\n" + "=" * 60)
    print("  Computational Cost (1 profile, CPU)")
    print("=" * 60)
    b1 = benchmark(hybrid_m, hybrid_R, 'Hybrid (SVDD+Recon)',  is_feat=False)
    b2 = benchmark(feat_m,   feat_R,   'Feat (SVDD+Stats)',    is_feat=True)

    for b in [b1, b2]:
        print(f"\n  [{b['label']}]")
        print(f"    Inference time : {b['mean_ms']:.3f} ± {b['std_ms']:.3f} ms")
        print(f"    Peak memory    : {b['peak_kb']:.1f} KB")
        print(f"    Parameters     : {b['params']:,}")
        print(f"    Model size     : {b['model_mb']:.2f} MB")

    # ── 시각화 ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Approach 1 (Hybrid) vs Approach 2 (Feat-augmented)', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 탐지율 비교 bar
    ax1 = fig.add_subplot(gs[0, :2])
    x   = np.arange(len(ANOMALY_TYPES))
    w   = 0.35
    r1s = [results[t][0] for t in ANOMALY_TYPES]
    r2s = [results[t][1] for t in ANOMALY_TYPES]
    ax1.bar(x - w/2, r1s, w, label='Hybrid (SVDD+Recon)', color='#2196F3', alpha=0.8)
    ax1.bar(x + w/2, r2s, w, label='Feat (SVDD+Stats)',   color='#FF9800', alpha=0.8)
    ax1.axhline(0.8, color='red', linestyle='--', linewidth=1.2, label='80% threshold')
    ax1.set_xticks(x); ax1.set_xticklabels(ANOMALY_TYPES, rotation=25, ha='right', fontsize=9)
    ax1.set_ylabel('Detection Rate'); ax1.set_ylim(0, 1.1)
    ax1.set_title('Detection Rate by Anomaly Type'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis='y')

    # 속도 비교
    ax2 = fig.add_subplot(gs[0, 2])
    labels_ = ['Hybrid\n(SVDD+Recon)', 'Feat\n(SVDD+Stats)']
    times_  = [b1['mean_ms'], b2['mean_ms']]
    colors_ = ['#2196F3', '#FF9800']
    bars    = ax2.bar(labels_, times_, color=colors_, alpha=0.8, width=0.5)
    for bar, t in zip(bars, times_):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{t:.3f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Inference Time (ms)'); ax2.set_title('Inference Time (1 profile)'); ax2.grid(True, alpha=0.3, axis='y')

    # excess_noise 프로파일 비교
    As, Bs = gen_samples('excess_noise', 15)
    ax3 = fig.add_subplot(gs[1, 0])
    for i in range(len(As)):
        X   = torch.tensor(normalize(As[i:i+1], Bs[i:i+1]))
        F   = torch.tensor(norm_feats(extract_features(normalize(As[i:i+1], Bs[i:i+1]))))
        s1  = hybrid_m.anomaly_score(X).item()
        s2  = feat_m.anomaly_score(X, F).item()
        d1  = s1 > hybrid_R**2; d2 = s2 > feat_R**2
        c   = '#F44336' if d2 else '#FF9800'
        ax3.plot(As[i], color=c, alpha=0.5, linewidth=0.8)
    ax3.plot(TEMPLATE_A, 'k--', linewidth=1.8)
    ax3.set_title('Excess Noise  (Feat: ✅/❌)', fontsize=10); ax3.grid(True, alpha=0.3)

    # shape_distortion 프로파일 비교
    As, Bs = gen_samples('shape_distortion', 15)
    ax4 = fig.add_subplot(gs[1, 1])
    for i in range(len(As)):
        X   = torch.tensor(normalize(As[i:i+1], Bs[i:i+1]))
        F   = torch.tensor(norm_feats(extract_features(normalize(As[i:i+1], Bs[i:i+1]))))
        s2  = feat_m.anomaly_score(X, F).item()
        d2  = s2 > feat_R**2
        c   = '#F44336' if d2 else '#FF9800'
        ax4.plot(As[i], color=c, alpha=0.5, linewidth=0.8)
    ax4.plot(TEMPLATE_A, 'k--', linewidth=1.8)
    ax4.set_title('Shape Distortion (Feat: ✅/❌)', fontsize=10); ax4.grid(True, alpha=0.3)

    # 메모리/모델크기
    ax5 = fig.add_subplot(gs[1, 2])
    metrics = ['Peak Mem (KB)', 'Model Size (×10 KB)']
    v1 = [b1['peak_kb'], b1['model_mb']*1024/10]
    v2 = [b2['peak_kb'], b2['model_mb']*1024/10]
    x5 = np.arange(len(metrics))
    ax5.bar(x5 - 0.2, v1, 0.35, label='Hybrid', color='#2196F3', alpha=0.8)
    ax5.bar(x5 + 0.2, v2, 0.35, label='Feat',   color='#FF9800', alpha=0.8)
    ax5.set_xticks(x5); ax5.set_xticklabels(metrics, fontsize=9)
    ax5.set_title('Memory & Model Size'); ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3, axis='y')

    plt.savefig('/home/imyo/svdd_project/comparison_v2.png', dpi=140, bbox_inches='tight')
    print("\n비교 이미지 저장: comparison_v2.png")
