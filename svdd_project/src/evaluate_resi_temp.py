"""
ACT.RESI / ACT.TEMP Deep SVDD 평가.

- 정상 프로파일 + inject_anomaly로 생성한 이상 프로파일을 섞은 테스트셋 구성
- ROC-AUC, anomaly_type별 탐지율, 정상/이상 score 분포, 추론 속도 측정

실행:
    python evaluate_resi_temp.py
"""
import sys, os, time, tracemalloc
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report

from model_v2 import FeatSVDD
from resi_temp_features import extract_features, ZONES_RESI, ZONES_TEMP, inject_hold_oscillation
from resi_temp_datagen import (
    RESI_CONFIG, TEMP_CONFIG, ANOMALY_TYPES,
    generate_paired_dataset, inject_anomaly,
)

DEVICE = torch.device('cpu')
ROOT   = os.path.dirname(os.path.dirname(__file__))

norm  = np.load(f'{ROOT}/resi_temp_norm.npy', allow_pickle=True).item()
fnorm = np.load(f'{ROOT}/resi_temp_feat_norm.npy', allow_pickle=True).item()


def normalize(resi, temp):
    r = (resi - norm['resi_min']) / (norm['resi_max'] - norm['resi_min'])
    t = (temp - norm['temp_min']) / (norm['temp_max'] - norm['temp_min'])
    return np.stack([r, t], axis=1).astype(np.float32)


def norm_feats(X):
    return ((extract_features(X) - fnorm['mean']) / fnorm['std']).astype(np.float32)


def load_model():
    ck = torch.load(f'{ROOT}/resi_temp_svdd_model.pt', map_location=DEVICE)
    m = FeatSVDD(ck['latent_dim'], ck['n_feats']).to(DEVICE)
    m.encoder.load_state_dict(ck['encoder'])
    m.c.copy_(ck['c']); m.R.data = torch.tensor(ck['R'])
    m.eval()
    return m, ck['R'] ** 2


@torch.no_grad()
def score_batch(model, resi, temp):
    X = torch.tensor(normalize(resi, temp))
    F = torch.tensor(norm_feats(normalize(resi, temp)))
    return model.anomaly_score(X, F).numpy()


def benchmark_one(model, n_rep=300):
    resi = np.random.uniform(RESI_CONFIG.raw_min, RESI_CONFIG.raw_max, (1, 300)).astype(np.float32)
    temp = np.random.uniform(TEMP_CONFIG.raw_min, TEMP_CONFIG.raw_max, (1, 300)).astype(np.float32)
    X = torch.tensor(normalize(resi, temp))
    F = torch.tensor(norm_feats(normalize(resi, temp)))
    fn = lambda: model.anomaly_score(X, F)
    with torch.no_grad():
        for _ in range(50): fn()
    times = []
    with torch.no_grad():
        for _ in range(n_rep):
            t0 = time.perf_counter(); fn(); times.append((time.perf_counter() - t0) * 1000)
    tracemalloc.start()
    with torch.no_grad(): fn()
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return np.mean(times), np.std(times), peak / 1024, sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    model, threshold = load_model()
    print(f"threshold (R^2) = {threshold:.4f}")

    # ── 테스트셋: 정상 700 + 이상(8종 랜덤) 300 ─────────────────────────────
    N_TEST, ANOMALY_RATIO = 1000, 0.3
    resi, temp, meta = generate_paired_dataset(
        n_profiles=N_TEST,
        anomaly_ratio=ANOMALY_RATIO,
        random_state=999,
    )
    # RESI/TEMP 각각 독립적으로 anomaly가 주입되므로, 둘 중 하나라도 이상이면 라벨 1
    labels = ((meta['resi_is_anomaly'] == 1) | (meta['temp_is_anomaly'] == 1)).astype(int).values

    scores = score_batch(model, resi, temp)
    preds  = (scores > threshold).astype(int)

    auc = roc_auc_score(labels, scores)
    print(f"\n전체 ROC-AUC: {auc:.4f}")
    print(f"이상 비율(실제): {labels.mean():.1%}   이상 비율(예측): {preds.mean():.1%}")
    print("\n" + classification_report(labels, preds, target_names=['정상', '이상']))

    # ── anomaly_type별 탐지율 (RESI, TEMP 각각 별도 주입해서 측정) ──────────
    print("=" * 56)
    print("  Anomaly Type별 탐지율 (RESI 채널에 주입, n=100)")
    print("=" * 56)
    rng = np.random.default_rng(2026)
    det_rates = {}
    normal_resi, normal_temp, _ = generate_paired_dataset(200, anomaly_ratio=0.0, random_state=1)
    for atype in ANOMALY_TYPES:
        idx = rng.choice(len(normal_resi), 100, replace=True)
        base_resi = normal_resi[idx].copy()
        base_temp = normal_temp[idx].copy()
        for i in range(len(base_resi)):
            base_resi[i] = np.clip(
                inject_anomaly(base_resi[i], atype, strength=rng.uniform(0.5, 2.0), rng=rng),
                RESI_CONFIG.raw_min, RESI_CONFIG.raw_max)
        s = score_batch(model, base_resi, base_temp)
        rate = (s > threshold).mean()
        det_rates[atype] = rate
        flag = '✅' if rate >= 0.5 else '❌'
        print(f"  {atype:<20} {flag} {rate:>7.0%}")

    # ── 실측 이상 패턴: hold 구간 진동 (mild=경미한 떨림, strong=뚜렷한 이상) ──
    print("\n" + "=" * 56)
    print("  Hold-region 진동 탐지율 (실제 이상 패턴, n=100)")
    print("=" * 56)
    hold_cases = [
        ('resi_mid_hold',  'resi', ZONES_RESI['mid_hold']),
        ('temp_peak_hold', 'temp', ZONES_TEMP['peak_hold']),
        ('temp_final_hold','temp', ZONES_TEMP['final_hold']),
    ]
    for label, ch, zone in hold_cases:
        for tag, strength_range in [('mild(1-2x)', (1, 2)), ('strong(4-8x)', (4, 8))]:
            idx = rng.choice(len(normal_resi), 100, replace=True)
            base_resi = normal_resi[idx].copy()
            base_temp = normal_temp[idx].copy()
            for i in range(100):
                strength = rng.uniform(*strength_range)
                if ch == 'resi':
                    base_resi[i] = np.clip(inject_hold_oscillation(base_resi[i], zone, strength, rng),
                                            RESI_CONFIG.raw_min, RESI_CONFIG.raw_max)
                else:
                    base_temp[i] = np.clip(inject_hold_oscillation(base_temp[i], zone, strength, rng),
                                            TEMP_CONFIG.raw_min, TEMP_CONFIG.raw_max)
            s = score_batch(model, base_resi, base_temp)
            rate = (s > threshold).mean()
            det_rates[f'{label}_{tag}'] = rate
            flag = '✅' if rate >= 0.5 else '❌'
            print(f"  {label:<18} {tag:<13} {flag} {rate:>7.0%}")

    # ── 추론 속도 ───────────────────────────────────────────────────────────
    mu, sd, mem, params = benchmark_one(model)
    print("\n" + "=" * 56)
    print("  Computational Cost (1 profile, CPU)")
    print("=" * 56)
    print(f"  Inference time : {mu:.3f} ± {sd:.3f} ms")
    print(f"  Peak memory    : {mem:.1f} KB")
    print(f"  Parameters     : {params:,}")

    # ── 시각화 ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Deep SVDD — ACT.RESI / ACT.TEMP (synthetic)', fontsize=13, fontweight='bold')

    axes[0].hist(scores[labels == 0], bins=40, color='#2196F3', alpha=0.7, density=True, label='Normal')
    axes[0].hist(scores[labels == 1], bins=40, color='#FF5722', alpha=0.6, density=True, label='Anomaly')
    axes[0].axvline(threshold, color='k', linestyle='--', linewidth=1.5, label=f'threshold={threshold:.2f}')
    axes[0].set_title(f'Score distribution (ROC-AUC={auc:.3f})')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    all_labels = list(det_rates.keys())
    is_hold = [lbl.startswith(('resi_mid_hold', 'temp_peak_hold', 'temp_final_hold')) for lbl in all_labels]
    colors  = ['#4a3aa7' if h else '#FF5722' for h in is_hold]   # hold-region 진동은 보라색으로 구분
    x = np.arange(len(all_labels))
    axes[1].bar(x, [det_rates[t] for t in all_labels], color=colors, alpha=0.85)
    axes[1].axhline(0.8, color='red', linestyle='--', linewidth=1.2)
    axes[1].set_xticks(x); axes[1].set_xticklabels(all_labels, rotation=45, ha='right', fontsize=7)
    axes[1].set_ylim(0, 1.1); axes[1].set_ylabel('Detection Rate')
    axes[1].set_title('Detection Rate by Anomaly Type'); axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_png = f'{ROOT}/resi_temp_eval.png'
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    print(f"\n이미지 저장: {out_png}")
