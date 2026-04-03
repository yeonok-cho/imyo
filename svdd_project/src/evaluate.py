"""
모델 평가 - 정상/이상 배치 테스트
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from model import DeepSVDD
from anomaly_generator import make_anomaly_batch, ANOMALY_TYPES

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model():
    ckpt = torch.load('/home/imyo/svdd_project/svdd_model.pt', map_location=DEVICE)
    model = DeepSVDD(ckpt['latent_dim']).to(DEVICE)
    model.encoder.load_state_dict(ckpt['encoder'])
    model.c.copy_(ckpt['c'])
    model.R.data = torch.tensor(ckpt['R'])
    model.eval()
    return model, ckpt['R']


_norm = np.load('/home/imyo/svdd_project/norm_params.npy', allow_pickle=True).item()

def normalize(A, B):
    A = (A - _norm['A_min']) / (_norm['A_max'] - _norm['A_min'])
    B = (B - _norm['B_min']) / (_norm['B_max'] - _norm['B_min'])
    return np.stack([A, B], axis=1).astype(np.float32)


@torch.no_grad()
def score_batch(model, X_np):
    X = torch.tensor(X_np).to(DEVICE)
    return model.anomaly_score(X).cpu().numpy()


def evaluate():
    model, R = load_model()
    print(f"모델 로드 완료 | R = {R:.4f}")

    normal_A = np.loadtxt('/home/imyo/svdd_project/data/profile_A.csv', delimiter=',').astype(np.float32)
    normal_B = np.loadtxt('/home/imyo/svdd_project/data/profile_B.csv', delimiter=',').astype(np.float32)

    # ── 테스트 배치 생성 ─────────────────────────────────────────────────────
    print("\n테스트 배치 생성 중...")
    batch_A, batch_B, labels = make_anomaly_batch(normal_A, normal_B, n=500, anomaly_ratio=0.3)
    X_np = normalize(batch_A, batch_B)

    scores = score_batch(model, X_np)
    preds  = (scores > R**2).astype(int)

    print(f"\n── 개별 샘플 판정 ──────────────────────────")
    print(classification_report(labels, preds, target_names=['Normal', 'Anomaly']))
    print(f"ROC-AUC: {roc_auc_score(labels, scores):.4f}")

    # ── 배치 레벨 판정 ───────────────────────────────────────────────────────
    anom_ratio = preds.mean()
    batch_alert = anom_ratio > 0.1
    print(f"\n── 배치 레벨 판정 ──────────────────────────")
    print(f"이상 비율: {anom_ratio:.1%}  →  배치 경보: {'🚨 이상' if batch_alert else '✅ 정상'}")

    # ── Anomaly type별 탐지율 ────────────────────────────────────────────────
    print(f"\n── Anomaly Type별 탐지율 ──────────────────")
    anom_A, anom_B, anom_labels = make_anomaly_batch(normal_A, normal_B, n=500, anomaly_ratio=1.0)

    # anomaly type 순서대로 50개씩 생성해서 개별 확인
    from anomaly_generator import TEMPLATE_A, TEMPLATE_B, _add_excess_noise, _add_spike_anomaly, \
        _add_mean_shift, _add_local_burst, _add_shape_distortion

    type_results = {}
    for atype in ANOMALY_TYPES:
        samples_A, samples_B = [], []
        for _ in range(100):
            a = TEMPLATE_A + np.random.normal(0, 0.05, 300)
            b = TEMPLATE_B + np.random.normal(0, 0.05, 300)
            if atype == 'excess_noise':
                a = _add_excess_noise(a); b = _add_excess_noise(b)
            elif atype == 'spike_low':
                a = _add_spike_anomaly(a, TEMPLATE_A, 'low'); b = _add_spike_anomaly(b, TEMPLATE_B, 'low')
            elif atype == 'spike_high':
                a = _add_spike_anomaly(a, TEMPLATE_A, 'high'); b = _add_spike_anomaly(b, TEMPLATE_B, 'high')
            elif atype == 'spike_missing':
                a = _add_spike_anomaly(a, TEMPLATE_A, 'missing'); b = _add_spike_anomaly(b, TEMPLATE_B, 'missing')
            elif atype == 'mean_shift':
                a = _add_mean_shift(a); b = _add_mean_shift(b)
            elif atype == 'local_burst':
                a = _add_local_burst(a); b = _add_local_burst(b)
            elif atype == 'shape_distortion':
                a = _add_shape_distortion(a, TEMPLATE_A); b = _add_shape_distortion(b, TEMPLATE_B)
            samples_A.append(a); samples_B.append(b)

        X_type = normalize(np.array(samples_A, dtype=np.float32), np.array(samples_B, dtype=np.float32))
        s = score_batch(model, X_type)
        detection_rate = (s > R**2).mean()
        type_results[atype] = detection_rate
        print(f"  {atype:<20s}: {detection_rate:.1%}")

    # ── 시각화 ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) score 분포
    normal_scores = scores[labels == 0]
    anom_scores   = scores[labels == 1]
    axes[0].hist(normal_scores, bins=40, alpha=0.6, color='steelblue', label='Normal', density=True)
    axes[0].hist(anom_scores,   bins=40, alpha=0.6, color='tomato',    label='Anomaly', density=True)
    axes[0].axvline(R**2, color='black', linestyle='--', label=f'R² = {R**2:.3f}')
    axes[0].set_title('Anomaly Score Distribution'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # 2) confusion matrix
    cm = confusion_matrix(labels, preds)
    im = axes[1].imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, cm[i,j], ha='center', va='center', fontsize=14, fontweight='bold')
    axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
    axes[1].set_xticklabels(['Pred Normal','Pred Anomaly'])
    axes[1].set_yticklabels(['True Normal','True Anomaly'])
    axes[1].set_title('Confusion Matrix')

    # 3) type별 탐지율
    axes[2].barh(list(type_results.keys()), list(type_results.values()), color='steelblue', alpha=0.8)
    axes[2].axvline(0.8, color='tomato', linestyle='--', label='80% threshold')
    axes[2].set_xlim(0, 1); axes[2].set_xlabel('Detection Rate')
    axes[2].set_title('Detection Rate by Anomaly Type')
    axes[2].legend(); axes[2].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('/home/imyo/svdd_project/evaluation.png', dpi=150, bbox_inches='tight')
    print("\n평가 결과 저장: evaluation.png")


if __name__ == '__main__':
    evaluate()
