import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from model import DeepSVDD
from anomaly_generator import (TEMPLATE_A, TEMPLATE_B,
    _add_excess_noise, _add_spike_anomaly, _add_mean_shift,
    _add_local_burst, _add_shape_distortion, ANOMALY_TYPES)

DEVICE = torch.device('cpu')

# ── 모델 로드 ─────────────────────────────────────────────────────────────────
ckpt  = torch.load('/home/imyo/svdd_project/svdd_model.pt', map_location=DEVICE)
model = DeepSVDD(ckpt['latent_dim']).to(DEVICE)
model.encoder.load_state_dict(ckpt['encoder'])
model.c.copy_(ckpt['c'])
model.R.data = torch.tensor(ckpt['R'])
model.eval()
R2 = ckpt['R'] ** 2

norm = np.load('/home/imyo/svdd_project/norm_params.npy', allow_pickle=True).item()

def normalize(A, B):
    A = (A - norm['A_min']) / (norm['A_max'] - norm['A_min'])
    B = (B - norm['B_min']) / (norm['B_max'] - norm['B_min'])
    return torch.tensor(np.stack([A, B], axis=1).astype(np.float32))

@torch.no_grad()
def get_score(A, B):
    x = normalize(A, B)
    return model.anomaly_score(x).numpy()

# ── anomaly 생성 함수 매핑 ────────────────────────────────────────────────────
def gen_samples(atype, n=30):
    As, Bs = [], []
    for _ in range(n):
        a = TEMPLATE_A + np.random.normal(0, 0.05, 300)
        b = TEMPLATE_B + np.random.normal(0, 0.05, 300)
        if atype == 'excess_noise':
            a = _add_excess_noise(a, std_scale=4.0)
            b = _add_excess_noise(b, std_scale=4.0)
        elif atype == 'spike_low':
            a = _add_spike_anomaly(a, TEMPLATE_A, 'low')
            b = _add_spike_anomaly(b, TEMPLATE_B, 'low')
        elif atype == 'spike_high':
            a = _add_spike_anomaly(a, TEMPLATE_A, 'high')
            b = _add_spike_anomaly(b, TEMPLATE_B, 'high')
        elif atype == 'spike_missing':
            a = _add_spike_anomaly(a, TEMPLATE_A, 'missing')
            b = _add_spike_anomaly(b, TEMPLATE_B, 'missing')
        elif atype == 'mean_shift':
            a = _add_mean_shift(a)
            b = _add_mean_shift(b)
        elif atype == 'local_burst':
            a = _add_local_burst(a)
            b = _add_local_burst(b)
        elif atype == 'shape_distortion':
            a = _add_shape_distortion(a, TEMPLATE_A)
            b = _add_shape_distortion(b, TEMPLATE_B)
        As.append(a); Bs.append(b)
    return np.array(As), np.array(Bs)

# ── 정상 샘플 score 기준 ──────────────────────────────────────────────────────
normal_A = np.loadtxt('/home/imyo/svdd_project/data/profile_A.csv', delimiter=',').astype(np.float32)
normal_B = np.loadtxt('/home/imyo/svdd_project/data/profile_B.csv', delimiter=',').astype(np.float32)
idx = np.random.choice(len(normal_A), 30)
normal_scores = get_score(normal_A[idx], normal_B[idx])

# ── 시각화 ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 28))
fig.suptitle('Anomaly Detection Results by Type', fontsize=16, fontweight='bold', y=0.98)

N_TYPES = len(ANOMALY_TYPES)
# 각 anomaly type: Profile A, Profile B, Score 분포 → 3열
# + 맨 위에 Normal 참조

# 행: Normal(1) + 7 types = 8행, 열: A / B / Score = 3열
n_rows = N_TYPES + 1
gs = fig.add_gridspec(n_rows, 3, hspace=0.55, wspace=0.35)

# 색상
C_NORMAL  = '#2196F3'
C_DETECT  = '#F44336'
C_MISS    = '#FF9800'
C_TMPL    = '#212121'

LABELS = {
    'excess_noise'     : 'Excess Noise\n(flat 구간 노이즈 과다)',
    'spike_low'        : 'Spike Low\n(spike 낮음)',
    'spike_high'       : 'Spike High\n(spike 높음)',
    'spike_missing'    : 'Spike Missing\n(spike 없음)',
    'mean_shift'       : 'Mean Shift\n(전체 수준 이동)',
    'local_burst'      : 'Local Burst\n(변곡점 noise 폭발)',
    'shape_distortion' : 'Shape Distortion\n(전환 구간 왜곡)',
}

# ── Normal 참조 행 ────────────────────────────────────────────────────────────
ax_na = fig.add_subplot(gs[0, 0])
ax_nb = fig.add_subplot(gs[0, 1])
ax_ns = fig.add_subplot(gs[0, 2])

for i in range(min(20, len(idx))):
    ax_na.plot(normal_A[idx[i]], color=C_NORMAL, alpha=0.4, linewidth=0.8)
    ax_nb.plot(normal_B[idx[i]], color=C_NORMAL, alpha=0.4, linewidth=0.8)
ax_na.plot(TEMPLATE_A, color=C_TMPL, linewidth=2.0, linestyle='--', label='Template')
ax_nb.plot(TEMPLATE_B, color=C_TMPL, linewidth=2.0, linestyle='--')
ax_na.set_title('Normal  –  Profile A', fontsize=10, fontweight='bold')
ax_nb.set_title('Normal  –  Profile B', fontsize=10, fontweight='bold')
ax_ns.set_title('Score Distribution', fontsize=10, fontweight='bold')
ax_ns.hist(normal_scores, bins=20, color=C_NORMAL, alpha=0.8, density=True)
ax_ns.axvline(R2, color='black', linestyle='--', linewidth=1.5, label=f'R²={R2:.3f}')
ax_ns.set_xlabel('Anomaly Score')
ax_ns.legend(fontsize=8)
for ax in [ax_na, ax_nb]: ax.grid(True, alpha=0.3); ax.set_ylabel('Value')

# ── Anomaly type별 행 ─────────────────────────────────────────────────────────
for row, atype in enumerate(ANOMALY_TYPES, start=1):
    As, Bs = gen_samples(atype, n=30)
    scores  = get_score(As, Bs)
    det_rate = (scores > R2).mean()
    detected = det_rate >= 0.5
    color    = C_DETECT if detected else C_MISS

    ax_a = fig.add_subplot(gs[row, 0])
    ax_b = fig.add_subplot(gs[row, 1])
    ax_s = fig.add_subplot(gs[row, 2])

    for i in range(len(As)):
        c = C_DETECT if scores[i] > R2 else C_MISS
        ax_a.plot(As[i], color=c, alpha=0.35, linewidth=0.8)
        ax_b.plot(Bs[i], color=c, alpha=0.35, linewidth=0.8)
    ax_a.plot(TEMPLATE_A, color=C_TMPL, linewidth=1.8, linestyle='--')
    ax_b.plot(TEMPLATE_B, color=C_TMPL, linewidth=1.8, linestyle='--')

    status = '✅ DETECTED' if detected else '❌ MISSED'
    label  = LABELS[atype]
    ax_a.set_title(f'{label}  {status}', fontsize=9, fontweight='bold', color=color)

    ax_s.hist(normal_scores, bins=15, color=C_NORMAL, alpha=0.5, density=True, label='Normal')
    ax_s.hist(scores,        bins=15, color=color,    alpha=0.7, density=True, label='Anomaly')
    ax_s.axvline(R2, color='black', linestyle='--', linewidth=1.5, label=f'R²={R2:.3f}')
    ax_s.set_xlabel('Anomaly Score')
    ax_s.set_title(f'Detection: {det_rate:.0%}', fontsize=9)
    ax_s.legend(fontsize=7)

    for ax in [ax_a, ax_b, ax_s]: ax.grid(True, alpha=0.3)
    for ax in [ax_a, ax_b]: ax.set_ylabel('Value')

# 범례
patches = [
    mpatches.Patch(color=C_NORMAL,  label='Normal'),
    mpatches.Patch(color=C_DETECT,  label='Detected (score > R²)'),
    mpatches.Patch(color=C_MISS,    label='Missed (score ≤ R²)'),
    mpatches.Patch(color=C_TMPL,    label='Template'),
]
fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, 0.005), frameon=True)

plt.savefig('/home/imyo/svdd_project/anomaly_detection_results.png',
            dpi=130, bbox_inches='tight')
print("저장 완료: anomaly_detection_results.png")
