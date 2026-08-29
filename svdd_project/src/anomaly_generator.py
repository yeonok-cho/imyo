"""
Anomaly data generator for Profile A and B.

Anomaly types:
  1. excess_noise     : flat/완만 구간 전체에 과도한 noise
  2. spike_anomaly    : spike 높이 이상 (너무 높거나 낮거나 없음)
  3. mean_shift       : 전체 프로파일 수준 이동 (drift 초과)
  4. local_burst      : 특정 구간에서 noise 폭발 (변곡점)
  5. shape_distortion : 전환 구간 기울기 변형
"""

import numpy as np
from scipy.interpolate import interp1d

np.random.seed(99)

# ── 정상 template 로드 ─────────────────────────────────────────────────────────
def _make_template_A():
    data = np.array([5.08,5.07,4.98,4.89,4.91,4.87,4.89,4.82,4.79,4.79,4.77,4.8,4.63,4.64,4.6,4.62,4.62,4.56,4.52,4.49,4.51,4.51,4.39,4.35,4.36,4.33,4.35,4.18,4.15,4.17,4.13,4.17,4.06,4.07,4.08,3.89,3.9,3.89,3.91,3.83,3.82,3.79,3.81,3.8,3.68,3.64,3.62,3.65,3.63,3.52,3.46,3.41,3.48,3.45,3.45,3.36,3.42,3.39,3.4,3.41,3.43,3.41,3.41,3.44,3.39,3.41,3.39,3.41,3.41,3.43,3.43,3.43,3.43,3.46,3.41,3.37,3.4,3.43,3.41,3.4,3.43,3.41,3.4,3.43,3.41,3.4,3.43,3.41,3.42,3.42,3.44,3.37,3.41,3.39,3.42,3.42,3.42,3.45,3.46,3.46,3.49,3.46,3.49,3.45,3.45,3.47,3.45,3.41,3.38,3.43,3.38,3.43,3.38,3.41,3.477,3.5,3.47,3.46,3.51,3.53,3.55,3.55,3.55,3.68,3.74,3.73,3.74,3.75,3.79,3.85,3.83,3.88,3.84,4.04,4.1,4.1,4.1,4.1,4.54,4.51,4.55,4.54,4.54,4.66,4.64,4.65,4.64,4.65,4.75,4.75,4.75,4.75,4.75,4.73,4.72,4.72,4.74,4.72,4.72,4.75,1.39,0.0076,0,0,0.025,0.015,0.025,0.027,0,0.0015,0,0,0.0015,0,0,0.012,0.0091,0,0.019,0,0,0,0,0,9.99,10.09,10.04,10.094,10.03,7.6,7.55,7.57,7.58,7.56,5.41,5.42,5.4,5.39,4.65,4.27,4.3,4,3.86,3.88,3.85,3.84,3.81,3.79,3.82,3.79,3.8,3.88,3.92,3.91,3.89,3.88,3.97,3.97,3.94,3.97,3.95,3.95,3.95,3.93,3.93,4,3.97,4.01,4,4,4,3.98,4.01,3.98,3.97,3.95,3.95,3.95,3.93,4,4,3.98])
    return interp1d(np.linspace(0,1,len(data)), data, kind='linear')(np.linspace(0,1,300))

def _make_template_B():
    ctrl_x = [  0,  35,  80, 130, 160, 188, 195, 203, 227, 232, 238, 252, 272, 299]
    ctrl_y = [3.0, 2.5, 1.9, 1.6, 1.5, 4.0, 6.8, 0.0, 0.0, 6.5, 7.0, 4.0, 2.0, 2.0]
    t = interp1d(ctrl_x, ctrl_y, kind='cubic')(np.arange(300))
    return np.clip(t, 0, None)

TEMPLATE_A = _make_template_A()
TEMPLATE_B = _make_template_B()

# ── 구간 정의 (Profile A 기준, B도 유사) ──────────────────────────────────────
ZONES = {
    'flat'       : (35,  160),   # flat 구간
    'transition' : (160, 200),   # 상승 전환
    'zero'       : (200, 215),   # 0 근처
    'spike'      : (215, 245),   # spike
    'tail'       : (245, 300),   # 마지막 flat
}

# ── Anomaly 생성 함수 ──────────────────────────────────────────────────────────
def _add_excess_noise(series, std_scale=4.0):
    """flat 구간 전체에 과도한 noise"""
    s, e = ZONES['flat']
    noise = np.random.normal(0, np.std(series[s:e]) * std_scale, e - s)
    out = series.copy()
    out[s:e] += noise
    return out

def _add_spike_anomaly(series, template, mode='low'):
    """spike 높이 이상: low=낮게, high=더높게, missing=없음"""
    s, e = ZONES['spike']
    out = series.copy()
    if mode == 'low':
        out[s:e] = template[s:e] * np.random.uniform(0.3, 0.55)
    elif mode == 'high':
        out[s:e] = template[s:e] * np.random.uniform(1.4, 1.8)
    elif mode == 'missing':
        out[s:e] = np.random.normal(0, 0.05, e - s)
    return out

def _add_mean_shift(series, shift=None):
    """전체 수준 이동 (노후화 초과)"""
    if shift is None:
        shift = np.random.choice([-1, 1]) * np.random.uniform(0.8, 1.5)
    return np.clip(series + shift, 0, None)

def _add_local_burst(series, std_scale=6.0):
    """변곡점 구간에서 noise 폭발"""
    s, e = ZONES['transition']
    noise = np.random.normal(0, std_scale, e - s)
    out = series.copy()
    out[s:e] += noise
    return out

def _add_shape_distortion(series, template):
    """전환 구간 기울기 왜곡"""
    s, e = ZONES['transition']
    out = series.copy()
    stretch = np.random.uniform(0.4, 0.7)
    distorted = template[s:e] * stretch + np.random.normal(0, 0.2, e - s)
    out[s:e] = distorted
    return out

# ── 배치 생성 ─────────────────────────────────────────────────────────────────
ANOMALY_TYPES = ['excess_noise', 'spike_low', 'spike_high', 'spike_missing',
                 'mean_shift', 'local_burst', 'shape_distortion']

def make_anomaly_batch(normal_A, normal_B, n=500, anomaly_ratio=0.3):
    """
    정상 배치에서 anomaly_ratio 비율만큼 이상 샘플로 교체한 배치 반환
    Returns: (batch_A, batch_B, labels)  labels: 0=정상, 1=이상
    """
    n_anom   = int(n * anomaly_ratio)
    n_normal = n - n_anom

    # 정상 샘플
    idx_A = np.random.choice(len(normal_A), n_normal, replace=True)
    idx_B = np.random.choice(len(normal_B), n_normal, replace=True)
    batch_A = normal_A[idx_A].copy()
    batch_B = normal_B[idx_B].copy()
    labels  = np.zeros(n_normal)

    # 이상 샘플
    anom_types = np.random.choice(ANOMALY_TYPES, n_anom)
    for atype in anom_types:
        base_A = TEMPLATE_A + np.random.normal(0, 0.05, 300)
        base_B = TEMPLATE_B + np.random.normal(0, 0.05, 300)

        if atype == 'excess_noise':
            a = _add_excess_noise(base_A)
            b = _add_excess_noise(base_B)
        elif atype == 'spike_low':
            a = _add_spike_anomaly(base_A, TEMPLATE_A, 'low')
            b = _add_spike_anomaly(base_B, TEMPLATE_B, 'low')
        elif atype == 'spike_high':
            a = _add_spike_anomaly(base_A, TEMPLATE_A, 'high')
            b = _add_spike_anomaly(base_B, TEMPLATE_B, 'high')
        elif atype == 'spike_missing':
            a = _add_spike_anomaly(base_A, TEMPLATE_A, 'missing')
            b = _add_spike_anomaly(base_B, TEMPLATE_B, 'missing')
        elif atype == 'mean_shift':
            shift = np.random.choice([-1,1]) * np.random.uniform(0.8, 1.5)
            a = _add_mean_shift(base_A, shift)
            b = _add_mean_shift(base_B, shift)
        elif atype == 'local_burst':
            a = _add_local_burst(base_A)
            b = _add_local_burst(base_B)
        elif atype == 'shape_distortion':
            a = _add_shape_distortion(base_A, TEMPLATE_A)
            b = _add_shape_distortion(base_B, TEMPLATE_B)

        batch_A = np.vstack([batch_A, a])
        batch_B = np.vstack([batch_B, b])
        labels  = np.append(labels, 1)

    # shuffle
    perm    = np.random.permutation(n)
    return batch_A[perm], batch_B[perm], labels[perm].astype(int)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    normal_A = np.loadtxt('/home/imyo/svdd_project/data/profile_A.csv', delimiter=',')
    normal_B = np.loadtxt('/home/imyo/svdd_project/data/profile_B.csv', delimiter=',')

    batch_A, batch_B, labels = make_anomaly_batch(normal_A, normal_B, n=500, anomaly_ratio=0.3)
    print(f"배치 shape: A={batch_A.shape}, B={batch_B.shape}")
    print(f"정상: {(labels==0).sum()}개, 이상: {(labels==1).sum()}개")

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    for ax, (title, idx) in zip(axes.flatten(), [
        ('Normal A',  np.where(labels==0)[0][:20]),
        ('Anomaly A', np.where(labels==1)[0][:20]),
        ('Normal B',  np.where(labels==0)[0][:20]),
        ('Anomaly B', np.where(labels==1)[0][:20]),
    ]):
        col = 'steelblue' if 'Normal' in title else 'tomato'
        data = batch_A if ' A' in title else batch_B
        for i in idx:
            ax.plot(data[i], color=col, alpha=0.4, linewidth=0.8)
        ax.set_title(title); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/imyo/svdd_project/data/anomaly_preview.png', dpi=150, bbox_inches='tight')
    print("미리보기 저장: anomaly_preview.png")
