"""
ACT.RESI / ACT.TEMP 전용 구간(zone) 정의 및 feature 추출.

기존 model_v2.py의 ZONES는 profile_A/B 템플릿(급격한 스파이크 파형) 기준으로
설계되어 RESI/TEMP의 실제 구조와 맞지 않는다 (flat 구간이 실제로는 가장 크게
변하는 구간, spike 구간이 오히려 완만한 구간 — zones_vs_shape.png 참조).

실측 구조를 바탕으로 채널별로 zone을 다시 정의한다.

TEMP: ramp_up → peak_hold(~10pt) → descent1 → mid_hold → descent2 → final_hold
      실제 이상(작은 떨림)은 peak_hold / final_hold 두 유지 구간에서 발생.

RESI: drop → recover → mid_hold(진짜 flat 구간) → rise → top_hold
      실제 이상은 mid_hold(진짜 flat 구간)부터 기울기가 변해 평균선 위아래로
      굵게 진동하는 패턴으로 나타남.

두 채널의 zone 개수가 달라도 되며(RESI 5개, TEMP 6개), 각 zone마다
[mean, std, diff_std] 3개 feature를 뽑아 concat한다.
"""
import numpy as np

ZONES_RESI = {
    'drop':     (0,   35),
    'recover':  (35,  176),
    'mid_hold': (176, 211),   # 실제 flat/안정 구간 — 이상 시 여기서부터 진동 발생
    'rise':     (211, 263),
    'top_hold': (263, 300),
}

ZONES_TEMP = {
    'ramp_up':    (0,   9),
    'peak_hold':  (9,   19),   # 최고온 도달 후 ~10타점 유지 구간
    'descent1':   (19,  150),
    'mid_hold':   (150, 213),
    'descent2':   (213, 251),
    'final_hold': (251, 300),  # 온도 하강 후 유지 구간 — 작은 떨림 발생
}

N_FEATS = 3 * (len(ZONES_RESI) + len(ZONES_TEMP))   # 3 * (5 + 6) = 33


def extract_features(x_np: np.ndarray) -> np.ndarray:
    """
    x_np: (N, 2, 300)  ch0=RESI, ch1=TEMP  (0-1 min-max 정규화된 상태)
    returns (N, 33) float32
    """
    resi = x_np[:, 0, :]
    temp = x_np[:, 1, :]

    feats = []
    for s, e in ZONES_RESI.values():
        seg = resi[:, s:e]
        feats.append(seg.mean(axis=1))
        feats.append(seg.std(axis=1))
        feats.append(np.diff(seg, axis=1).std(axis=1))
    for s, e in ZONES_TEMP.values():
        seg = temp[:, s:e]
        feats.append(seg.mean(axis=1))
        feats.append(seg.std(axis=1))
        feats.append(np.diff(seg, axis=1).std(axis=1))

    return np.stack(feats, axis=1).astype(np.float32)


def inject_hold_oscillation(profile: np.ndarray, zone: tuple, strength: float,
                             rng: np.random.Generator) -> np.ndarray:
    """
    실제 이상 패턴 재현: hold(flat) 구간에서 기울기가 변하며 평균선 위아래로
    굵게 진동하는 현상. RESI의 mid_hold, TEMP의 peak_hold/final_hold에 사용.

    strength: hold 구간 자체 std 대비 진동 진폭 배수 (1~2=경미한 떨림, 4~8=뚜렷한 이상)
    """
    x = profile.copy()
    s, e = zone
    n = e - s
    local_std = max(x[s:e].std(), 1e-4)
    cycles = rng.uniform(4.0, 10.0)
    osc = np.sin(np.linspace(0.0, 2.0 * np.pi * cycles, n)) * strength * local_std
    x[s:e] += osc
    return x
