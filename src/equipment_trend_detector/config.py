"""One place for the initial, intentionally uncalibrated detector defaults."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionConfig:
    min_side_n: int = 30
    baseline_min_wafers: int = 20
    baseline_max_history: int = 250
    eps: float = 1e-8

    short_window: int = 3
    trend_window: int = 5
    context_window: int = 10
    feature_z_elevated: float = 2.5
    feature_z_high: float = 3.5
    feature_z_extreme: float = 5.0
    trend_warning_min_abnormal: int = 3
    trend_alarm_min_abnormal: int = 4
    step_delta_z: float = 1.0
    drift_min_slope: float = 0.35
    ewma_alpha: float = 0.30
    cusum_k: float = 0.50
    cusum_h: float = 5.0
