"""Causal robust reference-baseline construction and feature normalization."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import DetectionConfig


RESI_FEATURES = ("MEAN", "STD", "MAX", "SD2_COUNT", "SD3_COUNT", "SD5_COUNT", "SD9_COUNT")
TEMP_FEATURES = ("TEMP_MEAN", "TEMP_MAX", "TEMP_P99_HIGH_COUNT", "TEMP_P99_HIGH_SHARE")


def calculate_baseline_statistics(values: Iterable[float], eps: float = 1e-8) -> dict[str, float]:
    """Return robust descriptive statistics for one feature's reference history."""
    clean = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if not len(clean):
        return {key: np.nan for key in ("median", "mad", "min", "max", "p90", "p95", "p99", "mean", "std", "count")}
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    return {
        "median": median,
        "mad": mad,
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "p90": float(np.quantile(clean, 0.90)),
        "p95": float(np.quantile(clean, 0.95)),
        "p99": float(np.quantile(clean, 0.99)),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "count": float(len(clean)),
    }


def _candidate_keys(row: object, has_family: bool) -> list[tuple[str, tuple[object, ...]]]:
    eqp_side = (row.EQP_ID, row.SIDE)
    keys = [("EQP_SIDE_PRODUCT", (*eqp_side, row.PRODUCT)), ("EQP_SIDE", eqp_side)]
    if has_family and pd.notna(row.EQP_FAMILY):
        family_side = (row.EQP_FAMILY, row.SIDE)
        keys.extend([
            ("FAMILY_SIDE_PRODUCT", (*family_side, row.PRODUCT)),
            ("FAMILY_SIDE", family_side),
        ])
    return keys


def build_reference_baseline(
    side_level: pd.DataFrame,
    features: Iterable[str],
    config: DetectionConfig,
    excluded_timestamps: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Build a bounded, strictly historical baseline for each side-level record.

    Histories are updated *after* emitting the current record, preventing
    look-ahead leakage. The bounded history prevents repeated full-history
    scans for long production series while retaining a configurable reference
    window.
    """
    feature_list = [feature for feature in features if feature in side_level.columns]
    required = {"EQP_ID", "SIDE", "PRODUCT", "TIMESTAMP"}
    missing = required - set(side_level.columns)
    if missing:
        raise ValueError(f"Cannot build baseline without {sorted(missing)}")
    frame = side_level.copy().sort_values(["TIMESTAMP", "EQP_ID", "SIDE"], kind="stable").reset_index()
    excluded = set(pd.to_datetime(list(excluded_timestamps or []), errors="coerce"))
    has_family = "EQP_FAMILY" in frame.columns
    histories: dict[tuple[str, tuple[object, ...]], deque[dict[str, float]]] = defaultdict(
        lambda: deque(maxlen=config.baseline_max_history)
    )
    baseline_rows: list[dict[str, object]] = []

    for row in frame.itertuples(index=False):
        candidates = _candidate_keys(row, has_family)
        selected_level, selected_history = "NONE", deque()
        for level, key in candidates:
            history = histories[(level, key)]
            if len(history) >= config.baseline_min_wafers:
                selected_level, selected_history = level, history
                break
        baseline: dict[str, object] = {"_ROW_INDEX": row.index, "BASELINE_LEVEL": selected_level}
        for feature in feature_list:
            stats = calculate_baseline_statistics((item.get(feature, np.nan) for item in selected_history), config.eps)
            for stat, value in stats.items():
                baseline[f"{feature}_BASELINE_{stat.upper()}"] = value
        baseline_rows.append(baseline)

        # Baseline exclusions prevent reference contamination but do not remove
        # the wafer from detection output.
        if pd.notna(row.TIMESTAMP) and row.TIMESTAMP not in excluded:
            values = {feature: getattr(row, feature) for feature in feature_list}
            for level, key in candidates:
                histories[(level, key)].append(values)

    baseline_frame = pd.DataFrame(baseline_rows).set_index("_ROW_INDEX")
    return side_level.join(baseline_frame, how="left")


def normalize_features(
    side_level: pd.DataFrame,
    features: Iterable[str],
    config: DetectionConfig,
    excluded_timestamps: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Add robust z-scores and own-side baseline ratios for available features."""
    output = build_reference_baseline(side_level, features, config, excluded_timestamps)
    for feature in features:
        if feature not in output:
            continue
        median = output[f"{feature}_BASELINE_MEDIAN"]
        mad = output[f"{feature}_BASELINE_MAD"]
        scale = 1.4826 * mad + config.eps
        z_score = (output[feature] - median) / scale
        # Quantized features can have MAD=0 even though two or more observed
        # values are normal. Keep any value inside the historical observed
        # range neutral; a value outside it retains a large, finite robust-z.
        zero_mad = mad.le(config.eps)
        within_observed_range = output[feature].between(
            output[f"{feature}_BASELINE_MIN"], output[f"{feature}_BASELINE_MAX"], inclusive="both"
        )
        output[f"{feature}_Z"] = z_score.mask(zero_mad & within_observed_range, 0.0).replace([np.inf, -np.inf], np.nan)
        output[f"{feature}_RATIO_BASELINE"] = (output[feature] / (median.abs() + config.eps)).replace(
            [np.inf, -np.inf], np.nan
        )
    return output


def build_resi_baseline(side_level: pd.DataFrame, config: DetectionConfig, **kwargs: object) -> pd.DataFrame:
    return normalize_features(side_level, RESI_FEATURES, config, **kwargs)


def build_temp_baseline(side_level: pd.DataFrame, config: DetectionConfig, **kwargs: object) -> pd.DataFrame:
    return normalize_features(side_level, TEMP_FEATURES, config, **kwargs)
