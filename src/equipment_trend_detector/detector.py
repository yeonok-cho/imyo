"""Wafer evidence and 3/5/10-window equipment-side trend evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .config import DetectionConfig


def _state(score: float, config: DetectionConfig) -> str:
    if pd.isna(score):
        return "NORMAL"
    if score >= config.feature_z_extreme:
        return "EXTREME"
    if score >= config.feature_z_high:
        return "HIGH"
    if score >= config.feature_z_elevated:
        return "ELEVATED"
    return "NORMAL"


def classify_feature_groups(
    normalized: pd.DataFrame,
    config: DetectionConfig,
    groups: Mapping[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """Classify conceptual groups, preserving feature independence within each group."""
    groups = groups or {
        "LEVEL": ("MEAN_Z",),
        "SPREAD": ("STD_Z",),
        "MAXIMUM_EXCEEDANCE": ("MAX_Z", "SD3_COUNT_Z", "SD5_COUNT_Z", "SD9_COUNT_Z"),
    }
    output = normalized.copy()
    for group, columns in groups.items():
        available = [column for column in columns if column in output]
        if not available:
            output[f"{group}_SCORE"] = np.nan
            output[f"{group}_STATE"] = "NORMAL"
            continue
        values = output[available]
        # Level movement can be positive or negative. Spread and maximum/
        # threshold-exceedance increases are one-sided positive shifts.
        score = values.abs().max(axis=1) if group == "LEVEL" else values.max(axis=1)
        output[f"{group}_SCORE"] = score
        output[f"{group}_STATE"] = score.map(lambda value: _state(value, config))
    return output


def classify_wafer_side(normalized: pd.DataFrame, config: DetectionConfig) -> pd.DataFrame:
    """Turn per-wafer group evidence into non-notifying wafer-side status."""
    output = classify_feature_groups(normalized, config)
    states = output[["LEVEL_STATE", "SPREAD_STATE", "MAXIMUM_EXCEEDANCE_STATE"]]
    abnormal_groups = states.ne("NORMAL").sum(axis=1)
    extreme_groups = states.eq("EXTREME").sum(axis=1)
    high_groups = states.isin(["HIGH", "EXTREME"]).sum(axis=1)
    valid_n = output["N"].ge(config.min_side_n) if "N" in output else pd.Series(True, index=output.index)
    lr_state = output.get("LR_CONTRAST_FOR_SIDE_STATE", pd.Series("NORMAL", index=output.index))
    lr_abnormal = lr_state.ne("NORMAL")
    lr_high = lr_state.isin(["HIGH", "EXTREME"])
    output["WAFER_SIDE_STATUS"] = np.select(
        [
            ~valid_n,
            extreme_groups >= 2,
            high_groups >= 2,
            (abnormal_groups >= 1) & lr_high,
            abnormal_groups >= 2,
            abnormal_groups == 1,
            lr_abnormal,
        ],
        ["INSUFFICIENT_DATA", "ALARM", "ALARM", "WARNING", "WARNING", "CHECK", "CHECK"],
        default="NORMAL",
    )
    output["WAFER_SIDE_REASON"] = [
        "; ".join(
            [
                f"{group}={row[f'{group}_STATE']}"
                for group in ("LEVEL", "SPREAD", "MAXIMUM_EXCEEDANCE")
                if row[f"{group}_STATE"] != "NORMAL"
            ]
            + ([f"LR_CONTRAST={row['LR_CONTRAST_FOR_SIDE_STATE']}"] if row.get("LR_CONTRAST_FOR_SIDE_STATE", "NORMAL") != "NORMAL" else [])
        )
        or "No baseline-normalized feature-group change"
        for _, row in output.iterrows()
    ]
    return output


def classify_temp_wafer_side(normalized: pd.DataFrame, config: DetectionConfig) -> pd.DataFrame:
    """TEMP evidence avoids using TEMP standard deviation as a primary feature."""
    groups = {
        "LEVEL": ("TEMP_MEAN_Z",),
        "SPREAD": tuple(),
        "MAXIMUM_EXCEEDANCE": ("TEMP_MAX_Z", "TEMP_P99_HIGH_COUNT_Z", "TEMP_P99_HIGH_SHARE_Z"),
    }
    output = classify_feature_groups(normalized, config, groups)
    states = output[["LEVEL_STATE", "MAXIMUM_EXCEEDANCE_STATE"]]
    abnormal_groups = states.ne("NORMAL").sum(axis=1)
    high_groups = states.isin(["HIGH", "EXTREME"]).sum(axis=1)
    valid_n = output["N"].ge(config.min_side_n) if "N" in output else pd.Series(True, index=output.index)
    output["WAFER_SIDE_STATUS"] = np.select(
        [~valid_n, high_groups >= 2, abnormal_groups >= 2, abnormal_groups == 1],
        ["INSUFFICIENT_DATA", "ALARM", "WARNING", "CHECK"],
        default="NORMAL",
    )
    output["WAFER_SIDE_REASON"] = [
        "; ".join(
            f"{group}={row[f'{group}_STATE']}"
            for group in ("LEVEL", "MAXIMUM_EXCEEDANCE")
            if row[f"{group}_STATE"] != "NORMAL"
        )
        or "No TEMP baseline-normalized feature-group change"
        for _, row in output.iterrows()
    ]
    return output


def calculate_trend_slope(values: Iterable[float]) -> float:
    clean = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if len(clean) < 2:
        return np.nan
    return float(np.polyfit(np.arange(len(clean)), clean, 1)[0])


def calculate_ewma(values: Iterable[float], alpha: float) -> float:
    result = np.nan
    for value in values:
        if pd.isna(value):
            continue
        result = value if pd.isna(result) else alpha * value + (1 - alpha) * result
    return float(result) if pd.notna(result) else np.nan


def calculate_cusum(values: Iterable[float], k: float) -> float:
    total = 0.0
    for value in values:
        if pd.notna(value):
            total = max(0.0, total + float(value) - k)
    return total


def calculate_window_statistics(
    history: pd.DataFrame,
    config: DetectionConfig,
    z_columns: tuple[str, str, str] = ("MEAN_Z", "STD_Z", "MAX_Z"),
) -> dict[str, object]:
    """Compute short, decision, and context window statistics for one side."""
    valid = history.loc[history["WAFER_SIDE_STATUS"].ne("INSUFFICIENT_DATA")].copy()
    recent = valid.tail(config.trend_window)
    short = valid.tail(config.short_window)
    context = valid.tail(config.context_window)
    abnormal = recent["WAFER_SIDE_STATUS"].isin(["CHECK", "WARNING", "ALARM"])
    result: dict[str, object] = {
        "RECENT_VALID_WAFERS": len(recent),
        "SHORT_ABNORMAL_COUNT": int(short["WAFER_SIDE_STATUS"].isin(["CHECK", "WARNING", "ALARM"]).sum()),
        "SHORT_WARNING_COUNT": int(short["WAFER_SIDE_STATUS"].isin(["WARNING", "ALARM"]).sum()),
        "SHORT_ALARM_COUNT": int(short["WAFER_SIDE_STATUS"].eq("ALARM").sum()),
        "ABNORMAL_COUNT_5": int(abnormal.sum()),
        "WARNING_COUNT_5": int(recent["WAFER_SIDE_STATUS"].isin(["WARNING", "ALARM"]).sum()),
        "ALARM_COUNT_5": int(recent["WAFER_SIDE_STATUS"].eq("ALARM").sum()),
        "PREVIOUS_ABNORMAL_COUNT_5": int(
            context.iloc[: config.trend_window]["WAFER_SIDE_STATUS"].isin(["CHECK", "WARNING", "ALARM"]).sum()
        ) if len(context) >= config.context_window else 0,
    }
    for group in ("LEVEL", "SPREAD", "MAXIMUM_EXCEEDANCE"):
        column = f"{group}_STATE"
        result[f"{group}_ABNORMAL_SHARE"] = float(recent[column].ne("NORMAL").mean()) if column in recent else 0.0
    lr_column = "LR_CONTRAST_FOR_SIDE_STATE"
    if lr_column in recent:
        result["LR_CONTRAST_ABNORMAL_COUNT_5"] = int(recent[lr_column].ne("NORMAL").sum())
        result["LR_CONTRAST_HIGH_COUNT_5"] = int(recent[lr_column].isin(["HIGH", "EXTREME"]).sum())
        result["LR_CONTRAST_SHARE"] = float(recent[lr_column].ne("NORMAL").mean())
        directional = recent.loc[recent[lr_column].ne("NORMAL"), "LR_CONTRAST_DIRECTION"]
        result["LR_CONTRAST_DIRECTION"] = directional.iloc[-1] if not directional.empty else "NONE"
    else:
        result.update({
            "LR_CONTRAST_ABNORMAL_COUNT_5": 0,
            "LR_CONTRAST_HIGH_COUNT_5": 0,
            "LR_CONTRAST_SHARE": 0.0,
            "LR_CONTRAST_DIRECTION": "NONE",
        })
    for canonical, column in zip(("MEAN", "STD", "MAX"), z_columns, strict=True):
        values = recent[column] if column in recent else pd.Series(dtype=float)
        result[f"{canonical}_Z_MEDIAN_5"] = float(values.median()) if len(values) else np.nan
        result[f"{canonical}_Z_SLOPE_5"] = calculate_trend_slope(values)
        result[f"EWMA_{canonical}"] = calculate_ewma(valid[column] if column in valid else [], config.ewma_alpha)
        result[f"CUSUM_{canonical}"] = calculate_cusum(valid[column] if column in valid else [], config.cusum_k)
        if len(context) >= config.context_window and column in context:
            previous, current = context.iloc[: config.trend_window][column], context.iloc[-config.trend_window :][column]
            result[f"DELTA_{canonical}_Z_5V5"] = float(current.median() - previous.median())
        else:
            result[f"DELTA_{canonical}_Z_5V5"] = np.nan
    transitions = context["WAFER_SIDE_STATUS"].isin(["CHECK", "WARNING", "ALARM"]).astype(int).diff().abs().sum()
    result["STATE_TRANSITIONS_10"] = int(transitions) if pd.notna(transitions) else 0
    return result


def classify_trend_type(stats: Mapping[str, object], config: DetectionConfig) -> str:
    abnormal = int(stats["ABNORMAL_COUNT_5"])
    shares = [
        float(stats[f"{group}_ABNORMAL_SHARE"])
        for group in ("LEVEL", "SPREAD", "MAXIMUM_EXCEEDANCE")
    ]
    persistent_groups = sum(share >= 0.6 for share in shares)
    deltas = [float(stats[f"DELTA_{name}_Z_5V5"]) for name in ("MEAN", "STD", "MAX") if pd.notna(stats[f"DELTA_{name}_Z_5V5"])]
    slopes = [float(stats[f"{name}_Z_SLOPE_5"]) for name in ("MEAN", "STD", "MAX") if pd.notna(stats[f"{name}_Z_SLOPE_5"])]
    cusum_alarm = any(float(stats[f"CUSUM_{name}"]) >= config.cusum_h for name in ("MEAN", "STD", "MAX"))
    if int(stats["STATE_TRANSITIONS_10"]) >= 4 and 1 <= abnormal <= config.trend_window - 1:
        return "OSCILLATION"
    if (
        abnormal >= config.trend_warning_min_abnormal
        and persistent_groups >= 2
        and sum(slope >= config.drift_min_slope for slope in slopes) >= 2
    ):
        return "GRADUAL_DRIFT"
    if (
        abnormal >= config.trend_alarm_min_abnormal
        and int(stats["PREVIOUS_ABNORMAL_COUNT_5"]) >= config.trend_alarm_min_abnormal
        and deltas
        and all(abs(delta) < config.step_delta_z for delta in deltas)
    ):
        return "SUSTAINED_SHIFT"
    if abnormal >= config.trend_alarm_min_abnormal and persistent_groups >= 2:
        if deltas and sum(delta >= config.step_delta_z for delta in deltas) >= 2:
            return "STEP_CHANGE"
        if cusum_alarm:
            return "SUSTAINED_SHIFT"
    if abnormal >= config.trend_warning_min_abnormal and slopes and max(slopes) >= config.drift_min_slope:
        return "GRADUAL_DRIFT"
    if int(stats["LR_CONTRAST_HIGH_COUNT_5"]) >= config.trend_warning_min_abnormal:
        return "SIDE_IMBALANCE"
    if abnormal == 1 and int(stats["SHORT_ABNORMAL_COUNT"]) <= 1:
        return "SPIKE"
    return "STABLE"


def evaluate_equipment_side(
    history: pd.DataFrame,
    config: DetectionConfig,
    z_columns: tuple[str, str, str] = ("MEAN_Z", "STD_Z", "MAX_Z"),
) -> dict[str, object]:
    """Evaluate one equipment-side only; this function never compares sides."""
    ordered = history.sort_values("TIMESTAMP", kind="stable")
    stats = calculate_window_statistics(ordered, config, z_columns)
    trend = classify_trend_type(stats, config)
    abnormal = int(stats["ABNORMAL_COUNT_5"])
    groups = sum(
        float(stats[f"{group}_ABNORMAL_SHARE"]) >= 0.6
        for group in ("LEVEL", "SPREAD", "MAXIMUM_EXCEEDANCE")
    )
    strong_cusum = any(float(stats[f"CUSUM_{name}"]) >= config.cusum_h for name in ("MEAN", "STD", "MAX"))
    lr_high = int(stats["LR_CONTRAST_HIGH_COUNT_5"])
    if trend in {"STEP_CHANGE", "GRADUAL_DRIFT", "SUSTAINED_SHIFT"} and (
        (abnormal >= config.trend_alarm_min_abnormal and groups >= 2) or strong_cusum
    ):
        status = "ALARM"
    elif abnormal >= config.trend_warning_min_abnormal and trend != "STABLE":
        status = "WARNING"
    elif lr_high >= config.trend_warning_min_abnormal:
        status = "WARNING"
    elif trend in {"SPIKE", "OSCILLATION"} or abnormal:
        status = "CHECK"
    else:
        status = "NORMAL"
    reason = (
        f"Trend={trend}; recent5 abnormal={abnormal}/{config.trend_window}; "
        f"persistent groups={groups}; STD median z={stats['STD_Z_MEDIAN_5']:.2f}; "
        f"MAX median z={stats['MAX_Z_MEDIAN_5']:.2f}; "
        f"LR contrast high={lr_high}/{config.trend_window} direction={stats['LR_CONTRAST_DIRECTION']}"
    )
    return {
        "EQP_ID": ordered["EQP_ID"].iloc[-1],
        "SIDE": ordered["SIDE"].iloc[-1],
        "EVALUATION_TIME": ordered["TIMESTAMP"].iloc[-1],
        **stats,
        "TREND_TYPE": trend,
        "SIDE_EQUIPMENT_STATUS": status,
        "SIDE_EQUIPMENT_SCORE": abnormal + groups,
        "SIDE_EQUIPMENT_REASON": reason,
    }


def combine_left_right_equipment_result(side_results: pd.DataFrame, metric_prefix: str = "RESI") -> pd.DataFrame:
    """Derive affected side only after independent side evaluation is complete."""
    rows: list[dict[str, object]] = []
    abnormal = {"CHECK", "WARNING", "ALARM"}
    for eqp_id, group in side_results.groupby("EQP_ID", sort=False):
        by_side = group.set_index("SIDE").to_dict("index")
        left = by_side.get("LEFT", {})
        right = by_side.get("RIGHT", {})
        left_status, right_status = left.get("SIDE_EQUIPMENT_STATUS", "NORMAL"), right.get("SIDE_EQUIPMENT_STATUS", "NORMAL")
        affected = "BOTH" if left_status in abnormal and right_status in abnormal else "LEFT" if left_status in abnormal else "RIGHT" if right_status in abnormal else "NONE"
        status_rank = {"NORMAL": 0, "CHECK": 1, "WARNING": 2, "ALARM": 3}
        overall = max((left_status, right_status), key=lambda status: status_rank[status])
        rows.append({
            "EQP_ID": eqp_id,
            "EVALUATION_TIME": max(group["EVALUATION_TIME"]),
            f"{metric_prefix}_LEFT_STATUS": left_status,
            f"{metric_prefix}_RIGHT_STATUS": right_status,
            f"{metric_prefix}_AFFECTED_SIDE": affected,
            f"{metric_prefix}_EQP_STATUS": overall,
            f"{metric_prefix}_TREND_TYPE": "/".join(sorted({str(item) for item in group["TREND_TYPE"]})),
            f"{metric_prefix}_EQP_REASON": " | ".join(group["SIDE_EQUIPMENT_REASON"]),
        })
    return pd.DataFrame(rows)
