"""Paired LEFT/RIGHT contrast signals for wafers processed concurrently."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import normalize_features
from .config import DetectionConfig


CONTRAST_COLUMNS = (
    "LR_MEAN_DELTA",
    "LR_STD_DELTA",
    "LR_MAX_DELTA",
    "LR_SD3_SHARE_DELTA",
    "LR_SD5_SHARE_DELTA",
    "LR_SD9_SHARE_DELTA",
)


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


def add_lr_contrast_features(side_level: pd.DataFrame, config: DetectionConfig) -> pd.DataFrame:
    """Add causal, product-aware paired LR contrast scores to each side row.

    A positive delta means RIGHT is high relative to LEFT; a negative delta
    means LEFT is high relative to RIGHT. The deltas are compared to their own
    historical pair baseline, so a stable intrinsic L/R asymmetry stays normal.
    """
    keys = ["EQP_ID", "WAFER_ID", "PRODUCT", "TIMESTAMP"]
    metrics = [column for column in ("N", "MEAN", "STD", "MAX", "SD3_COUNT", "SD5_COUNT", "SD9_COUNT") if column in side_level]
    left = side_level.loc[side_level["SIDE"].eq("LEFT"), keys + metrics].copy()
    right = side_level.loc[side_level["SIDE"].eq("RIGHT"), keys + metrics].copy()
    left = left.rename(columns={column: f"L_{column}" for column in metrics})
    right = right.rename(columns={column: f"R_{column}" for column in metrics})
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if paired.empty:
        output = side_level.copy()
        output["LR_CONTRAST_SCORE"] = np.nan
        output["LR_CONTRAST_STATE"] = "NORMAL"
        output["LR_CONTRAST_DIRECTION"] = "NONE"
        output["LR_CONTRAST_FOR_SIDE_STATE"] = "NORMAL"
        return output

    for metric in ("MEAN", "STD", "MAX"):
        paired[f"LR_{metric}_DELTA"] = np.log(
            (paired[f"R_{metric}"] + config.eps) / (paired[f"L_{metric}"] + config.eps)
        )
    for count in ("SD3_COUNT", "SD5_COUNT", "SD9_COUNT"):
        if f"L_{count}" in paired and f"R_{count}" in paired:
            left_share = paired[f"L_{count}"] / (paired["L_N"].clip(lower=1) + config.eps)
            right_share = paired[f"R_{count}"] / (paired["R_N"].clip(lower=1) + config.eps)
            paired[f"LR_{count.replace('_COUNT', '')}_SHARE_DELTA"] = np.log(
                (right_share + config.eps) / (left_share + config.eps)
            )
    paired["SIDE"] = "PAIR"
    normalized = normalize_features(paired, CONTRAST_COLUMNS, config)
    z_columns = [f"{column}_Z" for column in CONTRAST_COLUMNS if f"{column}_Z" in normalized]
    # Sparse threshold-exceedance counts are useful supporting evidence, but a 0-to-1 count
    # transition can be numerically unstable as a ratio. Localize a SIDE from
    # the continuous distribution features, and retain count contrast separately.
    primary_z_columns = [
        column for column in ("LR_MEAN_DELTA_Z", "LR_STD_DELTA_Z", "LR_MAX_DELTA_Z") if column in normalized
    ]
    exceedance_z_columns = [column for column in z_columns if column not in primary_z_columns]
    signed_z = normalized[primary_z_columns]
    magnitude = signed_z.abs()
    winning_z = magnitude.max(axis=1)
    has_baseline = magnitude.notna().any(axis=1)
    # pandas 3 raises for all-NA rows. Those rows are expected during the
    # baseline warm-up and must remain neutral rather than become an error.
    winning_column = magnitude.fillna(-np.inf).idxmax(axis=1).where(has_baseline)
    winning_signed_value = pd.Series(
        [signed_z.at[index, column] if pd.notna(column) else np.nan for index, column in winning_column.items()],
        index=normalized.index,
    )
    normalized["LR_CONTRAST_SCORE"] = winning_z
    normalized["LR_THRESHOLD_EXCEEDANCE_CONTRAST_SCORE"] = (
        normalized[exceedance_z_columns].abs().max(axis=1) if exceedance_z_columns else np.nan
    )
    normalized["LR_CONTRAST_STATE"] = winning_z.map(lambda value: _state(value, config))
    normalized["LR_CONTRAST_DIRECTION"] = np.select(
        [winning_signed_value.gt(0), winning_signed_value.lt(0)], ["RIGHT", "LEFT"], default="NONE"
    )
    keep = keys + z_columns + [
        "LR_CONTRAST_SCORE", "LR_THRESHOLD_EXCEEDANCE_CONTRAST_SCORE", "LR_CONTRAST_STATE", "LR_CONTRAST_DIRECTION"
    ]
    output = side_level.merge(normalized[keep], on=keys, how="left", validate="many_to_one")
    output["LR_CONTRAST_STATE"] = output["LR_CONTRAST_STATE"].fillna("NORMAL")
    output["LR_CONTRAST_DIRECTION"] = output["LR_CONTRAST_DIRECTION"].fillna("NONE")
    output["LR_CONTRAST_FOR_SIDE_STATE"] = np.where(
        output["SIDE"].eq(output["LR_CONTRAST_DIRECTION"]), output["LR_CONTRAST_STATE"], "NORMAL"
    )
    return output
