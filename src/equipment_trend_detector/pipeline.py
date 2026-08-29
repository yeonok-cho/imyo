"""Public pipeline entrypoints for independent RESI and TEMP evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .baseline import build_resi_baseline, build_temp_baseline
from .config import DetectionConfig
from .contrast import add_lr_contrast_features
from .detector import (
    classify_temp_wafer_side,
    classify_wafer_side,
    combine_left_right_equipment_result,
    evaluate_equipment_side,
)
from .side_level import deduplicate_wafer_measurements, prepare_side_level_data


def _evaluate_all_sides(
    wafer_results: pd.DataFrame,
    config: DetectionConfig,
    prefix: str,
    z_columns: tuple[str, str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = [
        evaluate_equipment_side(group, config, z_columns)
        for _, group in wafer_results.groupby(["EQP_ID", "SIDE"], sort=False)
        if not group.empty
    ]
    side = pd.DataFrame(records)
    equipment = combine_left_right_equipment_result(side, prefix) if not side.empty else pd.DataFrame()
    return side, equipment


def run_resi_detector(
    wide_resi: pd.DataFrame,
    config: DetectionConfig | None = None,
    excluded_timestamps: Iterable[object] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run wide RESI input through side preparation, causal baseline, and trend engine."""
    cfg = config or DetectionConfig()
    side_level = deduplicate_wafer_measurements(prepare_side_level_data(wide_resi))
    side_level = add_lr_contrast_features(side_level, cfg)
    normalized = build_resi_baseline(side_level, cfg, excluded_timestamps=excluded_timestamps)
    wafer = classify_wafer_side(normalized, cfg)
    side, equipment = _evaluate_all_sides(wafer, cfg, "RESI", ("MEAN_Z", "STD_Z", "MAX_Z"))
    return {"wafer_side": wafer, "equipment_side": side, "equipment": equipment}


def prepare_temp_side_level_data(wide_temp: pd.DataFrame) -> pd.DataFrame:
    """Prepare TEMP wide records while preserving quantization-friendly metrics."""
    return prepare_side_level_data(
        wide_temp,
        {
            "N": "n",
            "TEMP_MEAN": "temp_mean",
            "TEMP_MAX": "temp_max",
            "TEMP_P99_HIGH_COUNT": "temp_p99_high_count",
            "TEMP_P99_HIGH_SHARE": "temp_p99_high_share",
        },
    )


def calculate_temp_quantization_features(side_level: pd.DataFrame) -> pd.DataFrame:
    """Derive optional quantization flags without relying on raw TEMP std."""
    output = side_level.copy()
    if {"TEMP_UNIQUE_VALUE_COUNT", "N"}.issubset(output.columns):
        output["TEMP_QUANTIZED"] = output["TEMP_UNIQUE_VALUE_COUNT"].div(output["N"].clip(lower=1)).lt(0.10)
    return output


def run_temp_detector(
    wide_temp: pd.DataFrame,
    config: DetectionConfig | None = None,
    excluded_timestamps: Iterable[object] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run TEMP independently; callers fuse only the returned equipment result."""
    cfg = config or DetectionConfig()
    side_level = calculate_temp_quantization_features(deduplicate_wafer_measurements(prepare_temp_side_level_data(wide_temp)))
    normalized = build_temp_baseline(side_level, cfg, excluded_timestamps=excluded_timestamps)
    wafer = classify_temp_wafer_side(normalized, cfg)
    # The generic trend engine needs three normalized columns; TEMP has no
    # spread metric, so the maximum/exceedance composite is supplied twice.
    wafer["TEMP_MAXIMUM_EXCEEDANCE_Z"] = wafer[
        ["TEMP_MAX_Z", "TEMP_P99_HIGH_COUNT_Z", "TEMP_P99_HIGH_SHARE_Z"]
    ].max(axis=1)
    side, equipment = _evaluate_all_sides(
        wafer, cfg, "TEMP", ("TEMP_MEAN_Z", "TEMP_MAXIMUM_EXCEEDANCE_Z", "TEMP_MAX_Z")
    )
    return {"wafer_side": wafer, "equipment_side": side, "equipment": equipment}
