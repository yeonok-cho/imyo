"""Canonical side-level input construction and safe re-measurement handling."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


_IDENTITY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "EQP_ID": ("EQP_ID", "eqp_masked", "eqp_id"),
    "WAFER_ID": ("WAFER_ID", "wafer_masked", "wafer_id"),
    "PRODUCT": ("PRODUCT", "product_masked", "product"),
}


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...], required: bool = True) -> str | None:
    column = next((name for name in candidates if name in frame.columns), None)
    if required and column is None:
        raise ValueError(f"Missing required input column; expected one of {candidates}.")
    return column


def _timestamp_column(frame: pd.DataFrame) -> str:
    return _first_column(frame, ("measurement_time", "write_time", "TIMESTAMP", "date"))  # type: ignore[return-value]


def prepare_side_level_data(wide: pd.DataFrame, feature_map: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Unpivot L_/R_ wafer data without deriving anomaly direction from ratios.

    ``feature_map`` maps canonical output feature names to their source suffixes,
    e.g. ``{"MEAN": "mean", "SD3_COUNT": "3SD"}``. Missing optional source
    columns are represented as NaN so one partial metric cannot shift columns.
    """
    default_map = {
        "N": "n",
        "MEAN": "mean",
        "STD": "std",
        "MAX": "max",
        "SD2_COUNT": "2SD",
        "SD3_COUNT": "3SD",
        "SD5_COUNT": "5SD",
        "SD9_COUNT": "9SD",
    }
    mapping = feature_map or default_map
    timestamp = _timestamp_column(wide)
    identity = {key: _first_column(wide, values) for key, values in _IDENTITY_ALIASES.items()}
    records: list[pd.DataFrame] = []
    for prefix, side in (("L", "LEFT"), ("R", "RIGHT")):
        result = pd.DataFrame({output: wide[source] for output, source in identity.items()})
        result["TIMESTAMP"] = pd.to_datetime(wide[timestamp], errors="coerce")
        result["SIDE"] = side
        for output, suffix in mapping.items():
            source = f"{prefix}_{suffix}"
            result[output] = pd.to_numeric(wide[source], errors="coerce") if source in wide else np.nan
        records.append(result)
    output = pd.concat(records, ignore_index=True)
    # Pasted reports often end with prose after the TSV body. A record without
    # the minimum identity/time fields is not a wafer measurement and must not
    # become a phantom equipment with an INSUFFICIENT_DATA result.
    output = output.dropna(subset=["EQP_ID", "WAFER_ID", "TIMESTAMP"])
    return output.sort_values(["EQP_ID", "SIDE", "TIMESTAMP"], kind="stable").reset_index(drop=True)


def deduplicate_wafer_measurements(side_level: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest valid measurement; a later NO_DATA row cannot erase it."""
    required = {"EQP_ID", "WAFER_ID", "SIDE", "TIMESTAMP"}
    missing = required - set(side_level.columns)
    if missing:
        raise ValueError(f"Side-level data is missing {sorted(missing)}")
    frame = side_level.copy()
    status_column = next(
        (column for column in ("MEASUREMENT_STATUS", "measurement_status", "label") if column in frame.columns), None
    )
    status = frame[status_column].astype(str).str.upper() if status_column else pd.Series("", index=frame.index)
    numeric = frame[[col for col in ("N", "MEAN", "STD", "MAX") if col in frame]].notna().any(axis=1)
    positive_sample = frame["N"].gt(0) if "N" in frame else pd.Series(True, index=frame.index)
    frame["_VALID_MEASUREMENT"] = numeric & positive_sample & ~status.eq("NO_DATA")
    frame = frame.sort_values(["EQP_ID", "WAFER_ID", "SIDE", "TIMESTAMP"], kind="stable")
    valid = frame.loc[frame["_VALID_MEASUREMENT"]]
    latest_valid = valid.groupby(["EQP_ID", "WAFER_ID", "SIDE"], dropna=False, as_index=False).tail(1)
    no_valid_key = ~frame.set_index(["EQP_ID", "WAFER_ID", "SIDE"]).index.isin(
        latest_valid.set_index(["EQP_ID", "WAFER_ID", "SIDE"]).index
    )
    retained_invalid = frame.loc[no_valid_key].groupby(["EQP_ID", "WAFER_ID", "SIDE"], dropna=False, as_index=False).tail(1)
    output = pd.concat([latest_valid, retained_invalid], ignore_index=True)
    return output.drop(columns="_VALID_MEASUREMENT").sort_values(
        ["EQP_ID", "SIDE", "TIMESTAMP"], kind="stable"
    ).reset_index(drop=True)
