"""Explicit equipment-result fusion, deliberately separate from raw wafer logic."""

from __future__ import annotations

import pandas as pd


def _is_alarm(status: object) -> bool:
    return status == "ALARM"


def fuse_equipment_results(resi_result: pd.DataFrame, temp_result: pd.DataFrame) -> pd.DataFrame:
    """Fuse completed RESI and TEMP equipment results without cancellation.

    Input rows are expected to be one per EQP from the independent pipelines.
    Missing detector results are treated as NORMAL only for presentation; callers
    may instead require both inputs before issuing production notifications.
    """
    resi = resi_result.copy()
    temp = temp_result.copy()
    merged = resi.merge(temp, on="EQP_ID", how="outer", suffixes=("_RESI", "_TEMP"))
    rows: list[dict[str, object]] = []
    for row in merged.to_dict("records"):
        resi_status = row.get("RESI_EQP_STATUS", "NORMAL") if pd.notna(row.get("RESI_EQP_STATUS")) else "NORMAL"
        temp_status = row.get("TEMP_EQP_STATUS", "NORMAL") if pd.notna(row.get("TEMP_EQP_STATUS")) else "NORMAL"
        resi_side = row.get("RESI_AFFECTED_SIDE", "NONE") if pd.notna(row.get("RESI_AFFECTED_SIDE")) else "NONE"
        temp_side = row.get("TEMP_AFFECTED_SIDE", "NONE") if pd.notna(row.get("TEMP_AFFECTED_SIDE")) else "NONE"
        resi_alarm, temp_alarm = _is_alarm(resi_status), _is_alarm(temp_status)
        both_abnormal = resi_status != "NORMAL" and temp_status != "NORMAL"
        direction_match = resi_side == temp_side and resi_side not in {"NONE", "BOTH"}
        if resi_alarm and temp_alarm and direction_match:
            severity = "CRITICAL"
        elif both_abnormal:
            severity = "SEVERE"
        elif resi_alarm:
            severity = "RESI_ALARM"
        elif temp_alarm:
            severity = "TEMP_ALARM"
        elif resi_status == "WARNING" or temp_status == "WARNING":
            severity = "WARNING"
        elif resi_status == "CHECK" or temp_status == "CHECK":
            severity = "CHECK"
        else:
            severity = "NORMAL"
        rows.append({
            "EQP_ID": row["EQP_ID"],
            "RESI_EQP_STATUS": resi_status,
            "TEMP_EQP_STATUS": temp_status,
            "RESI_AFFECTED_SIDE": resi_side,
            "TEMP_AFFECTED_SIDE": temp_side,
            "DIRECTION_MATCH": bool(direction_match),
            "FINAL_SEVERITY": severity,
            "NOTIFICATION_REQUIRED": severity in {"RESI_ALARM", "TEMP_ALARM", "SEVERE", "CRITICAL"},
            "FINAL_REASON": f"RESI={resi_status}({resi_side}); TEMP={temp_status}({temp_side})",
        })
    return pd.DataFrame(rows)
