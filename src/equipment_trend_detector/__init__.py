"""Causal, side-independent equipment trend detection for RESI and TEMP."""

from .config import DetectionConfig
from .fusion import fuse_equipment_results
from .pipeline import run_resi_detector, run_temp_detector

__all__ = [
    "DetectionConfig",
    "fuse_equipment_results",
    "run_resi_detector",
    "run_temp_detector",
]
