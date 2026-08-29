---
name: equipment-trend-anomaly-detection
description: Implement or modify production RESI and TEMP equipment anomaly detection using per-equipment-side historical baselines and persistent time-series trends. Use for wafer-side feature processing, trend alarms, and RESI/TEMP equipment-result fusion; not for chip-level metric-definition changes alone.
metadata:
  short-description: Detect persistent RESI and TEMP equipment trends
---

# Equipment Trend Anomaly Detection for RESI / TEMP

Build a production-oriented detector that answers whether an `EQP × SIDE`
distribution changed relative to its own historical reference behavior. Treat a
single wafer anomaly as evidence, not as an equipment alarm.

Read [the implementation specification](references/trend-detection-spec.md)
before designing or changing this pipeline.

## Required approach

- Inspect the repository before changing code. Reuse existing I/O,
  configuration, logging, and the established definitions of 2SD, 3SD, 5SD,
  and 9SD counts.
- Convert RESI wide wafer data into a side-level table. Always evaluate LEFT
  and RIGHT independently; do not use `suspect_side`, `label`, or an L/R ratio
  as the primary anomaly signal.
- Order all calculations chronologically using measurement time or `write_time`
  (then `date` only as a fallback). Deduplicate re-measurements by retaining the
  latest valid measurement for the physical wafer.
- Build causal, robust reference baselines with the preferred hierarchy
  `EQP × SIDE × PRODUCT` then `EQP × SIDE`; never include future wafers and
  exclude supplied known-problem or maintenance periods.
- Normalize each side's features against its own baseline before trend analysis.
  Keep all thresholds centralized and configurable.
- Keep RESI and TEMP feature logic independent. Fuse their **equipment-level**
  results only after each detector completes its own trend evaluation.
- Make the output explainable: every warning or alarm must identify the shifted
  features, baseline versus recent values, persistence, trend type, and the
  unaffected side where useful.

## Delivery structure

Adapt the project architecture instead of imposing a duplicate module tree.
Keep wafer classification, equipment-side trend evaluation, left/right
combination, and RESI/TEMP fusion separated by clear interfaces. Prefer
vectorized dataframe operations and evaluate trends from wafer-level data only.

## Verification

Add or update tests covering the mandatory scenarios in the specification:
stable asymmetry, isolated spike, persistent step change, gradual drift,
product transition, both-side shift, same-side RESI/TEMP corroboration, and a
RESI-only trend. Run the relevant test suite and report any input assumptions
or unavailable data needed for calibration/backtesting.
