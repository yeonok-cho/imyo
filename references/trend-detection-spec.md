# Equipment Trend Anomaly Detection Specification

## Goal and decision boundary

The pipeline aggregates chip-level data into wafer-level features, assesses
each `EQP × SIDE` series against its own historical behavior, classifies the
shape and persistence of a change, then generates equipment-level results. It
supports independent RESI and TEMP trend detectors and a final fusion stage.

The primary question is: **Has this equipment-side distribution changed versus
its own historical reference baseline?** A stable LEFT/RIGHT asymmetry is not
an anomaly. `label`, `rl_ratio`, and `suspect_side` may be retained for
backward compatibility, but must not drive side selection or be treated as
equipment-trend truth.

## Data preparation

Convert wafer-wide RESI records into this canonical side-level form:

| Field | Source / role |
| --- | --- |
| `EQP_ID`, `WAFER_ID`, `PRODUCT`, `TIMESTAMP`, `SIDE` | Identity and time |
| `N`, `MEAN`, `STD`, `MAX` | Distribution features |
| `SD2_COUNT`, `SD3_COUNT`, `SD5_COUNT`, `SD9_COUNT` | Counts above the project's pre-existing thresholds |

Map `L_*` and `R_*` source columns into LEFT and RIGHT rows. Use chronological
order (`measurement_time`, then `write_time`, then `date`); never rely on
incoming dataframe order. If a wafer was re-measured, retain the latest valid
measurement for the same equipment and physical wafer. A late `NO_DATA` record
must not replace an earlier valid record. Retain raw records separately if
needed.

Mark a side record `INSUFFICIENT_DATA` when `N < MIN_SIDE_N` (default `30`),
and exclude it from equipment-trend voting unless configuration explicitly
enables it.

## Causal reference baselines

For a wafer at time `t`, baseline observations must have timestamps strictly
before `t`; this applies to offline backtesting too. Exclude optional known
problem wafers/dates/lots and maintenance windows.

Select the most specific sufficiently sampled baseline:

1. `EQP × SIDE × PRODUCT`
2. `EQP × SIDE`
3. equipment-family `× SIDE × PRODUCT`
4. equipment-family `× SIDE`

Use `BASELINE_MIN_WAFERS = 20` initially. Name this a **reference baseline**,
not a confirmed healthy baseline. For every feature calculate median, MAD,
P90, P95, and P99 (mean/std are optional support values). Normalize using:

```text
robust_z = (value - baseline_median) / (1.4826 * baseline_MAD + EPS)
```

with `EPS = 1e-8`, guarded against non-finite results. Where useful, also
calculate current-to-baseline ratios such as `STD_RATIO_BASELINE`; these are
never LEFT/RIGHT ratios.

Normalize `MEAN`, `STD`, `MAX`, and the SD counts (at minimum 3SD, 5SD, 9SD).
Product-specific normalization must occur before mixed-product equipment trend
analysis so product changes in raw values cannot resemble process shifts.

## RESI wafer evidence

Classify three evidence groups rather than treating every metric as independent:

- **LEVEL**: `MEAN` (and `MEDIAN` if available)
- **SPREAD**: `STD`
- **MAXIMUM_EXCEEDANCE**: `MAX`, `SD3_COUNT`, `SD5_COUNT`, `SD9_COUNT`; `SD2_COUNT` is weak support

`MAXIMUM_EXCEEDANCE` is one composite evidence group, not two independent
votes. `MAX` is the wafer's upper order statistic. For each pre-existing
threshold `T_k` used by the project, `SDk_COUNT` is the threshold-exceedance
count `sum_i 1[X_i > T_k]`. This group does not estimate a tail index, does not
assert that the distribution is heavy-tailed, and does not label observations
as outliers.

Classify each group as `NORMAL`, `ELEVATED`, `HIGH`, or `EXTREME` with
configurable initial robust-z thresholds `2.5`, `3.5`, and `5.0` (or
baseline-percentile equivalents). Classify wafer-side evidence as `NORMAL`,
`CHECK`, `WARNING`, or `ALARM`: one abnormal group is normally CHECK; two
independent groups normally warrant WARNING; strong multi-group changes can be
ALARM. A lone high maximum is not equipment-failure proof.

## Trend engine

Use valid recent wafer sequences per `EQP × SIDE`, with configurable defaults:

```text
SHORT_WINDOW = 3
TREND_WINDOW = 5
CONTEXT_WINDOW = 10
TREND_WARNING_MIN_ABNORMAL = 3
TREND_ALARM_MIN_ABNORMAL = 4
EWMA_ALPHA = 0.30
CUSUM_K, CUSUM_H = configurable
```

Calculate short-window abnormal, warning, and alarm counts plus medians of the
normalized core metrics. For the five-wafer decision window calculate counts,
group abnormal shares, medians, and slopes for `MEAN_Z`, `STD_Z`, and `MAX_Z`.
For ten context wafers, compare previous five and recent five medians. Use
normalized metrics for EWMA; optionally calculate positive CUSUM as:

```text
cusum_t = max(0, cusum_(t-1) + normalized_value - CUSUM_K)
```

Classify patterns without using future observations in online decisions:

| Pattern | Required interpretation |
| --- | --- |
| `SPIKE` | Isolated strong wafer that returns to baseline; wafer event, normally CHECK/NORMAL equipment state |
| `STEP_CHANGE` | Recent five are persistently abnormal, at least two groups shift, and recent five significantly exceed previous five |
| `GRADUAL_DRIFT` | Positive trend/slope or EWMA/CUSUM support, recent elevation, and persistent abnormalities |
| `SUSTAINED_SHIFT` | Previous and recent five both abnormal at approximately the same elevated regime |
| `OSCILLATION` | Repeated normal/abnormal transitions or high normalized instability; not automatically physical failure |
| `STABLE` | No persistent change |

Online classification may mark a current abnormal wafer as `CURRENT_SUSPECT`;
only later observations may retrospectively confirm an isolated spike.

Set side equipment status independently:

- `NORMAL`: no persistent trend.
- `CHECK`: isolated spike, weak shift, oscillation, or inadequate persistence.
- `WARNING`: at least 3 recent abnormal wafers and trend evidence.
- `ALARM`: at least 4 recent abnormal wafers, persistent movement in at least 2
  feature groups, and `STEP_CHANGE`, `GRADUAL_DRIFT`, or `SUSTAINED_SHIFT`; a
  sustained strong EWMA/CUSUM path may also alarm.

Then derive affected side: LEFT-only, RIGHT-only, BOTH, or NONE. Both sides
may be abnormal; no contrast between sides is required. Count affected products
as confidence support, but do not require multi-product confirmation.

## TEMP detector

TEMP remains a separate detector with the same causal baseline and 3/5/10
window concepts but TEMP-specific features: `TEMP_MEAN`, `TEMP_MAX`,
`TEMP_P99_HIGH_COUNT`, `TEMP_P99_HIGH_SHARE`, and optionally unique-value
count, top-value share, and a quantization flag. TEMP quantization can make
standard deviation zero or near-zero, so raw TEMP standard-deviation ratios
must not be its main signal. Output TEMP spike, step, drift, sustained-shift,
or stable trend results with persistent evidence required for an equipment
alarm.

## Result interfaces and fusion

The wafer-side result must include identity/time, raw features, baseline level,
normalized features, feature-group states, wafer status, and explanation.

The equipment-side result must include recent valid wafer count; 3- and 5-wafer
counts; five-wafer medians/slopes; EWMA and CUSUM values; trend type; status,
score, and evidence-based reason.

Combine left/right outputs into a RESI (and TEMP) equipment result with
left/right status, trends, scores, affected side, equipment status, and reason.
Fuse only those completed equipment results using an explicit decision table or
small helpers rather than deeply nested branching:

| RESI | TEMP | Final severity |
| --- | --- | --- |
| ALARM | NORMAL | `RESI_ALARM` |
| NORMAL | ALARM | `TEMP_ALARM` |
| abnormal | abnormal | `SEVERE` |
| ALARM, persistent and same affected side | ALARM, persistent and same affected side | `CRITICAL` |

TEMP must never cancel a RESI alarm. If abnormal RESI and TEMP trends affect
opposite sides, retain the event as `SEVERE`, set `DIRECTION_MATCH = FALSE`,
and do not promote it to `CRITICAL`. Outputs must include `FINAL_REASON` and
`NOTIFICATION_REQUIRED`.

## Design interfaces

Use clear modular functions, adapting names to existing architecture where
equivalent components already exist:

```text
prepare_side_level_data                  deduplicate_wafer_measurements
build_resi_baseline                      calculate_baseline_statistics
normalize_resi_features                  classify_feature_groups
classify_wafer_side                      calculate_window_statistics
calculate_trend_slope                    calculate_ewma
calculate_cusum                          classify_trend_type
evaluate_equipment_side                  combine_left_right_equipment_result
build_temp_baseline                      calculate_temp_quantization_features
normalize_temp_features                  classify_temp_wafer_side
evaluate_temp_equipment_side             fuse_equipment_results
```

Keep wafer detection, equipment evaluation, and final fusion separate. Keep
thresholds in a central configuration and preserve existing configuration and
logging conventions. Aggregate chip data once and operate the trend engine on
wafer-level data with vectorized pandas/numpy operations; avoid repeated
full-history scans and row-wise dataframe `apply` when a vectorized/grouped
method is available.

## Tests and calibration

Test at minimum:

1. Stable L/R asymmetry (LEFT std ~0.4, RIGHT std ~0.9) remains normal.
2. A single MAX spike followed by normal wafers is a spike/wafer event, not an
   equipment alarm.
3. Persistent STD + MAX elevation over 4–5 wafers is a side alarm and step
   change.
4. Increasing spread over consecutive wafers becomes gradual drift with
   WARNING then ALARM.
5. Product-specific raw-level changes normalize to no equipment anomaly.
6. Independent shifts on both sides yield `BOTH`.
7. Same-side RESI and TEMP persistent alarms yield `CRITICAL`.
8. RESI alarm with normal TEMP yields `RESI_ALARM`.

Backtest 3/5, 4/5, and 5/7 persistence alternatives against known failures and
reference periods. Measure detection rate, false alarms, alarms/day, detection
delay, and isolated-spike promotions. Use results to calibrate windows,
thresholds, EWMA, and CUSUM values rather than treating defaults as validated
production limits.
