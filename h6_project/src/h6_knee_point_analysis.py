import pickle
from pathlib import Path
import time

import numpy as np
import pandas as pd

# =====================================================
# 설정
# =====================================================

INPUT_DIR = "/workspace/h6산출/cow_bonding_profile_results_eqp_69"
OUTPUT_DIR = "/workspace/h6산출/h6_output_69"

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)

GROUP_COLS = [
    'eqp_id',
    'product',
    'wafer_id',
    'module_id',
    'x',
    'y'
]

# =====================================================
# Knee Point
# =====================================================

def find_knee_points(
    g,
    smooth_window=31,
    slope_ratio=0.1,
    plateau_time_window=100
):

    g = (
        g.sort_values('TIME')
         .drop_duplicates(subset='TIME')
         .copy()
    )

    if len(g) < 5:
        return None

    # ------------------
    # smoothing
    # ------------------

    g['ACT_POS_SMOOTH'] = (
        g['ACT_POS']
        .rolling(
            smooth_window,
            center=True,
            min_periods=1
        )
        .mean()
    )

    # ------------------
    # slope
    # ------------------

    g['SLOPE'] = np.gradient(
        g['ACT_POS_SMOOTH'].values,
        g['TIME'].values
    )

    max_slope = g['SLOPE'].max()
    threshold = max_slope * slope_ratio

    # ------------------
    # t0
    # ------------------

    t0_time = g['t0_time'].iloc[0]

    idx_t0 = (
        (g['TIME'] - t0_time)
        .abs()
        .idxmin()
    )

    actpos_t0 = g.loc[
        idx_t0,
        'ACT_POS'
    ]

    # ------------------
    # min act pos (상승 시작 기준점)
    # ------------------

    idx_min_act = g['ACT_POS'].idxmin()
    min_time    = g.loc[idx_min_act, 'TIME']
    actpos_min  = g.loc[idx_min_act, 'ACT_POS']

    post_min = g[g['TIME'] >= min_time]

    # ------------------
    # 상승 시작점 (slope가 threshold를 넘는 첫 지점)
    # ------------------

    rising_candidate = post_min[post_min['SLOPE'] > threshold]

    if len(rising_candidate):

        rise_start_time = rising_candidate['TIME'].iloc[0]

        after_rise = post_min[post_min['TIME'] >= rise_start_time]

        # ------------------
        # 상승 정지점 (knee) : 상승 시작 이후 slope가 다시
        # threshold 이하로 떨어지는 첫 지점
        # ------------------

        plateau_candidate = after_rise[after_rise['SLOPE'] <= threshold]

        if len(plateau_candidate):

            plateau_time = plateau_candidate['TIME'].iloc[0]

            search_region = g[
                (g['TIME'] >= plateau_time)
                &
                (g['TIME'] <= plateau_time + plateau_time_window)
            ]

            idx_forward = (
                search_region['ACT_POS_SMOOTH'].idxmax()
            )

        else:
            idx_forward = after_rise['ACT_POS_SMOOTH'].idxmax()

    else:
        idx_forward = post_min['ACT_POS_SMOOTH'].idxmax()

    time_forward = g.loc[
        idx_forward,
        'TIME'
    ]

    actpos_forward = g.loc[
        idx_forward,
        'ACT_POS'
    ]

    h6 = (
        actpos_forward
        -
        actpos_min
    )

    return pd.Series({

        'TIME_FORWARD':
            time_forward,

        'ACT_POS_FORWARD':
            actpos_forward,

        'ACT_POS_T0':
            actpos_t0,

        'h6':
            h6
    })

# =====================================================
# 파일 처리
# =====================================================

def process_one_file(pkl_file):

    print("\n" + "=" * 80)
    print(f"Processing : {pkl_file.name}")
    print("=" * 80)

    total_start = time.perf_counter()

    # -------------------------------------------------
    # load
    # -------------------------------------------------

    t0 = time.perf_counter()

    with open(pkl_file, 'rb') as f:
        df = pickle.load(f)

    print(
        f"Load Time : "
        f"{time.perf_counter()-t0:.2f}s"
    )

    print(
        f"Raw Shape : {df.shape}"
    )

    # -------------------------------------------------
    # datatype
    # -------------------------------------------------

    if (
        'ACT_POS' in df.columns
        and
        df['ACT_POS'].dtype == 'object'
    ):
        df['ACT_POS'] = pd.to_numeric(
            df['ACT_POS'],
            errors='coerce'
        )

    # -------------------------------------------------
    # duplicate
    # -------------------------------------------------

    before = len(df)

    df = df.drop_duplicates()

    print(
        f"Duplicate Removed : "
        f"{before-len(df)}"
    )

    # -------------------------------------------------
    # filter
    # -------------------------------------------------

    t0 = time.perf_counter()

    cols = [
        't0_time',
        'eqp_id',
        'wafer_id',
        'product',
        'module_id',
        'x',
        'y',
        'TIME',
        'ACT_POS'
    ]

    df_h6 = df[cols].copy()

    df_h6 = df_h6[
        (df_h6['t0_time'] <= df_h6['TIME'])
        &
        (df_h6['TIME'] <= 9701)
    ].copy()

    print(
        f"Filter Time : "
        f"{time.perf_counter()-t0:.2f}s"
    )

    print(
        f"Filtered Shape : "
        f"{df_h6.shape}"
    )

    # -------------------------------------------------
    # knee
    # -------------------------------------------------

    t0 = time.perf_counter()

    result = (
        df_h6
        .groupby(
            GROUP_COLS,
            observed=True
        )
        .apply(find_knee_points)
        .dropna()
        .reset_index()
    )

    print(
        f"Knee Time : "
        f"{time.perf_counter()-t0:.2f}s"
    )

    print(
        f"Result Shape : "
        f"{result.shape}"
    )

    # -------------------------------------------------
    # median forward
    # -------------------------------------------------

    t0 = time.perf_counter()

    median_forward = (
        result
        .groupby(
            [
                'eqp_id',
                'wafer_id',
                'module_id'
            ]
        )['TIME_FORWARD']
        .median()
        .round()
        .reset_index()
        .rename(
            columns={
                'TIME_FORWARD':
                'MEDIAN_FORWARD_TIME'
            }
        )
    )

    result = result.merge(
        median_forward,
        on=[
            'eqp_id',
            'wafer_id',
            'module_id'
        ],
        how='left'
    )

    print(
        f"Median Time : "
        f"{time.perf_counter()-t0:.2f}s"
    )

    # -------------------------------------------------
    # h6_alter
    # -------------------------------------------------

    t0 = time.perf_counter()

    lookup = (
        df_h6[
            GROUP_COLS +
            ['TIME', 'ACT_POS']
        ]
        .drop_duplicates()
        .rename(
            columns={
                'TIME':
                    'MEDIAN_FORWARD_TIME',
                'ACT_POS':
                    'ACT_POS_MEDIAN'
            }
        )
    )

    result['MEDIAN_FORWARD_TIME'] = (
        result['MEDIAN_FORWARD_TIME']
        .astype(float)
    )

    result = result.merge(
        lookup,
        on=[
            'eqp_id',
            'product',
            'wafer_id',
            'module_id',
            'x',
            'y',
            'MEDIAN_FORWARD_TIME'
        ],
        how='left'
    )

    result['h6_alter'] = (
        result['ACT_POS_MEDIAN']
        -
        result['ACT_POS_T0']
    )

    print(
        f"H6 Alter Time : "
        f"{time.perf_counter()-t0:.2f}s"
    )

    # -------------------------------------------------
    # final
    # -------------------------------------------------

    result = result[
        [
            'eqp_id',
            'wafer_id',
            'product',
            'module_id',
            'x',
            'y',

            'TIME_FORWARD',
            'MEDIAN_FORWARD_TIME',

            'ACT_POS_T0',
            'ACT_POS_FORWARD',
            'ACT_POS_MEDIAN',

            'h6',
            'h6_alter'
        ]
    ]

    print(
        f"Total File Time : "
        f"{time.perf_counter()-total_start:.2f}s"
    )

    return result

# =====================================================
# Main
# =====================================================

all_results = []

pkl_files = sorted(
    Path(INPUT_DIR).glob("*.pkl")
)

print(
    f"\nFound {len(pkl_files)} files"
)

for pkl_file in pkl_files:

    result = process_one_file(
        pkl_file
    )

    save_path = (
        Path(OUTPUT_DIR)
        /
        f"{pkl_file.stem}_h6.csv"
    )

    result.to_csv(
        save_path,
        index=False
    )

    print(
        f"Saved : {save_path}"
    )

    all_results.append(result)

# =====================================================
# Final Merge
# =====================================================

if len(all_results):

    final_df = pd.concat(
        all_results,
        ignore_index=True
    )

    final_path = (
        Path(OUTPUT_DIR)
        /
        "h6_result_all.csv"
    )

    final_df.to_csv(
        final_path,
        index=False
    )

    print("\n" + "=" * 80)
    print("FINAL")
    print("=" * 80)
    print(
        f"Final Shape : "
        f"{final_df.shape}"
    )
    print(
        f"Saved : {final_path}"
    )
