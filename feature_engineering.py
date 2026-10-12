"""
feature_engineering.py

Single source of truth for turning raw occupancy-sensor readings into
the feature set the occupancy model expects.

This module is imported by BOTH:
  - Occupancy_Training.ipynb   (to build the training dataset)
  - Occupancy_Inference.py     (to build features for a newly
                                 uploaded dataset, e.g. via Streamlit)

Keeping this logic in exactly one place is what prevents "train/serve
skew" -- the model silently seeing differently-computed features at
inference time than it saw during training.

Ported, unchanged in behaviour, from Occupancy_Prediction_Ver1_5.ipynb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Constants -- change these in ONE place if the model is ever retrained
# for a different forecast horizon or time resolution.
# ----------------------------------------------------------------------
FORECAST_HORIZON_MIN = 15
TIME_RESOLUTION_MIN = 5
HORIZON_STEPS = FORECAST_HORIZON_MIN // TIME_RESOLUTION_MIN   # 6
ROLLING_WINDOW_STEPS = 3                                       # 15 min

# Exact feature list / order the model was trained on.
MODEL_FEATURES = [
    'headcount',
    'headcount_rolling_15min',
    'headcount_trend',
    'is_weekend_future', 'working_weekday_hour_future',
    'hour_sin_future', 'hour_cos_future',
    'day_sin_future', 'day_cos_future',
]
TARGET_COLUMN = 'headcount_target_15min'

REQUIRED_RAW_COLUMNS = ['collecteddate', 'headcount']


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _parse_timestamps(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df = df.copy()
    df[timestamp_col] = (
        df[timestamp_col].astype(str).str.replace(r'\+.*$', '', regex=True)
    )
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)
    df = df.dropna(subset=[timestamp_col]).copy()
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    return df


def _build_5min_grid(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Resample to a regular 5-minute grid: take the latest reading in
    each minute, snap it onto the nearest 5-minute slot (backward,
    within 3 minutes tolerance), then forward-fill any still-missing
    slots."""
    df = df.copy()
    df['minute'] = df[timestamp_col].dt.floor('min')
    minute_latest = df.drop_duplicates(subset='minute', keep='last').copy()
    minute_latest = minute_latest.set_index('minute').sort_index()

    start = df[timestamp_col].min().floor('D')
    end = df[timestamp_col].max().ceil('D') - pd.Timedelta('1min')
    targets = pd.date_range(start=start, end=end, freq='5min')

    grid = pd.merge_asof(
        pd.DataFrame({'target_time': targets}),
        minute_latest,
        left_on='target_time',
        right_index=True,
        direction='backward',
        tolerance=pd.Timedelta('3min')
    )
    grid = grid.dropna(subset=[timestamp_col]).drop(columns=['minute'], errors='ignore')

    full_index = pd.date_range(
        start=grid['target_time'].min(),
        end=grid['target_time'].max(),
        freq='5min'
    )
    grid = (
        grid.set_index('target_time')
        .reindex(full_index)
        .ffill()
        .reset_index()
        .rename(columns={'index': 'target_time'})
    )
    return grid


def _add_cyclical_time_features(df: pd.DataFrame, time_col: str, suffix: str = '') -> pd.DataFrame:
    """Hour-of-day / day-of-week features, cyclically encoded, plus
    weekend and working-hour flags -- computed for whatever timestamp
    column is passed in (either "now" or the future forecast time)."""
    df = df.copy()
    df[f'hour_of_day{suffix}'] = df[time_col].dt.hour
    df[f'day_of_week{suffix}'] = df[time_col].dt.dayofweek

    df[f'hour_sin{suffix}'] = np.sin(2 * np.pi * df[f'hour_of_day{suffix}'] / 24)
    df[f'hour_cos{suffix}'] = np.cos(2 * np.pi * df[f'hour_of_day{suffix}'] / 24)
    df[f'day_sin{suffix}'] = np.sin(2 * np.pi * df[f'day_of_week{suffix}'] / 7)
    df[f'day_cos{suffix}'] = np.cos(2 * np.pi * df[f'day_of_week{suffix}'] / 7)

    df[f'is_weekend{suffix}'] = (df[f'day_of_week{suffix}'] >= 5).astype(int)
    is_working_hour = (
        (df[f'hour_of_day{suffix}'] >= 8) & (df[f'hour_of_day{suffix}'] < 18)
    ).astype(int)
    df[f'working_weekday_hour{suffix}'] = (
        (is_working_hour == 1) & (df[f'is_weekend{suffix}'] == 0)
    ).astype(int)
    return df


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def build_features(
    raw_df: pd.DataFrame,
    timestamp_col: str = 'collecteddate',
    headcount_col: str = 'headcount',
) -> pd.DataFrame:
    """
    Turn raw occupancy-sensor rows into the exact feature set the
    15-minute-ahead occupancy model expects.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Must contain a timestamp column and a headcount column (any
        other columns, e.g. an outdoor-temperature reading, are
        preserved and forward-filled onto the 5-minute grid, but are
        not used as model features here).
    timestamp_col, headcount_col : str
        Column names in raw_df, in case an upload uses different names.

    Returns
    -------
    pd.DataFrame, sorted by time, with columns:
        target_time, forecast_time, headcount,
        headcount_rolling_15min, headcount_trend, is_weekend_future,
        working_weekday_hour_future, hour_sin_future, hour_cos_future,
        day_sin_future, day_cos_future, headcount_target_15min
        (+ any other original columns, forward-filled)

    Notes
    -----
    Rows are NOT dropped here:
      - the first ROLLING_WINDOW_STEPS rows have NaN
        'headcount_rolling_15min' (not enough history yet)
      - the first 2 rows have NaN 'headcount_trend' (needs the prior
        two 5-minute readings, i.e. 10 minutes of history -- less
        strict than the rolling feature above, so it's never the
        binding warm-up constraint)
      - the last HORIZON_STEPS rows have NaN 'headcount_target_15min'
        (the future hasn't happened yet -- this is expected and
        correct for live inference, where there IS no future value)
    Callers decide what to do with these:
      - training drops both (see split_features_target)
      - live inference only cares about the most recent row(s) and
        never has a target
    """
    df = raw_df.rename(columns={timestamp_col: 'collecteddate', headcount_col: 'headcount'})

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded data is missing required column(s): {missing}. "
            f"Expected at least: {REQUIRED_RAW_COLUMNS}"
        )

    df = _parse_timestamps(df, 'collecteddate')
    if df.empty:
        raise ValueError(
            "No valid timestamps could be parsed from the "
            f"'{timestamp_col}' column."
        )

    grid = _build_5min_grid(df, 'collecteddate')

    grid['headcount'] = pd.to_numeric(grid['headcount'], errors='coerce')

    grid['headcount_rolling_15min'] = (
        grid['headcount']
        .shift(1)
        .rolling(window=ROLLING_WINDOW_STEPS, min_periods=ROLLING_WINDOW_STEPS)
        .mean()
    )

    # headcount_trend: simple two-point momentum indicator using ONLY
    # the two readings immediately before "now" -- t-5min vs t-10min
    # (grid is on a 5-minute step, so shift(1)/shift(2) are exactly
    # that). This never looks at the current or future reading, so
    # it's safe for both training and live inference.
    #   +1 -> occupancy was rising going into "now"
    #   -1 -> occupancy was falling going into "now"
    #    0 -> unchanged (or the two prior readings happen to be equal)
    grid['headcount_trend'] = np.sign(
        grid['headcount'].shift(1) - grid['headcount'].shift(2)
    )

    grid[TARGET_COLUMN] = grid['headcount'].shift(-HORIZON_STEPS)
    grid['forecast_time'] = grid['target_time'] + pd.Timedelta(minutes=FORECAST_HORIZON_MIN)

    grid = _add_cyclical_time_features(grid, 'forecast_time', suffix='_future')

    front_cols = ['target_time', 'forecast_time'] + MODEL_FEATURES + [TARGET_COLUMN]
    other_cols = [c for c in grid.columns if c not in front_cols]
    ordered_cols = front_cols + other_cols

    return grid[ordered_cols].reset_index(drop=True)


def split_features_target(features_df: pd.DataFrame, dropna: bool = True):
    """
    Convenience helper for TRAINING: returns (X, y) built from a
    build_features() output.

    dropna=True (the default, used for training) removes rows missing
    any required feature or the target -- i.e. the rolling-window
    warm-up rows at the start and the "future not known yet" rows at
    the end.
    """
    data = features_df.copy()
    if dropna:
        data = data.dropna(subset=MODEL_FEATURES + [TARGET_COLUMN])
    return data[MODEL_FEATURES], data[TARGET_COLUMN]
