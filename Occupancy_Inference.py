"""
Occupancy_Inference.py

Inference-only entry point for the 15-minute-ahead occupancy model.
No training code, no scikit-learn model-selection machinery, no
Optuna -- just: load the saved pipeline, engineer features with the
SAME function used at training time, predict.

Used by streamlit_app.py. Can also be run standalone for testing:
    python Occupancy_Inference.py new_data.csv
"""

from __future__ import annotations

import sys
import json

import joblib
import pandas as pd

from feature_engineering import build_features, MODEL_FEATURES, TARGET_COLUMN

DEFAULT_MODEL_PATH = 'occupancy_model_15min.pkl'
DEFAULT_CONFIG_PATH = 'occupancy_model_15min_config.json'

# headcount_rolling_15min needs 3 prior 5-minute readings before it's
# defined -- i.e. 15 minutes of continuous history before the first
# row a prediction can be made for.
MIN_WARMUP_MINUTES = 15


def load_model(model_path: str = DEFAULT_MODEL_PATH, config_path: str = DEFAULT_CONFIG_PATH):
    """Load the trained pipeline and its feature/target configuration.

    Raises a clear error if the config on disk doesn't match this
    version of feature_engineering.py, instead of silently producing
    wrong predictions.
    """
    model = joblib.load(model_path)
    with open(config_path) as f:
        config = json.load(f)

    if config.get('features') != MODEL_FEATURES:
        raise ValueError(
            "The loaded model's feature_config.json does not match "
            "the feature list in feature_engineering.py. This model "
            "was trained on a different feature set -- retrain, or "
            "point this code at the matching feature_engineering.py."
        )
    return model, config


def predict(
    raw_df: pd.DataFrame,
    model,
    timestamp_col: str = 'collecteddate',
    headcount_col: str = 'headcount',
) -> pd.DataFrame:
    """
    Run feature engineering + prediction on newly uploaded raw sensor
    data.

    Returns one row per usable timestamp with:
        timestamp, forecast_time, occupancy_now, occupancy_forecast_15min
        (clipped to >= 0, then rounded to the nearest whole person)
    plus, if the upload happens to contain enough trailing history:
        occupancy_actual_15min_later   (kept ONLY for optional
                                         forecast-accuracy validation --
                                         never fed back into the model
                                         or the HVAC controller)
    plus any other original columns (e.g. outdoor_temperature_C),
    passed through so hvac_simulation.py can use them directly.
    """
    features_df = build_features(raw_df, timestamp_col=timestamp_col, headcount_col=headcount_col)

    # A row is usable for prediction once every model feature is
    # populated. headcount_target_15min is allowed to be NaN here --
    # that's the normal, expected case for a live upload with no
    # known future.
    usable = features_df.dropna(subset=MODEL_FEATURES).copy()

    if usable.empty:
        raise ValueError(
            "Not enough continuous history in the uploaded data to "
            f"compute features. Need at least {MIN_WARMUP_MINUTES} "
            "minutes of continuous 5-minute readings before the first "
            "point you want a prediction for."
        )

    predictions = model.predict(usable[MODEL_FEATURES])

    result = pd.DataFrame({
        'timestamp': usable['target_time'].values,
        'forecast_time': usable['forecast_time'].values,
        'occupancy_now': usable['headcount'].values,
        'occupancy_forecast_15min': predictions,
    })
    result['occupancy_forecast_15min'] = result['occupancy_forecast_15min'].clip(lower=0).round()

    if TARGET_COLUMN in usable.columns:
        result['occupancy_actual_15min_later'] = usable[TARGET_COLUMN].values

    passthrough_cols = [
        c for c in usable.columns
        if c not in set(MODEL_FEATURES) | {'target_time', 'forecast_time', TARGET_COLUMN, 'headcount'}
    ]
    for c in passthrough_cols:
        result[c] = usable[c].values

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python Occupancy_Inference.py <path_to_new_data.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    raw_df = pd.read_csv(csv_path)

    model, _config = load_model()
    result = predict(raw_df, model)

    out_path = 'occupancy_predictions.csv'
    result.to_csv(out_path, index=False)
    print(f"Wrote {len(result)} predictions to {out_path}")
    print(result.head())


if __name__ == '__main__':
    main()
