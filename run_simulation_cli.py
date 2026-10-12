"""
run_simulation_cli.py

Command-line equivalent of streamlit_app.py: upload a raw occupancy CSV,
get a 15-minute-ahead occupancy forecast, run the 1R1C HVAC simulation
(scheduled baseline vs thermal-predictive control), and write the
results to disk -- no Streamlit involved.

Assumes this file lives in the same directory as:
    occupancy_model_15min.pkl
    occupancy_model_15min_config.json
    feature_engineering.py
    Occupancy_Inference.py
    hvac_simulation.py

Usage
-----
Basic (all defaults -- prototype HVAC parameters):
    python run_simulation_cli.py raw_data.csv

Custom columns / HVAC parameters / output location:
    python run_simulation_cli.py raw_data.csv \
        --timestamp-col collecteddate --headcount-col headcount \
        --outdoor-col outdoor_temperature_C \
        --comfort-min 22.8 --comfort-max 25.8 --cop 3.0 \
        --q-cooling-max-w 5000 --r-thermal 0.01 --c-thermal 1000000 \
        --schedule-start 8 --schedule-end 18 --weekdays-only \
        --output-dir results/

Outputs (written to --output-dir, default: current directory)
---------------------------------------------------------------
    occupancy_forecast.csv   -- per-timestamp forecast (from Occupancy_Inference.predict)
    hvac_comparison.csv      -- per-timestep scheduled vs predictive comparison
    hvac_summary.csv         -- KPI table (scheduled vs predictive vs difference)

KPIs are also printed to the console.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from Occupancy_Inference import load_model, predict, DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH
from hvac_simulation import run_simulation, HVACConfig


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the occupancy forecast + HVAC simulation on a raw sensor CSV (no Streamlit)."
    )
    p.add_argument("csv_path", help="Path to the raw sensor CSV (timestamp + headcount columns).")

    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help=f"Default: {DEFAULT_MODEL_PATH}")
    p.add_argument("--config-path", default=DEFAULT_CONFIG_PATH, help=f"Default: {DEFAULT_CONFIG_PATH}")

    p.add_argument("--timestamp-col", default="collecteddate")
    p.add_argument("--headcount-col", default="headcount")
    p.add_argument("--outdoor-col", default="outdoor_temperature_C",
                    help="Optional outdoor-temperature column. If absent from the CSV, "
                         "a constant fallback is used (see HVACConfig.fallback_outdoor_temp_c).")

    p.add_argument("--output-dir", default=".", help="Directory to write output CSVs into. Default: current directory.")

    # HVACConfig overrides -- omit any of these to keep the dataclass default.
    p.add_argument("--comfort-min", type=float, default=None, help="Comfort band min, deg C.")
    p.add_argument("--comfort-max", type=float, default=None, help="Comfort band max, deg C.")
    p.add_argument("--cop", type=float, default=None, help="HVAC coefficient of performance.")
    p.add_argument("--q-cooling-max-w", type=float, default=None, help="Max cooling capacity, W.")
    p.add_argument("--r-thermal", type=float, default=None, help="1R1C thermal resistance, K/W.")
    p.add_argument("--c-thermal", type=float, default=None, help="1R1C thermal capacitance, J/K.")
    p.add_argument("--schedule-start", type=float, default=None, help="Scheduled baseline start hour (0-24).")
    p.add_argument("--schedule-end", type=float, default=None, help="Scheduled baseline end hour (0-24).")
    p.add_argument("--weekdays-only", action="store_true", default=None,
                    help="Scheduled baseline runs weekdays only (pass this flag to enable).")

    return p.parse_args(argv)


def build_hvac_config(args: argparse.Namespace) -> HVACConfig:
    """Start from HVACConfig's dataclass defaults, override only the
    fields the user actually passed on the command line."""
    cfg = HVACConfig()
    overrides = {
        "comfort_min_c": args.comfort_min,
        "comfort_max_c": args.comfort_max,
        "cop": args.cop,
        "q_cooling_max_w": args.q_cooling_max_w,
        "r_thermal_k_per_w": args.r_thermal,
        "c_thermal_j_per_k": args.c_thermal,
        "schedule_start_hour": args.schedule_start,
        "schedule_end_hour": args.schedule_end,
    }
    for field, value in overrides.items():
        if value is not None:
            setattr(cfg, field, value)
    if args.weekdays_only is not None:
        cfg.schedule_weekdays_only = args.weekdays_only
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        raw_df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Could not read '{args.csv_path}': {e}", file=sys.stderr)
        return 1

    try:
        model, model_config = load_model(args.model_path, args.config_path)
    except FileNotFoundError as e:
        print(f"Could not load model/config: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Model/config mismatch: {e}", file=sys.stderr)
        return 1

    print(f"Loaded model from '{args.model_path}' "
          f"(forecast horizon: {model_config.get('forecast_horizon_minutes', '?')} min)")

    try:
        pred_df = predict(raw_df, model, timestamp_col=args.timestamp_col, headcount_col=args.headcount_col)
    except ValueError as e:
        print(f"Could not compute the forecast: {e}", file=sys.stderr)
        return 1

    print(f"Computed forecast for {len(pred_df)} timestamps.")

    cfg = build_hvac_config(args)
    outdoor_col = args.outdoor_col if args.outdoor_col in pred_df.columns else None
    if outdoor_col is None:
        print(f"Note: no '{args.outdoor_col}' column found in the data -- "
              f"using a constant fallback of {cfg.fallback_outdoor_temp_c} deg C.")

    try:
        sim = run_simulation(pred_df, cfg=cfg, outdoor_temp_column=args.outdoor_col)
    except ValueError as e:
        print(f"Could not run the HVAC simulation: {e}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    forecast_path = out_dir / "occupancy_forecast.csv"
    comparison_path = out_dir / "hvac_comparison.csv"
    summary_path = out_dir / "hvac_summary.csv"

    pred_df.to_csv(forecast_path, index=False)
    sim["comparison_df"].to_csv(comparison_path, index=False)
    sim["summary_df"].to_csv(summary_path, index=False)

    print(f"\nWrote:\n  {forecast_path}\n  {comparison_path}\n  {summary_path}\n")

    print("=== KPIs: Scheduled vs Predictive ===")
    print(sim["summary_df"].round(3).to_string(index=False))

    energy_saving = (
        sim["scheduled_kpi"]["Total HVAC Energy (kWh)"]
        - sim["predictive_kpi"]["Total HVAC Energy (kWh)"]
    )
    print(f"\nPredictive strategy energy vs scheduled: {-energy_saving:+.2f} kWh "
          f"(negative = predictive uses less)")
    print(f"Outdoor temperature source: {sim['outdoor_source']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
