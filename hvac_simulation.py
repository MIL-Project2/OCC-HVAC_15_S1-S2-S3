"""
hvac_simulation.py

1R1C HVAC simulator: scheduled baseline vs thermal-predictive control.

This module knows nothing about how occupancy_now / occupancy_forecast
were produced -- it only needs a DataFrame with the columns described
in `run_simulation`. That decoupling is what lets the same simulator
run on a historical backtest (with a known future to validate against)
or on a live Streamlit upload (with no known future).

Ported, unchanged in behaviour, from Occupancy_Prediction_Ver1_5.ipynb,
section "HVAC Model".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class HVACConfig:
    # Time resolution of the input data
    simulation_step_min: float = 5.0

    # Predictive control is refreshed once per forecast horizon.
    prediction_horizon_min: float = 15.0

    # Comfort band
    comfort_min_c: float = 22.8
    comfort_max_c: float = 25.8

    # Scheduled HVAC baseline
    schedule_start_hour: float = 8.0
    schedule_end_hour: float = 18.0
    schedule_weekdays_only: bool = True

    # Outdoor temperature fallback, used only if no outdoor-temperature
    # column is supplied.
    fallback_outdoor_temp_c: float = 33.0

    # 1R1C building parameters -- PROTOTYPE VALUES.
    # These must be calibrated against the real room before any energy
    # or comfort number from this simulator is reported as a result.
    r_thermal_k_per_w: float = 0.01
    c_thermal_j_per_k: float = 1_000_000.0

    # Occupant heat gain
    heat_gain_per_person_w: float = 100.0
    other_occupied_w: float = 0.0
    other_unoccupied_w: float = 0.0

    # HVAC hardware
    q_cooling_max_w: float = 5000.0
    cooling_kp_w_per_k: float = 2000.0
    cop: float = 3.0
    fan_power_w: float = 0.0
    initial_room_temp_c: float = 33.0

    # Predictive controller
    predictive_occupancy_threshold: float = 1.0
    forecast_trigger_margin_c: float = 0.05
    predictive_bisection_iterations: int = 24

    @property
    def dt_seconds(self) -> float:
        return self.simulation_step_min * 60

    @property
    def dt_hours(self) -> float:
        return self.simulation_step_min / 60

    @property
    def thermostat_setpoint_c(self) -> float:
        return (self.comfort_min_c + self.comfort_max_c) / 2

    @property
    def thermostat_deadband_c(self) -> float:
        return self.comfort_max_c - self.comfort_min_c

    @property
    def predictive_target_c(self) -> float:
        return self.thermostat_setpoint_c

    @property
    def control_steps(self) -> int:
        return max(1, int(self.prediction_horizon_min / self.simulation_step_min))

    @property
    def prediction_horizon_steps(self) -> int:
        return max(1, int(self.prediction_horizon_min / self.simulation_step_min))

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = HVACConfig()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def classify_occupancy_state(occupancy_value):
    """S1/S2/S3 labels for interpretation and dashboarding."""
    occupancy_value = max(float(occupancy_value), 0.0)
    occupancy_control = int(np.rint(occupancy_value))

    if occupancy_control == 0:
        state, description = 'S1', 'Vacant'
    elif occupancy_control <= 3:
        state, description = 'S2', 'Occupied Low'
    else:
        state, description = 'S3', 'Occupied High'

    return occupancy_control, state, description


def scheduled_enable(timestamp, cfg: HVACConfig):
    """Fixed-time baseline HVAC availability."""
    hour_decimal = timestamp.hour + timestamp.minute / 60
    if cfg.schedule_weekdays_only and timestamp.weekday() >= 5:
        return False
    return cfg.schedule_start_hour <= hour_decimal < cfg.schedule_end_hour


def working_hours_mask(timestamp: pd.Series, cfg: HVACConfig) -> pd.Series:
    """
    Vectorised version of scheduled_enable()'s window, for filtering a
    results DataFrame down to the working-hours window (cfg.schedule_start_hour
    to cfg.schedule_end_hour, weekdays only if cfg.schedule_weekdays_only).

    Used by calculate_kpis() so KPIs reflect working hours only, regardless
    of which strategy ('scheduled' or 'predictive') actually controlled the
    HVAC at any given timestep.
    """
    timestamp = pd.to_datetime(timestamp)
    hour_decimal = timestamp.dt.hour + timestamp.dt.minute / 60
    mask = (hour_decimal >= cfg.schedule_start_hour) & (hour_decimal < cfg.schedule_end_hour)
    if cfg.schedule_weekdays_only:
        mask &= timestamp.dt.weekday < 5
    return mask


def forecast_1r1c_temperature(
    T_start_C, T_outdoor_C, occ_now, occ_future, cfg: HVACConfig,
    Q_cooling_W=0.0, horizon_steps=None
):
    """
    Forecast future room temperature with the 1R1C model, blending
    occupancy from its CURRENT measured value (occ_now) to its
    forecast-horizon-ahead predicted value (occ_future) across the
    horizon, rather than assuming occupancy is constant.
    """
    if horizon_steps is None:
        horizon_steps = cfg.prediction_horizon_steps

    T_pred_C = float(T_start_C)
    occ_now = max(float(occ_now), 0.0)
    occ_future = max(float(occ_future), 0.0)

    for step in range(horizon_steps):
        frac = (step + 1) / horizon_steps
        occ_step = occ_now + (occ_future - occ_now) * frac

        Q_occ_pred_W = occ_step * cfg.heat_gain_per_person_w
        Q_other_pred_W = (
            cfg.other_occupied_w
            if occ_step >= cfg.predictive_occupancy_threshold
            else cfg.other_unoccupied_w
        )

        Q_envelope_W = (float(T_outdoor_C) - T_pred_C) / cfg.r_thermal_k_per_w
        Q_net_W = Q_envelope_W + Q_occ_pred_W + Q_other_pred_W - float(Q_cooling_W)
        T_pred_C += (Q_net_W / cfg.c_thermal_j_per_k) * cfg.dt_seconds

    return T_pred_C


def minimum_predictive_cooling(T_room_C, T_outdoor_C, occ_now, occ_future, cfg: HVACConfig):
    """
    1R1C-based predictive HVAC decision that considers BOTH the
    current measured occupancy and the forecast-horizon-ahead
    occupancy prediction.
    """
    occ_now_c, _, _ = classify_occupancy_state(occ_now)
    occ_future_c, state_label, state_description = classify_occupancy_state(occ_future)

    T_future_no_hvac_C = forecast_1r1c_temperature(
        T_room_C, T_outdoor_C, occ_now_c, occ_future_c, cfg, Q_cooling_W=0.0
    )

    if max(occ_now_c, occ_future_c) < cfg.predictive_occupancy_threshold:
        return 0.0, T_future_no_hvac_C, T_future_no_hvac_C, state_label, state_description

    if T_future_no_hvac_C <= cfg.comfort_max_c + cfg.forecast_trigger_margin_c:
        return 0.0, T_future_no_hvac_C, T_future_no_hvac_C, state_label, state_description

    low_W, high_W = 0.0, cfg.q_cooling_max_w

    T_at_max_C = forecast_1r1c_temperature(
        T_room_C, T_outdoor_C, occ_now_c, occ_future_c, cfg, Q_cooling_W=high_W
    )
    if T_at_max_C > cfg.predictive_target_c:
        return high_W, T_future_no_hvac_C, T_at_max_C, state_label, state_description

    for _ in range(cfg.predictive_bisection_iterations):
        mid_W = 0.5 * (low_W + high_W)
        T_mid_C = forecast_1r1c_temperature(
            T_room_C, T_outdoor_C, occ_now_c, occ_future_c, cfg, Q_cooling_W=mid_W
        )
        if T_mid_C > cfg.predictive_target_c:
            low_W = mid_W
        else:
            high_W = mid_W

    Q_command_W = high_W
    T_future_with_command_C = forecast_1r1c_temperature(
        T_room_C, T_outdoor_C, occ_now_c, occ_future_c, cfg, Q_cooling_W=Q_command_W
    )

    return Q_command_W, T_future_no_hvac_C, T_future_with_command_C, state_label, state_description


def thermostat_cooling(T_room_C, hvac_enable, cooling_on_memory, cfg: HVACConfig):
    """Common thermostat logic used by BOTH strategies."""
    upper_limit = cfg.thermostat_setpoint_c + cfg.thermostat_deadband_c / 2
    lower_limit = cfg.thermostat_setpoint_c - cfg.thermostat_deadband_c / 2

    cooling_on = bool(cooling_on_memory)

    if not hvac_enable:
        cooling_on = False
    elif T_room_C >= upper_limit:
        cooling_on = True
    elif T_room_C <= lower_limit:
        cooling_on = False

    if hvac_enable and cooling_on:
        temperature_error = max(T_room_C - cfg.thermostat_setpoint_c, 0.0)
        Q_cooling_W = float(np.clip(
            cfg.cooling_kp_w_per_k * temperature_error, 0.0, cfg.q_cooling_max_w
        ))
    else:
        Q_cooling_W = 0.0

    return cooling_on, Q_cooling_W, lower_limit, upper_limit


def resolve_outdoor_temperature(occupancy_df: pd.DataFrame, cfg: HVACConfig, column: str | None = None):
    """
    Return an outdoor-temperature Series aligned to occupancy_df.

    If `column` is given and present, use it (forward/back-filling any
    gaps). Otherwise fall back to a constant, and say so -- silently
    assuming a constant outdoor temperature would make the "predictive
    pre-cooling" results look better or worse than they'd really be.
    """
    if column and column in occupancy_df.columns:
        outdoor = pd.to_numeric(occupancy_df[column], errors='coerce')
        if outdoor.notna().any():
            outdoor = outdoor.ffill().bfill().fillna(cfg.fallback_outdoor_temp_c)
            return outdoor.astype(float), column

    outdoor = pd.Series(cfg.fallback_outdoor_temp_c, index=occupancy_df.index, dtype=float)
    return outdoor, 'constant_fallback'


# ============================================================
# CORE SIMULATOR
# ============================================================

def run_1r1c_strategy(sim_data: pd.DataFrame, strategy_name: str, cfg: HVACConfig) -> pd.DataFrame:
    """
    Run one closed-loop 1R1C HVAC scenario.

    strategy_name:
        'scheduled'  -> fixed timetable + thermostat baseline
        'predictive' -> occupancy_now + occupancy_forecast drive the
                        1R1C future-temperature control decision

    Physical room heat gain ALWAYS uses the CURRENT MEASURED occupancy
    (occupancy_now), regardless of strategy. The forecast is used only
    to decide whether to pre-cool.

    Required columns in sim_data: timestamp, occupancy_now,
    occupancy_forecast_15min, outdoor_temperature_C
    """
    if strategy_name not in ('scheduled', 'predictive'):
        raise ValueError("strategy_name must be 'scheduled' or 'predictive'")

    T_room_C = cfg.initial_room_temp_c
    scheduled_cooling_on = False

    current_predictive_Q_W = 0.0
    current_T_future_no_hvac_C = np.nan
    current_T_future_with_command_C = np.nan
    current_state_label = 'S1'
    current_state_description = 'Vacant'

    cumulative_energy_kWh = 0.0
    results = []

    for i, row in sim_data.reset_index(drop=True).iterrows():
        timestamp = row['timestamp']
        occ_now = max(float(row['occupancy_now']), 0.0)
        occ_future = max(float(row['occupancy_forecast_15min']), 0.0)
        T_outdoor_C = float(row['outdoor_temperature_C'])

        occ_now_control, occ_now_state, occ_now_description = classify_occupancy_state(occ_now)

        if strategy_name == 'scheduled':
            control_update = True
            hvac_enable = scheduled_enable(timestamp, cfg)

            scheduled_cooling_on, Q_cooling_W, _, _ = thermostat_cooling(
                T_room_C, hvac_enable, scheduled_cooling_on, cfg
            )
            cooling_on = bool(Q_cooling_W > 0)

            T_future_no_hvac_C = forecast_1r1c_temperature(
                T_room_C, T_outdoor_C, occ_now_control, occ_now_control, cfg, Q_cooling_W=0.0
            )
            T_future_with_command_C = np.nan
            predictive_Q_command_W = 0.0
            occupancy_state, state_description = occ_now_state, occ_now_description

        else:
            control_update = (i % cfg.control_steps == 0)

            if control_update:
                (
                    current_predictive_Q_W,
                    current_T_future_no_hvac_C,
                    current_T_future_with_command_C,
                    current_state_label,
                    current_state_description
                ) = minimum_predictive_cooling(T_room_C, T_outdoor_C, occ_now, occ_future, cfg)

            predictive_Q_command_W = current_predictive_Q_W
            T_future_no_hvac_C = current_T_future_no_hvac_C
            T_future_with_command_C = current_T_future_with_command_C
            occupancy_state = current_state_label
            state_description = current_state_description

            if T_room_C <= cfg.comfort_min_c:
                Q_cooling_W = 0.0
            else:
                Q_cooling_W = float(np.clip(predictive_Q_command_W, 0.0, cfg.q_cooling_max_w))

            cooling_on = bool(Q_cooling_W > 0)
            hvac_enable = cooling_on

        # Physical heat gain ALWAYS uses the true, currently-measured occupancy.
        Q_occupancy_W = occ_now * cfg.heat_gain_per_person_w
        Q_other_W = cfg.other_occupied_w if occ_now > 0 else cfg.other_unoccupied_w
        Q_envelope_W = (T_outdoor_C - T_room_C) / cfg.r_thermal_k_per_w
        Q_net_W = Q_envelope_W + Q_occupancy_W + Q_other_W - Q_cooling_W
        dT_C = (Q_net_W / cfg.c_thermal_j_per_k) * cfg.dt_seconds
        T_room_next_C = T_room_C + dT_C

        compressor_power_W = Q_cooling_W / cfg.cop if Q_cooling_W > 0 else 0.0
        fan_power_W = cfg.fan_power_w if cooling_on else 0.0
        hvac_electric_power_W = compressor_power_W + fan_power_W

        interval_energy_kWh = hvac_electric_power_W * cfg.dt_hours / 1000
        cumulative_energy_kWh += interval_energy_kWh

        occupied_now = occ_now > 0
        comfort_ok = (cfg.comfort_min_c <= T_room_C <= cfg.comfort_max_c) if occupied_now else np.nan
        unnecessary_hvac_enable = bool(hvac_enable and not occupied_now)

        results.append({
            'timestamp': timestamp,
            'strategy': strategy_name,
            'occupancy_now': occ_now,
            'occupancy_now_state': occ_now_state,
            'occupancy_forecast_15min': occ_future,
            'occupancy_state_used_for_control': occupancy_state,
            'state_description_used_for_control': state_description,
            'control_update': int(control_update),
            'hvac_enable': int(hvac_enable),
            'cooling_on': int(cooling_on),
            'outdoor_temperature_C': T_outdoor_C,
            'room_temperature_C': T_room_C,
            'predicted_future_temp_no_hvac_C': T_future_no_hvac_C,
            'predicted_future_temp_with_command_C': T_future_with_command_C,
            'predictive_cooling_command_W': predictive_Q_command_W,
            'cooling_thermal_power_W': Q_cooling_W,
            'hvac_electric_power_W': hvac_electric_power_W,
            'hvac_energy_interval_kWh': interval_energy_kWh,
            'hvac_energy_cumulative_kWh': cumulative_energy_kWh,
            'occupied_now': int(occupied_now),
            'comfort_ok_when_occupied': comfort_ok,
            'unnecessary_hvac_enable': int(unnecessary_hvac_enable),
        })

        T_room_C = T_room_next_C

    return pd.DataFrame(results)


def calculate_kpis(result_df: pd.DataFrame, cfg: HVACConfig) -> dict:
    """
    KPIs are computed over working hours only -- rows outside the
    window defined by cfg.schedule_start_hour, cfg.schedule_end_hour,
    and cfg.schedule_weekdays_only are excluded before any KPI is
    calculated. This applies to both strategies' result_df (not just
    'scheduled'), so the comparison stays apples-to-apples.
    """
    result_df = result_df.loc[working_hours_mask(result_df['timestamp'], cfg)]

    if result_df.empty:
        return {
            'Total HVAC Energy (kWh)': np.nan,
            'Peak HVAC Power (kW)': np.nan,
            'Occupied Comfort Compliance (%)': np.nan,
            'Occupied Discomfort (h)': np.nan,
            'HVAC Enabled Runtime (h)': np.nan,
            'Cooling Runtime (h)': np.nan,
            'Unnecessary HVAC Enabled (h)': np.nan,
        }

    total_energy_kWh = result_df['hvac_energy_interval_kWh'].sum()
    peak_power_kW = result_df['hvac_electric_power_W'].max() / 1000

    occupied_mask = result_df['occupied_now'] == 1
    if occupied_mask.any():
        comfort_compliance_pct = (
            result_df.loc[occupied_mask, 'comfort_ok_when_occupied'].astype(float).mean() * 100
        )
        occupied_discomfort_hours = (
            (~result_df.loc[occupied_mask, 'comfort_ok_when_occupied'].astype(bool)).sum() * cfg.dt_hours
        )
    else:
        comfort_compliance_pct = np.nan
        occupied_discomfort_hours = np.nan

    return {
        'Total HVAC Energy (kWh)': total_energy_kWh,
        'Peak HVAC Power (kW)': peak_power_kW,
        'Occupied Comfort Compliance (%)': comfort_compliance_pct,
        'Occupied Discomfort (h)': occupied_discomfort_hours,
        'HVAC Enabled Runtime (h)': result_df['hvac_enable'].sum() * cfg.dt_hours,
        'Cooling Runtime (h)': result_df['cooling_on'].sum() * cfg.dt_hours,
        'Unnecessary HVAC Enabled (h)': result_df['unnecessary_hvac_enable'].sum() * cfg.dt_hours,
    }


def run_simulation(occupancy_df: pd.DataFrame, cfg: HVACConfig = None, outdoor_temp_column: str = 'outdoor_temperature_C'):
    """
    Run BOTH strategies and package the results.

    Parameters
    ----------
    occupancy_df : pd.DataFrame
        Must contain:
          - timestamp                 : the "now" time of each row
          - occupancy_now              : current, measured occupancy
          - occupancy_forecast_15min   : forecast occupancy N minutes
                                          ahead (N = cfg.prediction_horizon_min)
        Optional:
          - outdoor_temp_column (default 'outdoor_temperature_C'): if
            missing, a constant fallback is used (see HVACConfig).
    cfg : HVACConfig, optional
        Simulation / building / HVAC parameters. Defaults to
        DEFAULT_CONFIG (the same prototype values used during
        development -- calibrate before trusting the numbers).

    Returns
    -------
    dict with keys: scheduled_df, predictive_df, comparison_df,
    summary_df, scheduled_kpi, predictive_kpi, outdoor_source
    """
    cfg = cfg or DEFAULT_CONFIG

    required = ['timestamp', 'occupancy_now', 'occupancy_forecast_15min']
    missing = [c for c in required if c not in occupancy_df.columns]
    if missing:
        raise ValueError(f"occupancy_df is missing required column(s): {missing}")

    sim_input = occupancy_df.copy()
    sim_input['timestamp'] = pd.to_datetime(sim_input['timestamp'])
    sim_input['occupancy_now'] = pd.to_numeric(sim_input['occupancy_now'], errors='coerce').clip(lower=0)
    sim_input['occupancy_forecast_15min'] = pd.to_numeric(
        sim_input['occupancy_forecast_15min'], errors='coerce'
    ).clip(lower=0)

    outdoor_series, outdoor_source = resolve_outdoor_temperature(sim_input, cfg, outdoor_temp_column)
    sim_input['outdoor_temperature_C'] = outdoor_series

    sim_input = (
        sim_input
        .dropna(subset=['timestamp', 'occupancy_now', 'occupancy_forecast_15min'])
        .sort_values('timestamp')
        .reset_index(drop=True)
    )

    if sim_input.empty:
        raise ValueError("No usable rows left after cleaning -- check the input data.")

    scheduled_df = run_1r1c_strategy(sim_input, 'scheduled', cfg)
    predictive_df = run_1r1c_strategy(sim_input, 'predictive', cfg)

    scheduled_kpi = calculate_kpis(scheduled_df, cfg)
    predictive_kpi = calculate_kpis(predictive_df, cfg)

    summary_df = pd.DataFrame({
        'KPI': list(scheduled_kpi.keys()),
        'Scheduled HVAC': list(scheduled_kpi.values()),
        'Predictive HVAC': list(predictive_kpi.values()),
    })
    summary_df['Difference (Predictive - Scheduled)'] = (
        summary_df['Predictive HVAC'] - summary_df['Scheduled HVAC']
    )

    comparison_df = pd.DataFrame({
        'timestamp': sim_input['timestamp'],
        'occupancy_now': sim_input['occupancy_now'],
        'occupancy_forecast_15min': sim_input['occupancy_forecast_15min'],
        'outdoor_temperature_C': sim_input['outdoor_temperature_C'],
        'scheduled_room_temperature_C': scheduled_df['room_temperature_C'],
        'predictive_room_temperature_C': predictive_df['room_temperature_C'],
        'scheduled_hvac_power_kW': scheduled_df['hvac_electric_power_W'] / 1000,
        'predictive_hvac_power_kW': predictive_df['hvac_electric_power_W'] / 1000,
        'scheduled_energy_cumulative_kWh': scheduled_df['hvac_energy_cumulative_kWh'],
        'predictive_energy_cumulative_kWh': predictive_df['hvac_energy_cumulative_kWh'],
        'scheduled_comfort_ok': scheduled_df['comfort_ok_when_occupied'],
        'predictive_comfort_ok': predictive_df['comfort_ok_when_occupied'],
    })

    # If the caller had ground-truth "actual occupancy 15 min later"
    # (e.g. a historical backtest), carry it through for validation
    # plots -- purely informational, never used by the controller.
    if 'occupancy_actual_15min_later' in occupancy_df.columns:
        comparison_df['occupancy_actual_15min_later'] = pd.to_numeric(
            occupancy_df['occupancy_actual_15min_later'], errors='coerce'
        ).reindex(sim_input.index).values
        if 'forecast_time' in occupancy_df.columns:
            comparison_df['forecast_time'] = pd.to_datetime(
                occupancy_df['forecast_time']
            ).reindex(sim_input.index).values

    return {
        'scheduled_df': scheduled_df,
        'predictive_df': predictive_df,
        'comparison_df': comparison_df,
        'summary_df': summary_df,
        'scheduled_kpi': scheduled_kpi,
        'predictive_kpi': predictive_kpi,
        'outdoor_source': outdoor_source,
    }
