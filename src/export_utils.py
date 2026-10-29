"""
export_utils.py — CSV and JSON export utilities for GridSwarm.
"""

from __future__ import annotations
import json
import io
import numpy as np
import pandas as pd
from typing import Optional

from .energy_model import EnergySchedule
from .config import SystemConfig
from .evaluation import ComparisonResult


def schedule_to_dataframe(schedule: EnergySchedule) -> pd.DataFrame:
    """
    Convert an EnergySchedule to a pandas DataFrame.

    Columns:
      hour, grid_import_kw, solar_used_kw, solar_curtailed_kw,
      battery_power_kw, battery_soc_pct,
      ev_0_power_kw, ev_1_power_kw, ...,
      tariff_eur_per_kwh, emissions_kg_per_kwh,
      total_ev_power_kw, total_cost_increment_eur
    """
    cfg = schedule.config
    T = cfg.n_intervals
    hours = cfg.time_hours
    dt = cfg.time_step_h

    data = {
        "hour": hours,
        "grid_import_kw": schedule.grid_import_kw,
        "solar_used_kw": schedule.solar_used_kw,
        "solar_curtailed_kw": schedule.solar_curtailed_kw,
        "battery_power_kw": schedule.battery_power_kw,
        "battery_soc_pct": schedule.battery_soc[:T] * 100,
        "total_ev_power_kw": schedule.total_ev_power_kw,
    }

    for i, ev in enumerate(cfg.evs):
        data[f"ev_{ev.ev_id}_power_kw"] = schedule.ev_power_kw[i]
        data[f"ev_{ev.ev_id}_soc_pct"] = schedule.ev_soc[i, :T] * 100

    if cfg.tariff_profile is not None:
        data["tariff_eur_per_kwh"] = cfg.tariff_profile
        data["cost_increment_eur"] = schedule.grid_import_kw * cfg.tariff_profile * dt

    if cfg.emissions_profile is not None:
        data["emissions_kg_per_kwh"] = cfg.emissions_profile
        data["emissions_increment_kg"] = schedule.grid_import_kw * cfg.emissions_profile * dt

    return pd.DataFrame(data)


def comparison_to_dataframe(comparison: ComparisonResult) -> pd.DataFrame:
    """
    Convert ComparisonResult to a summary DataFrame for download.
    """
    b = comparison.baseline
    p = comparison.pso
    imp = comparison.improvements

    rows = [
        {"Metric": "Total Cost (€)",
         "Baseline": round(b.total_cost_eur, 3),
         "PSO": round(p.total_cost_eur, 3),
         "Improvement (%)": round(imp.get("cost_improvement_pct", 0), 2)},
        {"Metric": "Peak Demand (kW)",
         "Baseline": round(b.peak_demand_kw, 3),
         "PSO": round(p.peak_demand_kw, 3),
         "Improvement (%)": round(imp.get("peak_reduction_pct", 0), 2)},
        {"Metric": "CO2 Emissions (kg)",
         "Baseline": round(b.total_emissions_kg, 3),
         "PSO": round(p.total_emissions_kg, 3),
         "Improvement (%)": round(imp.get("emissions_reduction_pct", 0), 2)},
        {"Metric": "Solar Utilization (%)",
         "Baseline": round(b.solar_utilization_pct, 2),
         "PSO": round(p.solar_utilization_pct, 2),
         "Improvement (%)": round(imp.get("solar_improvement_pct", 0), 2)},
        {"Metric": "EV Target Satisfaction (%)",
         "Baseline": round(b.ev_satisfaction_pct, 2),
         "PSO": round(p.ev_satisfaction_pct, 2),
         "Improvement (%)": None},
        {"Metric": "Grid Energy (kWh)",
         "Baseline": round(b.total_grid_energy_kwh, 3),
         "PSO": round(p.total_grid_energy_kwh, 3),
         "Improvement (%)": None},
        {"Metric": "Feasible",
         "Baseline": "Yes" if b.is_feasible else "No",
         "PSO": "Yes" if p.is_feasible else "No",
         "Improvement (%)": None},
    ]
    if p.runtime_seconds is not None:
        rows.append({"Metric": "PSO Runtime (s)",
                     "Baseline": "—",
                     "PSO": round(p.runtime_seconds, 3),
                     "Improvement (%)": None})
    return pd.DataFrame(rows)


def config_to_dict(config: SystemConfig) -> dict:
    """Convert SystemConfig to a JSON-serializable dictionary."""
    return {
        "time_horizon_h": config.time_horizon_h,
        "time_step_h": config.time_step_h,
        "n_intervals": config.n_intervals,
        "evs": [
            {
                "ev_id": ev.ev_id,
                "arrival_hour": ev.arrival_hour,
                "departure_hour": ev.departure_hour,
                "battery_capacity_kwh": ev.battery_capacity_kwh,
                "initial_soc": ev.initial_soc,
                "target_soc": ev.target_soc,
                "max_charge_power_kw": ev.max_charge_power_kw,
                "energy_needed_kwh": round(ev.energy_needed_kwh, 3),
            }
            for ev in config.evs
        ],
        "battery": (
            {
                "capacity_kwh": config.battery.capacity_kwh,
                "initial_soc": config.battery.initial_soc,
                "min_soc": config.battery.min_soc,
                "max_soc": config.battery.max_soc,
                "max_charge_power_kw": config.battery.max_charge_power_kw,
                "max_discharge_power_kw": config.battery.max_discharge_power_kw,
                "charge_efficiency": config.battery.charge_efficiency,
                "discharge_efficiency": config.battery.discharge_efficiency,
            }
            if config.battery else None
        ),
        "grid": {
            "max_import_kw": config.grid.max_import_kw,
        },
        "solar_profile_kw": (
            config.solar_profile_kw.tolist() if config.solar_profile_kw is not None else None
        ),
        "tariff_profile": (
            config.tariff_profile.tolist() if config.tariff_profile is not None else None
        ),
        "emissions_profile": (
            config.emissions_profile.tolist() if config.emissions_profile is not None else None
        ),
    }


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return a DataFrame as UTF-8 encoded CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def dict_to_json_bytes(data: dict) -> bytes:
    """Return a dict as UTF-8 encoded JSON bytes."""
    return json.dumps(data, indent=2).encode("utf-8")
