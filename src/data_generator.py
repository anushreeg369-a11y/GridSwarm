"""
data_generator.py — Synthetic profile generation for GridSwarm.

Generates:
- Solar PV generation profile (bell-curve peaking at noon)
- Time-of-use electricity tariff profile
- Grid carbon-intensity emissions profile
- Default EV fleet configuration
- Default stationary battery configuration
- Default system configuration

All data is clearly labelled as SIMULATED / SYNTHETIC.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from .config import SystemConfig, EVConfig, BatteryConfig, GridConfig


def _gaussian_solar(
    hours: np.ndarray,
    peak_kw: float = 8.0,
    peak_hour: float = 12.5,
    sigma_h: float = 2.5,
) -> np.ndarray:
    """Gaussian bell-curve solar profile."""
    profile = peak_kw * np.exp(-0.5 * ((hours - peak_hour) / sigma_h) ** 2)
    profile[profile < 0.0] = 0.0
    # Zero out nighttime (before 5:30 and after 20:00)
    profile[hours < 5.5] = 0.0
    profile[hours > 20.0] = 0.0
    return profile


def generate_solar_profile(
    n_intervals: int,
    time_step_h: float,
    peak_kw: float = 8.0,
    peak_hour: float = 12.5,
    sigma_h: float = 2.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a synthetic solar PV generation profile.

    Parameters
    ----------
    n_intervals : number of time steps
    time_step_h : duration of each step (hours)
    peak_kw     : peak solar power (kW)
    peak_hour   : hour at which solar peaks (e.g. 12.5 = 12:30)
    sigma_h     : Gaussian width (hours)
    seed        : random seed for optional noise addition

    Returns
    -------
    np.ndarray of shape (n_intervals,) in kW
    """
    hours = np.arange(n_intervals) * time_step_h
    base = _gaussian_solar(hours, peak_kw, peak_hour, sigma_h)
    if seed is not None:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 0.05 * peak_kw, size=n_intervals)
        base = np.clip(base + noise, 0.0, None)
        # Re-zero nighttime after noise
        base[hours < 5.5] = 0.0
        base[hours > 20.0] = 0.0
    return base


def generate_tariff_profile(
    n_intervals: int,
    time_step_h: float,
) -> np.ndarray:
    """
    Generate a synthetic time-of-use electricity tariff (€/kWh).

    Periods:
    - Off-peak night  (00:00–07:00): 0.10 €/kWh
    - Morning ramp    (07:00–09:00): rising to 0.28 €/kWh
    - Day rate        (09:00–17:00): 0.22 €/kWh
    - Evening peak    (17:00–21:00): 0.32 €/kWh
    - Late evening    (21:00–24:00): 0.14 €/kWh
    """
    hours = np.arange(n_intervals) * time_step_h
    tariff = np.zeros(n_intervals)
    for i, h in enumerate(hours):
        if h < 7.0:
            tariff[i] = 0.10
        elif h < 9.0:
            # Linear ramp from 0.10 to 0.28
            tariff[i] = 0.10 + (h - 7.0) / 2.0 * 0.18
        elif h < 17.0:
            tariff[i] = 0.22
        elif h < 21.0:
            tariff[i] = 0.32
        else:
            tariff[i] = 0.14
    return tariff


def generate_emissions_profile(
    n_intervals: int,
    time_step_h: float,
) -> np.ndarray:
    """
    Generate a synthetic grid carbon intensity profile (kg CO₂/kWh).

    Typically higher when renewables are low (night/early morning)
    and lower at solar peak hours.
    """
    hours = np.arange(n_intervals) * time_step_h
    # Base: 0.35 kg CO2/kWh, reduced around solar peak
    emissions = 0.35 - 0.15 * np.exp(-0.5 * ((hours - 12.5) / 3.0) ** 2)
    # Morning peak (high gas generation)
    morning_peak = 0.05 * np.exp(-0.5 * ((hours - 8.0) / 1.5) ** 2)
    emissions += morning_peak
    # Evening peak (high gas/coal)
    evening_peak = 0.08 * np.exp(-0.5 * ((hours - 18.5) / 1.5) ** 2)
    emissions += evening_peak
    return np.clip(emissions, 0.10, 0.55)


def generate_default_evs(time_step_h: float = 1.0) -> list[EVConfig]:
    """
    Generate a default fleet of 5 EVs with diverse schedules.

    All values are synthetic/illustrative.
    """
    evs = [
        EVConfig(
            ev_id=0,
            arrival_hour=7.0,
            departure_hour=17.0,
            battery_capacity_kwh=40.0,
            initial_soc=0.25,
            target_soc=0.90,
            max_charge_power_kw=7.4,
        ),
        EVConfig(
            ev_id=1,
            arrival_hour=8.0,
            departure_hour=18.0,
            battery_capacity_kwh=60.0,
            initial_soc=0.30,
            target_soc=0.80,
            max_charge_power_kw=11.0,
        ),
        EVConfig(
            ev_id=2,
            arrival_hour=6.0,
            departure_hour=14.0,
            battery_capacity_kwh=30.0,
            initial_soc=0.50,
            target_soc=0.95,
            max_charge_power_kw=7.4,
        ),
        EVConfig(
            ev_id=3,
            arrival_hour=17.0,
            departure_hour=23.0,
            battery_capacity_kwh=75.0,
            initial_soc=0.15,
            target_soc=0.70,
            max_charge_power_kw=22.0,
        ),
        EVConfig(
            ev_id=4,
            arrival_hour=9.0,
            departure_hour=16.0,
            battery_capacity_kwh=50.0,
            initial_soc=0.40,
            target_soc=0.85,
            max_charge_power_kw=11.0,
        ),
    ]
    return evs


def generate_default_battery() -> BatteryConfig:
    """Default stationary battery configuration (synthetic data)."""
    return BatteryConfig(
        capacity_kwh=20.0,
        initial_soc=0.50,
        min_soc=0.10,
        max_soc=0.90,
        max_charge_power_kw=5.0,
        max_discharge_power_kw=5.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )


def build_default_config(
    time_step_h: float = 1.0,
    time_horizon_h: float = 24.0,
    solar_peak_kw: float = 8.0,
    seed: Optional[int] = 42,
) -> SystemConfig:
    """
    Build a fully-populated default SystemConfig for the 24-hour simulation.

    This is the primary entry point for the default scenario.
    All data is SYNTHETIC / SIMULATED.
    """
    n_intervals = int(round(time_horizon_h / time_step_h))

    solar = generate_solar_profile(n_intervals, time_step_h, peak_kw=solar_peak_kw, seed=None)
    tariff = generate_tariff_profile(n_intervals, time_step_h)
    emissions = generate_emissions_profile(n_intervals, time_step_h)

    config = SystemConfig(
        time_horizon_h=time_horizon_h,
        time_step_h=time_step_h,
        evs=generate_default_evs(time_step_h),
        battery=generate_default_battery(),
        grid=GridConfig(max_import_kw=50.0),
        solar_profile_kw=solar,
        tariff_profile=tariff,
        emissions_profile=emissions,
    )
    return config


def build_infeasible_config(time_step_h: float = 1.0) -> SystemConfig:
    """
    Build a scenario with an intentionally infeasible EV requirement.
    Used for testing infeasibility detection.
    """
    n_intervals = int(round(24.0 / time_step_h))
    solar = generate_solar_profile(n_intervals, time_step_h)
    tariff = generate_tariff_profile(n_intervals, time_step_h)
    emissions = generate_emissions_profile(n_intervals, time_step_h)

    evs = [
        EVConfig(
            ev_id=0,
            arrival_hour=8.0,
            departure_hour=9.0,   # Only 1 hour
            battery_capacity_kwh=50.0,
            initial_soc=0.10,
            target_soc=0.90,      # Needs 40 kWh in 1 hour at max 7.4 kW → infeasible
            max_charge_power_kw=7.4,
        ),
        EVConfig(
            ev_id=1,
            arrival_hour=10.0,
            departure_hour=16.0,
            battery_capacity_kwh=40.0,
            initial_soc=0.30,
            target_soc=0.80,
            max_charge_power_kw=11.0,
        ),
    ]

    config = SystemConfig(
        time_horizon_h=24.0,
        time_step_h=time_step_h,
        evs=evs,
        battery=generate_default_battery(),
        grid=GridConfig(max_import_kw=50.0),
        solar_profile_kw=solar,
        tariff_profile=tariff,
        emissions_profile=emissions,
    )
    return config
