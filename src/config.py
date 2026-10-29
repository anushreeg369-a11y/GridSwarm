"""
config.py — System configuration dataclasses for GridSwarm.

All physical quantities use:
  - Power:  kW  (kilowatts)
  - Energy: kWh (kilowatt-hours)
  - Time:   hours

Sign convention for stationary battery:
  - Positive battery_power → charging (absorbing energy from the bus)
  - Negative battery_power → discharging (injecting energy into the bus)

Grid import is always ≥ 0 (export not modelled in the base system).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class EVConfig:
    """Configuration for a single Electric Vehicle."""
    ev_id: int
    arrival_hour: float          # Hour of day EV arrives (0–23)
    departure_hour: float        # Hour of day EV must leave (arrival < departure)
    battery_capacity_kwh: float  # Total battery capacity (kWh)
    initial_soc: float           # Initial state-of-charge as fraction [0, 1]
    target_soc: float            # Required SOC at departure [0, 1]
    max_charge_power_kw: float   # Maximum charging power (kW)

    def __post_init__(self) -> None:
        assert 0.0 <= self.arrival_hour < 24.0, "Arrival hour must be in [0, 24)"
        assert self.arrival_hour < self.departure_hour <= 24.0, (
            f"EV {self.ev_id}: departure must be after arrival and ≤ 24h"
        )
        assert self.battery_capacity_kwh > 0, "Battery capacity must be positive"
        assert 0.0 <= self.initial_soc <= 1.0, "Initial SOC must be in [0, 1]"
        assert 0.0 <= self.target_soc <= 1.0, "Target SOC must be in [0, 1]"
        assert self.max_charge_power_kw > 0, "Max charge power must be positive"

    @property
    def energy_needed_kwh(self) -> float:
        """Energy required to reach target SOC from initial SOC (kWh)."""
        return max(0.0, (self.target_soc - self.initial_soc) * self.battery_capacity_kwh)

    @property
    def max_energy_kwh(self) -> float:
        """Maximum energy that can be stored (target_soc × capacity)."""
        return self.target_soc * self.battery_capacity_kwh


@dataclass
class BatteryConfig:
    """Configuration for the stationary battery storage system."""
    capacity_kwh: float          # Total energy capacity (kWh)
    initial_soc: float           # Initial state-of-charge as fraction [0, 1]
    min_soc: float               # Minimum allowed SOC fraction
    max_soc: float               # Maximum allowed SOC fraction
    max_charge_power_kw: float   # Maximum charging power (kW, positive)
    max_discharge_power_kw: float  # Maximum discharging power (kW, positive magnitude)
    charge_efficiency: float     # Charging round-trip efficiency [0, 1]
    discharge_efficiency: float  # Discharging round-trip efficiency [0, 1]

    def __post_init__(self) -> None:
        assert self.capacity_kwh > 0
        assert 0.0 <= self.min_soc < self.max_soc <= 1.0
        assert 0.0 <= self.initial_soc <= 1.0
        assert self.max_charge_power_kw > 0
        assert self.max_discharge_power_kw > 0
        assert 0.0 < self.charge_efficiency <= 1.0
        assert 0.0 < self.discharge_efficiency <= 1.0

    @property
    def max_energy_kwh(self) -> float:
        return self.max_soc * self.capacity_kwh

    @property
    def min_energy_kwh(self) -> float:
        return self.min_soc * self.capacity_kwh


@dataclass
class GridConfig:
    """Configuration for the utility grid connection."""
    max_import_kw: float         # Maximum grid import power (kW)
    # tariff_profile and emissions_profile are set in SystemConfig


@dataclass
class SystemConfig:
    """
    Top-level system configuration for one simulation run.

    time_horizon_h: total simulation horizon in hours (default 24)
    time_step_h:    interval duration in hours (e.g. 1.0 for 60-min steps)
    """
    time_horizon_h: float = 24.0
    time_step_h: float = 1.0
    evs: List[EVConfig] = field(default_factory=list)
    battery: Optional[BatteryConfig] = None
    grid: GridConfig = field(default_factory=lambda: GridConfig(max_import_kw=100.0))

    # Per-interval profiles (length must equal n_intervals)
    solar_profile_kw: Optional[np.ndarray] = None   # Solar PV generation (kW)
    tariff_profile: Optional[np.ndarray] = None     # Electricity price (€/kWh)
    emissions_profile: Optional[np.ndarray] = None  # Grid carbon intensity (kg CO₂/kWh)

    def __post_init__(self) -> None:
        assert self.time_step_h > 0
        assert self.time_horizon_h > 0

    @property
    def n_intervals(self) -> int:
        """Number of time intervals in the simulation."""
        return int(round(self.time_horizon_h / self.time_step_h))

    @property
    def n_evs(self) -> int:
        return len(self.evs)

    @property
    def time_hours(self) -> np.ndarray:
        """Array of interval start times (hours)."""
        return np.arange(self.n_intervals) * self.time_step_h

    def validate(self) -> List[str]:
        """Return a list of validation error strings (empty = valid)."""
        errors: List[str] = []
        T = self.n_intervals

        if self.solar_profile_kw is not None and len(self.solar_profile_kw) != T:
            errors.append(f"solar_profile length {len(self.solar_profile_kw)} != n_intervals {T}")
        if self.tariff_profile is not None and len(self.tariff_profile) != T:
            errors.append(f"tariff_profile length {len(self.tariff_profile)} != n_intervals {T}")
        if self.emissions_profile is not None and len(self.emissions_profile) != T:
            errors.append(f"emissions_profile length {len(self.emissions_profile)} != n_intervals {T}")

        for ev in self.evs:
            available_h = ev.departure_hour - ev.arrival_hour
            max_energy_possible = ev.max_charge_power_kw * available_h
            if max_energy_possible < ev.energy_needed_kwh - 1e-6:
                errors.append(
                    f"EV {ev.ev_id}: infeasible — needs {ev.energy_needed_kwh:.2f} kWh "
                    f"but max possible is {max_energy_possible:.2f} kWh"
                )

        return errors
