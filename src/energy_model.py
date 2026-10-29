"""
energy_model.py — Energy balance computation for GridSwarm.

This module decodes raw PSO particle vectors (or baseline schedules) into
physically interpreted energy schedules and computes the grid import residually.

Sign Convention (documented in config.py):
  - EV charging power:      ≥ 0 (always absorbing energy)
  - Battery power:          + = charging, − = discharging
  - Grid import:            ≥ 0 (no export in base model)
  - Solar:                  ≥ 0 (generation)

Energy Balance per interval t (when export is disabled):
  grid_import[t] + solar_used[t] + battery_discharge_energy[t]
    = ev_charging_energy[t] + battery_charging_energy[t] + curtailed_solar[t]

Which rearranges to:
  grid_import[t] = total_demand[t] - solar_used[t] - battery_discharge_energy[t]

where:
  total_demand[t]           = sum_ev(ev_power[ev, t]) + max(0, battery_power[t])
  battery_discharge_energy  = max(0, -battery_power[t]) × discharge_efficiency × Δt
  battery_charging_energy   = max(0, battery_power[t]) / charge_efficiency × Δt  (energy into battery)
  solar_used[t]             = min(solar[t], available demand)
  curtailed_solar[t]        = solar[t] - solar_used[t]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .config import SystemConfig, BatteryConfig


@dataclass
class EnergySchedule:
    """
    A fully decoded, physically interpreted energy schedule.

    All arrays have shape (n_intervals,) for scalars per interval,
    or (n_evs, n_intervals) for per-EV quantities.

    Units: kW for power, kWh for energy (= kW × time_step_h).
    """
    # Per-EV charging power [n_evs × n_intervals] (kW)
    ev_power_kw: np.ndarray

    # Battery signed power [n_intervals] (kW); + = charge, − = discharge
    battery_power_kw: np.ndarray

    # Solar generation used [n_intervals] (kW)
    solar_used_kw: np.ndarray

    # Solar curtailed (excess not usable) [n_intervals] (kW)
    solar_curtailed_kw: np.ndarray

    # Grid import [n_intervals] (kW), always ≥ 0
    grid_import_kw: np.ndarray

    # Battery State of Charge [n_intervals + 1] (fraction [0,1])
    battery_soc: np.ndarray

    # EV State of Charge [n_evs × (n_intervals + 1)] (fraction [0,1])
    ev_soc: np.ndarray

    # Reference to the system config used
    config: SystemConfig

    @property
    def n_intervals(self) -> int:
        return self.config.n_intervals

    @property
    def time_hours(self) -> np.ndarray:
        return self.config.time_hours

    @property
    def total_ev_power_kw(self) -> np.ndarray:
        """Sum of all EV charging powers per interval (kW)."""
        return self.ev_power_kw.sum(axis=0)

    @property
    def total_demand_kw(self) -> np.ndarray:
        """Total load on the bus (EV + battery charging) per interval (kW)."""
        return self.total_ev_power_kw + np.maximum(0.0, self.battery_power_kw)

    @property
    def battery_discharge_kw(self) -> np.ndarray:
        """Battery discharge power magnitude (kW), ≥ 0."""
        return np.maximum(0.0, -self.battery_power_kw)

    @property
    def battery_charge_kw(self) -> np.ndarray:
        """Battery charging power (kW), ≥ 0."""
        return np.maximum(0.0, self.battery_power_kw)

    def total_cost(self) -> float:
        """Total electricity cost (€)."""
        if self.config.tariff_profile is None:
            return 0.0
        return float(np.sum(self.grid_import_kw * self.config.tariff_profile * self.config.time_step_h))

    def peak_demand(self) -> float:
        """Peak grid import power (kW)."""
        return float(np.max(self.grid_import_kw))

    def total_emissions(self) -> float:
        """Total grid-related CO₂ emissions (kg)."""
        if self.config.emissions_profile is None:
            return 0.0
        return float(np.sum(self.grid_import_kw * self.config.emissions_profile * self.config.time_step_h))

    def solar_utilization(self) -> float:
        """Fraction of available solar energy actually used [0, 1]."""
        total_solar = float(np.sum(self.config.solar_profile_kw) * self.config.time_step_h)
        used = float(np.sum(self.solar_used_kw) * self.config.time_step_h)
        if total_solar < 1e-9:
            return 1.0
        return min(1.0, used / total_solar)


class EnergyModel:
    """
    Decodes a raw decision vector into an EnergySchedule by:
      1. Applying EV availability masks and power limits.
      2. Computing battery SoC trajectory.
      3. Allocating solar to demand.
      4. Computing grid import residually.
      5. Enforcing non-negativity of grid import (clipping any excess).

    This is a 'feasibility repair + decode' model, not a pure penalty model.
    """

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self._precompute_ev_masks()

    def _precompute_ev_masks(self) -> None:
        """
        Precompute boolean availability masks for each EV.
        mask[ev, t] = True if EV is present at interval t.
        """
        cfg = self.config
        T = cfg.n_intervals
        hours = cfg.time_hours  # start of each interval
        self.ev_masks = np.zeros((cfg.n_evs, T), dtype=bool)
        for i, ev in enumerate(cfg.evs):
            for t in range(T):
                interval_start = hours[t]
                interval_end = hours[t] + cfg.time_step_h
                # EV is available if interval overlaps [arrival, departure)
                self.ev_masks[i, t] = (
                    interval_end > ev.arrival_hour and interval_start < ev.departure_hour
                )

    def decode(self, raw_ev_power: np.ndarray, raw_battery_power: np.ndarray) -> EnergySchedule:
        """
        Decode and repair raw decision variables into a feasible EnergySchedule.

        Parameters
        ----------
        raw_ev_power : shape (n_evs, n_intervals) — unconstrained EV charging powers (kW)
        raw_battery_power : shape (n_intervals,) — unconstrained battery power (kW, signed)

        Returns
        -------
        EnergySchedule with all physical constraints enforced.
        """
        cfg = self.config
        dt = cfg.time_step_h
        T = cfg.n_intervals

        # --- Step 1: Repair EV charging powers ---
        ev_power = raw_ev_power.copy()

        for i, ev in enumerate(cfg.evs):
            # Zero out when EV is not present
            ev_power[i, ~self.ev_masks[i]] = 0.0
            # Clip to [0, max_charge_power]
            ev_power[i] = np.clip(ev_power[i], 0.0, ev.max_charge_power_kw)

            # Enforce energy target: scale down if over-charging, boost if under-charging
            ev_power[i] = self._repair_ev_energy(ev_power[i], ev, i, dt)

        # --- Step 2: Repair battery power ---
        battery_power = self._repair_battery(raw_battery_power, dt)

        # --- Step 3: Compute solar allocation ---
        solar_profile = cfg.solar_profile_kw if cfg.solar_profile_kw is not None else np.zeros(T)
        solar_used, solar_curtailed = self._allocate_solar(ev_power, battery_power, solar_profile)

        # --- Step 4: Compute grid import (residual) ---
        total_ev = ev_power.sum(axis=0)  # kW
        battery_charge = np.maximum(0.0, battery_power)   # kW going into battery
        battery_discharge = np.maximum(0.0, -battery_power)  # kW coming out of battery

        # Discharge energy available to the bus (after efficiency loss)
        batt_discharge_to_bus = battery_discharge * cfg.battery.discharge_efficiency if cfg.battery else battery_discharge

        # Net demand on grid: load minus solar used minus battery discharge to bus
        grid_import = total_ev + battery_charge - batt_discharge_to_bus - solar_used
        grid_import = np.maximum(0.0, grid_import)  # no negative import (no export)

        # Clip to grid capacity
        grid_import = np.minimum(grid_import, cfg.grid.max_import_kw)

        # --- Step 5: Compute battery SoC trajectory ---
        battery_soc = self._compute_battery_soc(battery_power, dt)

        # --- Step 6: Compute EV SoC trajectories ---
        ev_soc = self._compute_ev_soc(ev_power, dt)

        return EnergySchedule(
            ev_power_kw=ev_power,
            battery_power_kw=battery_power,
            solar_used_kw=solar_used,
            solar_curtailed_kw=solar_curtailed,
            grid_import_kw=grid_import,
            battery_soc=battery_soc,
            ev_soc=ev_soc,
            config=cfg,
        )

    def _repair_ev_energy(
        self,
        ev_power: np.ndarray,
        ev,
        ev_idx: int,
        dt: float,
    ) -> np.ndarray:
        """
        Repair EV charging schedule so that the total energy delivered
        matches the target (clamped to feasible maximum).

        Strategy:
        1. Compute current total energy.
        2. If > max allowable (target_soc × capacity), scale down proportionally.
        3. If < needed, distribute remaining energy uniformly over available slots.
        """
        mask = self.ev_masks[ev_idx]
        available_slots = np.where(mask)[0]
        if len(available_slots) == 0:
            return ev_power

        target_energy = ev.energy_needed_kwh
        max_energy = ev.max_energy_kwh - ev.initial_soc * ev.battery_capacity_kwh
        # max_energy = energy we can add = (target_soc - initial_soc) × capacity
        # (may be negative if initial_soc > target_soc, which means no charging needed)
        if target_energy <= 0.0:
            return np.zeros_like(ev_power)

        # Current delivered energy
        current_energy = float(ev_power[mask].sum() * dt)

        # Maximum possible energy given time windows
        max_possible = float(ev.max_charge_power_kw * len(available_slots) * dt)
        target_energy = min(target_energy, max_possible)  # cap at physically possible

        if current_energy > target_energy + 1e-6:
            # Scale down
            scale = target_energy / (current_energy + 1e-12)
            ev_power[mask] *= scale
        elif current_energy < target_energy - 1e-6:
            # Need to add energy: fill uniformly
            deficit = target_energy - current_energy
            extra_per_slot = deficit / (len(available_slots) * dt)
            ev_power[available_slots] = np.minimum(
                ev_power[available_slots] + extra_per_slot,
                ev.max_charge_power_kw,
            )
            # If still not enough (due to clipping), redistribute
            new_energy = float(ev_power[mask].sum() * dt)
            if new_energy < target_energy - 1e-3:
                # Fill all available slots to max
                ev_power[available_slots] = ev.max_charge_power_kw

        return ev_power

    def _repair_battery(self, raw_power: np.ndarray, dt: float) -> np.ndarray:
        """
        Repair battery power profile to satisfy SoC limits and power limits.
        Prevents simultaneous charge and discharge by sign.
        """
        if self.config.battery is None:
            return np.zeros(self.config.n_intervals)

        bat = self.config.battery
        T = self.config.n_intervals
        battery_power = np.clip(raw_power, -bat.max_discharge_power_kw, bat.max_charge_power_kw)

        # Simulate SoC and correct violations
        soc = bat.initial_soc
        repaired = np.zeros(T)

        for t in range(T):
            p = battery_power[t]
            if p > 0.0:  # charging
                # Energy added to battery
                energy_added = p * dt * bat.charge_efficiency
                new_soc = soc + energy_added / bat.capacity_kwh
                if new_soc > bat.max_soc:
                    # Reduce charging power
                    allowed_energy = (bat.max_soc - soc) * bat.capacity_kwh
                    p = allowed_energy / (dt * bat.charge_efficiency + 1e-12)
                    p = max(0.0, p)
                    new_soc = soc + p * dt * bat.charge_efficiency / bat.capacity_kwh
            elif p < 0.0:  # discharging
                # Energy removed from battery
                energy_removed = abs(p) * dt / bat.discharge_efficiency
                new_soc = soc - energy_removed / bat.capacity_kwh
                if new_soc < bat.min_soc:
                    # Reduce discharge power
                    allowed_energy = (soc - bat.min_soc) * bat.capacity_kwh
                    p = -(allowed_energy * bat.discharge_efficiency / (dt + 1e-12))
                    p = min(0.0, p)
                    energy_removed = abs(p) * dt / bat.discharge_efficiency
                    new_soc = soc - energy_removed / bat.capacity_kwh
            else:
                new_soc = soc

            repaired[t] = p
            soc = float(np.clip(new_soc, bat.min_soc, bat.max_soc))

        return repaired

    def _allocate_solar(
        self,
        ev_power: np.ndarray,
        battery_power: np.ndarray,
        solar_profile: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Allocate solar energy to demand. Solar is used first; excess is curtailed.

        Returns
        -------
        solar_used    : kW of solar actually consumed
        solar_curtailed : kW of solar curtailed (excess over demand)
        """
        T = self.config.n_intervals
        total_ev = ev_power.sum(axis=0)
        battery_charge = np.maximum(0.0, battery_power)

        # Total bus demand (excluding battery discharge, which contributes to supply)
        total_demand = total_ev + battery_charge

        solar_used = np.minimum(solar_profile, total_demand)
        solar_curtailed = solar_profile - solar_used

        return solar_used, np.maximum(0.0, solar_curtailed)

    def _compute_battery_soc(self, battery_power: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute battery SoC trajectory.

        Returns
        -------
        np.ndarray of shape (n_intervals + 1,) where index 0 is initial SoC.
        """
        if self.config.battery is None:
            return np.zeros(self.config.n_intervals + 1)

        bat = self.config.battery
        T = self.config.n_intervals
        soc = np.zeros(T + 1)
        soc[0] = bat.initial_soc

        for t in range(T):
            p = battery_power[t]
            if p > 0.0:
                delta_soc = p * dt * bat.charge_efficiency / bat.capacity_kwh
            elif p < 0.0:
                delta_soc = p * dt / (bat.discharge_efficiency * bat.capacity_kwh)
            else:
                delta_soc = 0.0
            soc[t + 1] = np.clip(soc[t] + delta_soc, bat.min_soc, bat.max_soc)

        return soc

    def _compute_ev_soc(self, ev_power: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute EV SoC trajectories.

        Returns
        -------
        np.ndarray of shape (n_evs, n_intervals + 1)
        """
        cfg = self.config
        T = cfg.n_intervals
        ev_soc = np.zeros((cfg.n_evs, T + 1))

        for i, ev in enumerate(cfg.evs):
            ev_soc[i, 0] = ev.initial_soc
            for t in range(T):
                delta_soc = ev_power[i, t] * dt / ev.battery_capacity_kwh
                ev_soc[i, t + 1] = np.clip(ev_soc[i, 0 if t == 0 else t] + delta_soc, 0.0, 1.0)

        # Recompute properly (cumulative)
        ev_soc = np.zeros((cfg.n_evs, T + 1))
        for i, ev in enumerate(cfg.evs):
            ev_soc[i, 0] = ev.initial_soc
            for t in range(T):
                delta_soc = ev_power[i, t] * dt / ev.battery_capacity_kwh
                ev_soc[i, t + 1] = np.clip(ev_soc[i, t] + delta_soc, 0.0, 1.0)

        return ev_soc

    def particle_to_schedule(self, particle: np.ndarray) -> EnergySchedule:
        """
        Convert a 1-D PSO particle vector to an EnergySchedule.

        Particle layout:
          [0 : n_evs*T]       — EV charging powers, row-major (ev_idx × time)
          [n_evs*T : n_evs*T+T] — Battery power (signed)
        """
        cfg = self.config
        T = cfg.n_intervals
        n_evs = cfg.n_evs

        ev_raw = particle[: n_evs * T].reshape(n_evs, T)
        bat_raw = particle[n_evs * T: n_evs * T + T]

        return self.decode(ev_raw, bat_raw)
