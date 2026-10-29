"""
baseline_scheduler.py — Rule-based baseline scheduler for GridSwarm.

Strategy: Earliest-Departure-First (EDF) with Solar Priority

Rules (applied in order at each interval):
  1. Prioritize EVs with earliest departure.
  2. Charge EVs at maximum power whenever grid allows, during availability.
  3. Use available solar to offset EV demand before drawing from grid.
  4. If solar exceeds EV demand, store excess in battery (if not full).
  5. Discharge battery during peak tariff periods (if battery has energy).
  6. Grid import makes up the remaining demand.

This creates a transparent, deterministic schedule that serves as the
performance comparison baseline for PSO.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from .config import SystemConfig
from .energy_model import EnergyModel, EnergySchedule


class BaselineScheduler:
    """
    Transparent rule-based energy scheduler (Earliest-Departure-First + Solar Priority).

    This produces a deterministic schedule that does NOT require optimization.
    It serves as the comparison baseline for PSO.
    """

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.energy_model = EnergyModel(config)

    def schedule(self) -> EnergySchedule:
        """
        Build and return the baseline schedule.

        Returns
        -------
        EnergySchedule decoded and validated through EnergyModel.
        """
        cfg = self.config
        T = cfg.n_intervals
        n_evs = cfg.n_evs
        dt = cfg.time_step_h
        hours = cfg.time_hours

        solar = cfg.solar_profile_kw if cfg.solar_profile_kw is not None else np.zeros(T)
        tariff = cfg.tariff_profile if cfg.tariff_profile is not None else np.ones(T) * 0.20
        bat = cfg.battery

        # Sort EVs by departure hour (EDF order)
        ev_order = sorted(range(n_evs), key=lambda i: cfg.evs[i].departure_hour)

        # Track remaining energy needed per EV
        ev_energy_remaining = np.array([ev.energy_needed_kwh for ev in cfg.evs], dtype=float)

        # Initialize arrays
        ev_power = np.zeros((n_evs, T))
        battery_power = np.zeros(T)

        bat_soc = bat.initial_soc if bat else 0.0

        # Identify peak tariff hours for battery discharge
        if bat is not None and cfg.tariff_profile is not None:
            tariff_threshold = np.percentile(tariff, 60)  # discharge during top-40% price hours
            is_peak_hour = tariff >= tariff_threshold
        else:
            is_peak_hour = np.zeros(T, dtype=bool)

        for t in range(T):
            h_start = hours[t]
            h_end = h_start + dt
            solar_available = solar[t]
            grid_capacity_remaining = cfg.grid.max_import_kw

            # --- Step 1: Charge EVs (EDF order) ---
            for ev_idx in ev_order:
                ev = cfg.evs[ev_idx]
                # Check availability
                if h_end <= ev.arrival_hour or h_start >= ev.departure_hour:
                    continue
                if ev_energy_remaining[ev_idx] <= 1e-6:
                    continue

                # Determine charging power
                max_p = ev.max_charge_power_kw
                needed_p = ev_energy_remaining[ev_idx] / dt
                charge_p = min(max_p, needed_p)

                # Use solar first
                solar_used_for_ev = min(solar_available, charge_p)
                grid_used_for_ev = charge_p - solar_used_for_ev

                # Check grid capacity
                if grid_used_for_ev > grid_capacity_remaining:
                    grid_used_for_ev = grid_capacity_remaining
                    charge_p = solar_used_for_ev + grid_used_for_ev
                    solar_used_for_ev = min(solar_available, charge_p)
                    grid_used_for_ev = charge_p - solar_used_for_ev

                ev_power[ev_idx, t] = charge_p
                solar_available -= solar_used_for_ev
                solar_available = max(0.0, solar_available)
                grid_capacity_remaining -= grid_used_for_ev
                grid_capacity_remaining = max(0.0, grid_capacity_remaining)
                ev_energy_remaining[ev_idx] -= charge_p * dt
                ev_energy_remaining[ev_idx] = max(0.0, ev_energy_remaining[ev_idx])

            # --- Step 2: Store excess solar in battery ---
            if bat is not None and solar_available > 0.0:
                soc_headroom = (bat.max_soc - bat_soc) * bat.capacity_kwh
                max_charge_energy = bat.max_charge_power_kw * dt * bat.charge_efficiency
                charge_energy = min(solar_available * dt, soc_headroom, max_charge_energy)
                charge_p = charge_energy / (dt + 1e-12)
                if charge_p > 0:
                    battery_power[t] += charge_p
                    bat_soc += charge_energy / bat.capacity_kwh
                    bat_soc = min(bat_soc, bat.max_soc)
                    solar_available -= charge_p
                    solar_available = max(0.0, solar_available)

            # --- Step 3: Discharge battery during peak hours ---
            if bat is not None and is_peak_hour[t] and battery_power[t] == 0.0:
                soc_available = (bat_soc - bat.min_soc) * bat.capacity_kwh
                max_discharge_energy = bat.max_discharge_power_kw * dt / bat.discharge_efficiency
                discharge_energy = min(soc_available, max_discharge_energy)
                discharge_p = discharge_energy / (dt + 1e-12)
                if discharge_p > 0:
                    battery_power[t] -= discharge_p  # negative = discharging
                    bat_soc -= discharge_energy / bat.capacity_kwh
                    bat_soc = max(bat_soc, bat.min_soc)

            # --- Update battery SoC for next interval ---
            if bat is not None:
                p = battery_power[t]
                if p > 0:
                    bat_soc += p * dt * bat.charge_efficiency / bat.capacity_kwh
                elif p < 0:
                    bat_soc -= abs(p) * dt / (bat.discharge_efficiency * bat.capacity_kwh)
                bat_soc = float(np.clip(bat_soc, bat.min_soc, bat.max_soc))

        # Decode through EnergyModel to get consistent EnergySchedule
        return self.energy_model.decode(ev_power, battery_power)
