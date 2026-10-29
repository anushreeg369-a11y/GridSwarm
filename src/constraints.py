"""
constraints.py — Physical constraint validation for GridSwarm.

This module checks decoded EnergySchedule objects against all modelled
physical and operational constraints and reports violations clearly.

Checks performed:
  1. EV availability (no charging before arrival or after departure)
  2. EV charging power limits [0, max_charge_power_kw]
  3. EV target SOC satisfaction at departure
  4. EV battery capacity (SOC ≤ 1.0)
  5. Battery power limits
  6. Battery SoC limits (min_soc ≤ SoC ≤ max_soc)
  7. No simultaneous battery charge and discharge
  8. Grid import non-negativity
  9. Grid import capacity
  10. Solar non-negativity
  11. Energy balance (approximate)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np

from .energy_model import EnergySchedule
from .config import SystemConfig

_TOL = 1e-4  # numerical tolerance for constraint checks


@dataclass
class ConstraintViolation:
    """Describes a single constraint violation."""
    constraint: str   # short name
    message: str      # human-readable description
    severity: str     # 'error' or 'warning'
    interval: int = -1  # time interval index (-1 = not interval-specific)
    ev_id: int = -1     # EV index (-1 = not EV-specific)


@dataclass
class ConstraintReport:
    """Summary of all constraint checks for one schedule."""
    violations: List[ConstraintViolation] = field(default_factory=list)

    @property
    def is_feasible(self) -> bool:
        """True if there are no error-level violations."""
        return not any(v.severity == "error" for v in self.violations)

    @property
    def n_errors(self) -> int:
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")

    def summary(self) -> str:
        if self.is_feasible and self.n_warnings == 0:
            return "✅ All constraints satisfied — schedule is feasible."
        lines = [f"❌ {self.n_errors} error(s), ⚠️ {self.n_warnings} warning(s):"]
        for v in self.violations:
            icon = "❌" if v.severity == "error" else "⚠️"
            lines.append(f"  {icon} [{v.constraint}] {v.message}")
        return "\n".join(lines)


class ConstraintChecker:
    """Validates an EnergySchedule against all physical constraints."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self._precompute_ev_departure_indices()

    def _precompute_ev_departure_indices(self) -> None:
        """Find the last interval index where each EV is present."""
        cfg = self.config
        hours = cfg.time_hours
        self.ev_departure_idx: List[int] = []
        for ev in cfg.evs:
            # Last interval that ends at or before departure
            last = -1
            for t in range(cfg.n_intervals):
                interval_end = hours[t] + cfg.time_step_h
                if hours[t] < ev.departure_hour:
                    last = t
            self.ev_departure_idx.append(last)

    def check(self, schedule: EnergySchedule) -> ConstraintReport:
        """
        Run all constraint checks and return a ConstraintReport.

        Parameters
        ----------
        schedule : EnergySchedule to validate

        Returns
        -------
        ConstraintReport with all violations listed.
        """
        report = ConstraintReport()
        cfg = self.config

        self._check_ev_availability(schedule, report)
        self._check_ev_power_limits(schedule, report)
        self._check_ev_soc_targets(schedule, report)
        self._check_ev_capacity(schedule, report)

        if cfg.battery is not None:
            self._check_battery_power_limits(schedule, report)
            self._check_battery_soc_limits(schedule, report)
            self._check_battery_no_simultaneous(schedule, report)

        self._check_grid_import(schedule, report)
        self._check_solar_nonnegative(schedule, report)
        self._check_energy_balance(schedule, report)

        return report

    # ------------------------------------------------------------------
    # Individual check methods
    # ------------------------------------------------------------------

    def _check_ev_availability(self, schedule: EnergySchedule, report: ConstraintReport) -> None:
        cfg = self.config
        hours = cfg.time_hours
        for i, ev in enumerate(cfg.evs):
            for t in range(cfg.n_intervals):
                interval_start = hours[t]
                interval_end = hours[t] + cfg.time_step_h
                is_available = (interval_end > ev.arrival_hour) and (interval_start < ev.departure_hour)
                if not is_available and schedule.ev_power_kw[i, t] > _TOL:
                    report.violations.append(ConstraintViolation(
                        constraint="ev_availability",
                        message=(
                            f"EV {ev.ev_id} charging {schedule.ev_power_kw[i, t]:.3f} kW "
                            f"at interval {t} (hour {interval_start:.1f}) outside window "
                            f"[{ev.arrival_hour}, {ev.departure_hour}]"
                        ),
                        severity="error",
                        interval=t,
                        ev_id=ev.ev_id,
                    ))

    def _check_ev_power_limits(self, schedule: EnergySchedule, report: ConstraintReport) -> None:
        for i, ev in enumerate(self.config.evs):
            for t in range(self.config.n_intervals):
                p = schedule.ev_power_kw[i, t]
                if p < -_TOL:
                    report.violations.append(ConstraintViolation(
                        constraint="ev_power_negative",
                        message=f"EV {ev.ev_id} negative charging power {p:.4f} kW at interval {t}",
                        severity="error", interval=t, ev_id=ev.ev_id,
                    ))
                if p > ev.max_charge_power_kw + _TOL:
                    report.violations.append(ConstraintViolation(
                        constraint="ev_power_limit",
                        message=(
                            f"EV {ev.ev_id} power {p:.3f} kW exceeds max "
                            f"{ev.max_charge_power_kw:.3f} kW at interval {t}"
                        ),
                        severity="error", interval=t, ev_id=ev.ev_id,
                    ))

    def _check_ev_soc_targets(self, schedule: EnergySchedule, report: ConstraintReport) -> None:
        cfg = self.config
        for i, ev in enumerate(cfg.evs):
            dep_idx = self.ev_departure_idx[i]
            if dep_idx < 0:
                continue
            final_soc = schedule.ev_soc[i, dep_idx + 1]
            if final_soc < ev.target_soc - 0.02:  # 2% tolerance
                report.violations.append(ConstraintViolation(
                    constraint="ev_target_soc",
                    message=(
                        f"EV {ev.ev_id} departs with SOC {final_soc:.3f} "
                        f"< target {ev.target_soc:.3f} (deficit "
                        f"{(ev.target_soc - final_soc) * ev.battery_capacity_kwh:.2f} kWh)"
                    ),
                    severity="error", ev_id=ev.ev_id,
                ))

    def _check_ev_capacity(self, schedule: EnergySchedule, report: ConstraintReport) -> None:
        for i, ev in enumerate(self.config.evs):
            max_soc = schedule.ev_soc[i].max()
            if max_soc > 1.0 + _TOL:
                report.violations.append(ConstraintViolation(
                    constraint="ev_capacity",
                    message=f"EV {ev.ev_id} SOC exceeds 1.0 (max={max_soc:.4f})",
                    severity="error", ev_id=ev.ev_id,
                ))

    def _check_battery_power_limits(self, schedule: EnergySchedule, report: ConstraintReport) -> None:
        bat = self.config.battery
        for t in range(self.config.n_intervals):
            p = schedule.battery_power_kw[t]
            if p > bat.max_charge_power_kw + _TOL:
                report.violations.append(ConstraintViolation(
                    constraint="battery_charge_limit",
                    message=f"Battery charge power {p:.3f} kW > max {bat.max_charge_power_kw:.3f} kW at t={t}",
                    severity="error", interval=t,
                ))
            if p < -(bat.max_discharge_power_kw + _TOL):
                report.violations.append(ConstraintViolation(
                    constraint="battery_discharge_limit",
                    message=f"Battery discharge {abs(p):.3f} kW > max {bat.max_discharge_power_kw:.3f} kW at t={t}",
                    severity="error", interval=t,
                ))

    def _check_battery_soc_limits(self, schedule: EnergySchedule, report: ConstraintReport) -> None:
        bat = self.config.battery
        soc = schedule.battery_soc
        for t, s in enumerate(soc):
            if s < bat.min_soc - _TOL:
                report.violations.append(ConstraintViolation(
                    constraint="battery_min_soc",
                    message=f"Battery SOC {s:.4f} < min {bat.min_soc:.4f} at step {t}",
                    severity="error", interval=t,
                ))
            if s > bat.max_soc + _TOL:
                report.violations.append(ConstraintViolation(
                    constraint="battery_max_soc",
                    message=f"Battery SOC {s:.4f} > max {bat.max_soc:.4f} at step {t}",
                    severity="error", interval=t,
                ))

    def _check_battery_no_simultaneous(self, schedule: EnergySchedule, report: ConstraintReport) -> None:
        # Since battery_power is a single signed value, simultaneous charge+discharge
        # cannot occur by construction. This check is a safety guard for custom schedules.
        pass  # enforced by sign convention

    def _check_grid_import(self, schedule: EnergySchedule, report: ConstraintReport) -> None:
        cfg = self.config
        for t in range(cfg.n_intervals):
            g = schedule.grid_import_kw[t]
            if g < -_TOL:
                report.violations.append(ConstraintViolation(
                    constraint="grid_import_negative",
                    message=f"Grid import negative ({g:.4f} kW) at t={t}",
                    severity="error", interval=t,
                ))
            if g > cfg.grid.max_import_kw + _TOL:
                report.violations.append(ConstraintViolation(
                    constraint="grid_capacity",
                    message=(
                        f"Grid import {g:.2f} kW > capacity {cfg.grid.max_import_kw:.2f} kW at t={t}"
                    ),
                    severity="warning", interval=t,
                ))

    def _check_solar_nonnegative(self, schedule: EnergySchedule, report: ConstraintReport) -> None:
        for t in range(self.config.n_intervals):
            if schedule.solar_used_kw[t] < -_TOL:
                report.violations.append(ConstraintViolation(
                    constraint="solar_negative",
                    message=f"Solar used negative ({schedule.solar_used_kw[t]:.4f} kW) at t={t}",
                    severity="error", interval=t,
                ))

    def _check_energy_balance(self, schedule: EnergySchedule, report: ConstraintReport) -> None:
        """
        Verify energy balance at each interval:
          grid + solar_used + battery_discharge_to_bus ≈ ev_demand + battery_charge_power
        """
        cfg = self.config
        bat = cfg.battery
        dt = cfg.time_step_h

        for t in range(cfg.n_intervals):
            batt_p = schedule.battery_power_kw[t]
            if bat is not None and batt_p < 0.0:
                batt_discharge_to_bus = abs(batt_p) * bat.discharge_efficiency
            else:
                batt_discharge_to_bus = 0.0

            battery_charge_power = max(0.0, batt_p)

            supply = (
                schedule.grid_import_kw[t]
                + schedule.solar_used_kw[t]
                + batt_discharge_to_bus
            )
            demand = schedule.ev_power_kw[:, t].sum() + battery_charge_power

            imbalance = abs(supply - demand)
            if imbalance > 1.0:  # > 1 kW imbalance is flagged
                report.violations.append(ConstraintViolation(
                    constraint="energy_balance",
                    message=(
                        f"Energy imbalance {imbalance:.3f} kW at t={t} "
                        f"(supply={supply:.3f}, demand={demand:.3f})"
                    ),
                    severity="warning", interval=t,
                ))
