"""
evaluation.py — Metrics computation and comparison for GridSwarm.

Computes:
- Individual schedule metrics (cost, peak, emissions, solar utilization, EV satisfaction)
- Baseline vs PSO comparison with safe percentage improvement calculations
- Human-readable results summary
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

from .energy_model import EnergySchedule
from .constraints import ConstraintChecker, ConstraintReport
from .config import SystemConfig


@dataclass
class ScheduleMetrics:
    """All computed metrics for one schedule."""
    label: str                          # "Baseline" or "PSO"
    total_cost_eur: float               # Total electricity cost (€)
    peak_demand_kw: float               # Peak grid import (kW)
    total_emissions_kg: float           # Total CO₂ emissions (kg)
    solar_utilization: float            # Solar self-consumption fraction [0,1]
    solar_energy_used_kwh: float        # Solar energy actually consumed (kWh)
    solar_energy_total_kwh: float       # Total available solar energy (kWh)
    total_grid_energy_kwh: float        # Total grid energy imported (kWh)
    ev_target_satisfaction: float       # Fraction of EVs meeting departure target [0,1]
    ev_target_deficits_kwh: list        # Energy deficit per EV (kWh), 0 if met
    constraint_report: ConstraintReport
    runtime_seconds: Optional[float] = None
    fitness_value: Optional[float] = None
    convergence_history: Optional[list] = None

    @property
    def is_feasible(self) -> bool:
        return self.constraint_report.is_feasible

    @property
    def solar_utilization_pct(self) -> float:
        return self.solar_utilization * 100.0

    @property
    def ev_satisfaction_pct(self) -> float:
        return self.ev_target_satisfaction * 100.0


@dataclass
class ComparisonResult:
    """Side-by-side comparison of baseline and PSO metrics."""
    baseline: ScheduleMetrics
    pso: ScheduleMetrics
    improvements: Dict[str, float] = field(default_factory=dict)
    summary_text: str = ""

    def compute_improvements(self) -> None:
        """
        Compute percentage improvements of PSO over baseline.
        Handles zero-valued baselines safely.
        """
        b = self.baseline
        p = self.pso

        def safe_pct_improvement(baseline_val: float, pso_val: float) -> float:
            """Positive = PSO improved over baseline (lower is better)."""
            if abs(baseline_val) < 1e-9:
                return 0.0
            return 100.0 * (baseline_val - pso_val) / abs(baseline_val)

        def safe_pct_improvement_higher_better(baseline_val: float, pso_val: float) -> float:
            """Positive = PSO improved over baseline (higher is better)."""
            if abs(baseline_val) < 1e-9:
                return 0.0
            return 100.0 * (pso_val - baseline_val) / abs(baseline_val)

        self.improvements = {
            "cost_improvement_pct": safe_pct_improvement(b.total_cost_eur, p.total_cost_eur),
            "peak_reduction_pct": safe_pct_improvement(b.peak_demand_kw, p.peak_demand_kw),
            "emissions_reduction_pct": safe_pct_improvement(b.total_emissions_kg, p.total_emissions_kg),
            "solar_improvement_pct": safe_pct_improvement_higher_better(b.solar_utilization, p.solar_utilization),
        }

    def generate_summary(self) -> str:
        """Generate a human-readable comparison summary based on actual metrics."""
        self.compute_improvements()
        b = self.baseline
        p = self.pso
        imp = self.improvements

        lines = [
            "=" * 60,
            "  GridSwarm — Baseline vs PSO Optimization Results",
            "=" * 60,
            "",
            f"  {'Metric':<30} {'Baseline':>12} {'PSO':>12} {'Δ':>10}",
            f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}",
            f"  {'Total Cost (€)':<30} {b.total_cost_eur:>12.2f} {p.total_cost_eur:>12.2f} "
            f"  {imp['cost_improvement_pct']:>+8.1f}%",
            f"  {'Peak Demand (kW)':<30} {b.peak_demand_kw:>12.2f} {p.peak_demand_kw:>12.2f} "
            f"  {imp['peak_reduction_pct']:>+8.1f}%",
            f"  {'CO₂ Emissions (kg)':<30} {b.total_emissions_kg:>12.2f} {p.total_emissions_kg:>12.2f} "
            f"  {imp['emissions_reduction_pct']:>+8.1f}%",
            f"  {'Solar Utilization (%)':<30} {b.solar_utilization_pct:>12.1f} {p.solar_utilization_pct:>12.1f} "
            f"  {imp['solar_improvement_pct']:>+8.1f}%",
            f"  {'EV Target Satisfaction (%)':<30} {b.ev_satisfaction_pct:>12.1f} {p.ev_satisfaction_pct:>12.1f} ",
            f"  {'Grid Energy (kWh)':<30} {b.total_grid_energy_kwh:>12.2f} {p.total_grid_energy_kwh:>12.2f} ",
            "",
            f"  Baseline feasible:  {'Yes ✅' if b.is_feasible else 'No ❌'}",
            f"  PSO feasible:       {'Yes ✅' if p.is_feasible else 'No ❌'}",
        ]

        if p.runtime_seconds is not None:
            lines.append(f"  PSO runtime:        {p.runtime_seconds:.2f} s")

        lines.append("=" * 60)
        self.summary_text = "\n".join(lines)
        return self.summary_text


class Evaluator:
    """Computes metrics for a given EnergySchedule."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.checker = ConstraintChecker(config)

    def evaluate(
        self,
        schedule: EnergySchedule,
        label: str = "Schedule",
        runtime: Optional[float] = None,
        fitness: Optional[float] = None,
        convergence_history: Optional[list] = None,
    ) -> ScheduleMetrics:
        """Evaluate a schedule and return all metrics."""
        cfg = self.config
        dt = cfg.time_step_h
        bat = cfg.battery

        total_cost = schedule.total_cost()
        peak = schedule.peak_demand()
        emissions = schedule.total_emissions()
        solar_util = schedule.solar_utilization()

        total_solar_kwh = float(np.sum(cfg.solar_profile_kw) * dt) if cfg.solar_profile_kw is not None else 0.0
        solar_used_kwh = float(np.sum(schedule.solar_used_kw) * dt)
        grid_energy_kwh = float(np.sum(schedule.grid_import_kw) * dt)

        # EV departure satisfaction
        ev_deficits = []
        satisfied = 0
        checker = self.checker
        for i, ev in enumerate(cfg.evs):
            dep_idx = checker.ev_departure_idx[i]
            if dep_idx < 0:
                ev_deficits.append(0.0)
                satisfied += 1
                continue
            final_soc = schedule.ev_soc[i, dep_idx + 1]
            deficit_kwh = max(0.0, (ev.target_soc - final_soc) * ev.battery_capacity_kwh)
            ev_deficits.append(deficit_kwh)
            if deficit_kwh < 0.5:  # within 0.5 kWh tolerance
                satisfied += 1

        ev_satisfaction = satisfied / max(len(cfg.evs), 1)

        constraint_report = self.checker.check(schedule)

        return ScheduleMetrics(
            label=label,
            total_cost_eur=total_cost,
            peak_demand_kw=peak,
            total_emissions_kg=emissions,
            solar_utilization=solar_util,
            solar_energy_used_kwh=solar_used_kwh,
            solar_energy_total_kwh=total_solar_kwh,
            total_grid_energy_kwh=grid_energy_kwh,
            ev_target_satisfaction=ev_satisfaction,
            ev_target_deficits_kwh=ev_deficits,
            constraint_report=constraint_report,
            runtime_seconds=runtime,
            fitness_value=fitness,
            convergence_history=convergence_history,
        )

    def compare(
        self,
        baseline_schedule: EnergySchedule,
        pso_schedule: EnergySchedule,
        pso_runtime: Optional[float] = None,
        pso_fitness: Optional[float] = None,
        pso_convergence: Optional[list] = None,
    ) -> ComparisonResult:
        """Compare baseline and PSO schedules and return a ComparisonResult."""
        baseline_metrics = self.evaluate(baseline_schedule, label="Baseline")
        pso_metrics = self.evaluate(
            pso_schedule,
            label="PSO",
            runtime=pso_runtime,
            fitness=pso_fitness,
            convergence_history=pso_convergence,
        )
        result = ComparisonResult(baseline=baseline_metrics, pso=pso_metrics)
        result.compute_improvements()
        result.generate_summary()
        return result
