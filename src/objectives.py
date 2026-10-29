"""
objectives.py — Multi-objective weighted-sum fitness function for GridSwarm.

Formulation
-----------
This is a WEIGHTED-SUM multi-objective optimizer (not a Pareto-front algorithm).

The four objectives are independently normalized to [0, 1] using reference
bounds derived from the problem (naive upper/lower bounds), then combined:

    J = w_cost     × norm_cost
      + w_peak     × norm_peak_demand
      + w_emissions × norm_emissions
      + w_solar    × (1 - solar_utilization)

Lower J is better.

Normalization bounds
--------------------
- Cost:       [0, max_possible_cost]  where max = max_tariff × grid_cap × T × dt
- Peak:       [0, grid_cap]
- Emissions:  [0, max_possible_emissions]
- Solar util: already in [0, 1]

All weights must be nonnegative. They do NOT need to sum to 1 (they are
applied as-is, not automatically normalized), but a note is shown when
they don't sum to 1.0.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .energy_model import EnergySchedule
from .config import SystemConfig


@dataclass
class ObjectiveWeights:
    """Configurable weights for the multi-objective fitness function."""
    cost: float = 0.40       # Minimize electricity cost
    peak: float = 0.30       # Minimize peak grid demand
    emissions: float = 0.20  # Minimize carbon emissions
    solar: float = 0.10      # Maximize solar utilization (1 - util)

    def __post_init__(self) -> None:
        assert self.cost >= 0, "cost weight must be >= 0"
        assert self.peak >= 0, "peak weight must be >= 0"
        assert self.emissions >= 0, "emissions weight must be >= 0"
        assert self.solar >= 0, "solar weight must be >= 0"

    @property
    def total(self) -> float:
        return self.cost + self.peak + self.emissions + self.solar

    def as_array(self) -> np.ndarray:
        return np.array([self.cost, self.peak, self.emissions, self.solar])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ObjectiveWeights":
        return cls(cost=float(arr[0]), peak=float(arr[1]),
                   emissions=float(arr[2]), solar=float(arr[3]))


class ObjectiveFunction:
    """
    Computes the multi-objective weighted-sum fitness value for a schedule.

    The normalization bounds are computed once from the SystemConfig.
    """

    def __init__(self, config: SystemConfig, weights: ObjectiveWeights | None = None) -> None:
        self.config = config
        self.weights = weights or ObjectiveWeights()
        self._compute_normalization_bounds()

    def _compute_normalization_bounds(self) -> None:
        """Pre-compute upper bounds for normalization."""
        cfg = self.config
        T = cfg.n_intervals
        dt = cfg.time_step_h

        # Worst-case cost: full grid capacity at peak tariff for all hours
        max_tariff = float(np.max(cfg.tariff_profile)) if cfg.tariff_profile is not None else 0.35
        self._max_cost = max_tariff * cfg.grid.max_import_kw * T * dt + 1e-6

        # Peak demand upper bound = grid capacity
        self._max_peak = cfg.grid.max_import_kw + 1e-6

        # Worst-case emissions: full grid at max emissions intensity
        max_emissions = float(np.max(cfg.emissions_profile)) if cfg.emissions_profile is not None else 0.55
        self._max_emissions = max_emissions * cfg.grid.max_import_kw * T * dt + 1e-6

    def evaluate(self, schedule: EnergySchedule) -> float:
        """
        Compute the composite weighted-sum objective value for one schedule.

        Lower is better.

        Returns
        -------
        float : composite fitness score
        """
        w = self.weights

        cost = schedule.total_cost()
        peak = schedule.peak_demand()
        emissions = schedule.total_emissions()
        sol_util = schedule.solar_utilization()

        norm_cost = cost / self._max_cost
        norm_peak = peak / self._max_peak
        norm_emissions = emissions / self._max_emissions
        norm_solar_penalty = 1.0 - sol_util  # maximizing util = minimizing penalty

        fitness = (
            w.cost * norm_cost
            + w.peak * norm_peak
            + w.emissions * norm_emissions
            + w.solar * norm_solar_penalty
        )
        return float(fitness)

    def component_values(self, schedule: EnergySchedule) -> dict:
        """
        Return individual objective values (un-normalized) and normalized scores.

        Returns a dictionary with keys:
          cost_eur, peak_kw, emissions_kg, solar_utilization,
          norm_cost, norm_peak, norm_emissions, norm_solar_penalty,
          composite_fitness
        """
        cost = schedule.total_cost()
        peak = schedule.peak_demand()
        emissions = schedule.total_emissions()
        sol_util = schedule.solar_utilization()

        norm_cost = cost / self._max_cost
        norm_peak = peak / self._max_peak
        norm_emissions = emissions / self._max_emissions
        norm_solar_penalty = 1.0 - sol_util

        return {
            "cost_eur": cost,
            "peak_kw": peak,
            "emissions_kg": emissions,
            "solar_utilization": sol_util,
            "norm_cost": norm_cost,
            "norm_peak": norm_peak,
            "norm_emissions": norm_emissions,
            "norm_solar_penalty": norm_solar_penalty,
            "composite_fitness": (
                self.weights.cost * norm_cost
                + self.weights.peak * norm_peak
                + self.weights.emissions * norm_emissions
                + self.weights.solar * norm_solar_penalty
            ),
        }
