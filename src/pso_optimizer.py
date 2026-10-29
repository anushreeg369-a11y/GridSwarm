"""
pso_optimizer.py — Particle Swarm Optimization implementation for GridSwarm.

Biological Inspiration
----------------------
PSO is inspired by collective animal behaviour: birds flocking, fish schooling,
insect swarming. Each individual (particle) represents a candidate solution
(energy schedule). Particles "fly" through the search space, attracted by
the best position they personally found (pbest) and the best found by the
entire swarm (gbest).

Mathematical Formulation
------------------------
Position:  x(i,t+1) = x(i,t) + v(i,t+1)

Velocity:
  v(i,t+1) = w(t) · v(i,t)
            + c1 · r1 · (pbest(i) - x(i,t))   ← cognitive component
            + c2 · r2 · (gbest    - x(i,t))   ← social component

Inertia weight uses linear decay:
  w(t) = w_max - (w_max - w_min) × t / T_max

Particle Encoding
-----------------
Particle vector (length = n_evs × T + T):
  [ev_0_t0, ev_0_t1, ..., ev_0_tT-1,
   ev_1_t0, ..., ev_n-1_tT-1,
   bat_t0, bat_t1, ..., bat_tT-1]

EV elements ∈ [0, max_charge_power_kw]
Battery elements ∈ [-max_discharge_power, max_charge_power]

After position update, particles are decoded into EnergySchedules using
EnergyModel.particle_to_schedule(), which applies repair strategies to
ensure physical feasibility before fitness evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Callable
import time
import numpy as np

from .config import SystemConfig
from .energy_model import EnergyModel, EnergySchedule
from .objectives import ObjectiveFunction, ObjectiveWeights


@dataclass
class PSOConfig:
    """Hyperparameters for the PSO algorithm."""
    n_particles: int = 40        # Swarm size
    n_iterations: int = 100      # Maximum number of iterations
    w_max: float = 0.9           # Maximum inertia weight
    w_min: float = 0.4           # Minimum inertia weight (linear decay)
    c1: float = 2.05             # Cognitive acceleration coefficient
    c2: float = 2.05             # Social acceleration coefficient
    seed: Optional[int] = 42     # Random seed for reproducibility
    verbose: bool = False        # Print iteration progress


@dataclass
class PSOResult:
    """Result of one PSO optimization run."""
    best_schedule: EnergySchedule          # Best decoded schedule found
    best_fitness: float                    # Best (lowest) fitness value
    convergence_history: List[float]       # Best fitness at each iteration
    gbest_particle: np.ndarray             # Raw gbest particle vector
    runtime_seconds: float                 # Wall-clock time for optimization
    n_iterations_run: int                  # Actual iterations executed
    pso_config: PSOConfig
    objective_weights: ObjectiveWeights


class PSOOptimizer:
    """
    Particle Swarm Optimizer for smart energy management scheduling.

    Usage
    -----
    optimizer = PSOOptimizer(system_config, pso_config, weights)
    result = optimizer.run()
    """

    def __init__(
        self,
        system_config: SystemConfig,
        pso_config: PSOConfig | None = None,
        weights: ObjectiveWeights | None = None,
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> None:
        self.system_config = system_config
        self.pso_config = pso_config or PSOConfig()
        self.weights = weights or ObjectiveWeights()
        self.progress_callback = progress_callback

        self.energy_model = EnergyModel(system_config)
        self.objective_fn = ObjectiveFunction(system_config, self.weights)

        self._setup_bounds()

    def _setup_bounds(self) -> None:
        """Compute lower and upper bounds for the particle vector."""
        cfg = self.system_config
        T = cfg.n_intervals
        n_evs = cfg.n_evs
        bat = cfg.battery

        lb_ev = np.zeros(n_evs * T)
        ub_ev = np.zeros(n_evs * T)
        for i, ev in enumerate(cfg.evs):
            start = i * T
            end = (i + 1) * T
            ub_ev[start:end] = ev.max_charge_power_kw

        if bat is not None:
            lb_bat = np.full(T, -bat.max_discharge_power_kw)
            ub_bat = np.full(T, bat.max_charge_power_kw)
        else:
            lb_bat = np.zeros(T)
            ub_bat = np.zeros(T)

        self.lb = np.concatenate([lb_ev, lb_bat])
        self.ub = np.concatenate([ub_ev, ub_bat])
        self.dim = len(self.lb)

    def _initialize_swarm(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize particle positions and velocities.

        Positions: uniform random within [lb, ub]
        Velocities: uniform random in [-0.5*(ub-lb), 0.5*(ub-lb)]
        """
        n = self.pso_config.n_particles
        d = self.dim
        span = self.ub - self.lb

        positions = self.lb + rng.random((n, d)) * span
        velocities = -0.5 * span + rng.random((n, d)) * span

        return positions, velocities

    def _evaluate_particle(self, position: np.ndarray) -> tuple[float, EnergySchedule]:
        """Decode particle and compute fitness."""
        schedule = self.energy_model.particle_to_schedule(position)
        fitness = self.objective_fn.evaluate(schedule)
        return fitness, schedule

    def run(self) -> PSOResult:
        """
        Execute the PSO optimization.

        Returns
        -------
        PSOResult containing the best schedule, fitness, and convergence history.
        """
        cfg = self.pso_config
        rng = np.random.default_rng(cfg.seed)

        t_start = time.perf_counter()

        # Initialize swarm
        positions, velocities = self._initialize_swarm(rng)
        n = cfg.n_particles

        # Evaluate initial population
        pbest_positions = positions.copy()
        pbest_fitness = np.full(n, np.inf)
        gbest_position = positions[0].copy()
        gbest_fitness = np.inf
        gbest_schedule: Optional[EnergySchedule] = None

        for i in range(n):
            fit, schedule = self._evaluate_particle(positions[i])
            pbest_fitness[i] = fit
            if fit < gbest_fitness:
                gbest_fitness = fit
                gbest_position = positions[i].copy()
                gbest_schedule = schedule

        convergence_history: List[float] = [gbest_fitness]

        # Main PSO loop
        for iteration in range(cfg.n_iterations):
            # Linear inertia decay
            w = cfg.w_max - (cfg.w_max - cfg.w_min) * iteration / max(cfg.n_iterations - 1, 1)

            # Generate random coefficients
            r1 = rng.random((n, self.dim))
            r2 = rng.random((n, self.dim))

            # Update velocities
            velocities = (
                w * velocities
                + cfg.c1 * r1 * (pbest_positions - positions)
                + cfg.c2 * r2 * (gbest_position - positions)
            )

            # Clamp velocities (prevent explosion)
            v_max = 0.5 * (self.ub - self.lb)
            velocities = np.clip(velocities, -v_max, v_max)

            # Update positions and clamp to bounds
            positions = positions + velocities
            positions = np.clip(positions, self.lb, self.ub)

            # Evaluate new positions
            for i in range(n):
                fit, schedule = self._evaluate_particle(positions[i])
                if fit < pbest_fitness[i]:
                    pbest_fitness[i] = fit
                    pbest_positions[i] = positions[i].copy()
                    if fit < gbest_fitness:
                        gbest_fitness = fit
                        gbest_position = positions[i].copy()
                        gbest_schedule = schedule

            convergence_history.append(gbest_fitness)

            if cfg.verbose:
                print(f"  Iter {iteration+1:4d}/{cfg.n_iterations}  gbest={gbest_fitness:.6f}")

            if self.progress_callback is not None:
                self.progress_callback(iteration + 1, gbest_fitness)

        runtime = time.perf_counter() - t_start

        # If gbest_schedule is still None (shouldn't happen), decode gbest particle
        if gbest_schedule is None:
            _, gbest_schedule = self._evaluate_particle(gbest_position)

        return PSOResult(
            best_schedule=gbest_schedule,
            best_fitness=gbest_fitness,
            convergence_history=convergence_history,
            gbest_particle=gbest_position,
            runtime_seconds=runtime,
            n_iterations_run=cfg.n_iterations,
            pso_config=cfg,
            objective_weights=self.weights,
        )
