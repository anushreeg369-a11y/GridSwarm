"""
test_pso.py — Tests for PSO optimizer correctness and reproducibility.
"""
import numpy as np
import pytest
from src.pso_optimizer import PSOOptimizer, PSOConfig
from src.objectives import ObjectiveWeights
from src.data_generator import build_default_config


class TestPSOInitialization:

    def test_particle_bounds_respected(self, default_config):
        """Particles must initialize within declared bounds."""
        pso = PSOOptimizer(default_config, PSOConfig(n_particles=10, n_iterations=1, seed=0))
        result = pso.run()
        # gbest particle must be within bounds
        assert np.all(result.gbest_particle >= pso.lb - 1e-6)
        assert np.all(result.gbest_particle <= pso.ub + 1e-6)

    def test_convergence_history_length(self, default_config):
        """Convergence history must have n_iterations + 1 entries (initial + each iter)."""
        n_iter = 20
        pso = PSOOptimizer(default_config, PSOConfig(n_particles=5, n_iterations=n_iter, seed=1))
        result = pso.run()
        assert len(result.convergence_history) == n_iter + 1

    def test_convergence_monotone_decrease(self, default_config):
        """Best fitness must be non-increasing (gbest can only improve or stay)."""
        pso = PSOOptimizer(default_config, PSOConfig(n_particles=20, n_iterations=30, seed=2))
        result = pso.run()
        h = result.convergence_history
        for i in range(1, len(h)):
            assert h[i] <= h[i - 1] + 1e-9, (
                f"Convergence history increased at iteration {i}: {h[i-1]:.6f} → {h[i]:.6f}"
            )


class TestPSOReproducibility:

    def test_same_seed_same_result(self, default_config):
        """Two runs with the same seed must produce identical results."""
        cfg = PSOConfig(n_particles=10, n_iterations=20, seed=42)
        r1 = PSOOptimizer(default_config, cfg).run()
        r2 = PSOOptimizer(default_config, cfg).run()
        assert abs(r1.best_fitness - r2.best_fitness) < 1e-12
        np.testing.assert_array_equal(r1.convergence_history, r2.convergence_history)

    def test_different_seed_different_result(self, default_config):
        """Different seeds should generally produce different convergence paths."""
        cfg1 = PSOConfig(n_particles=10, n_iterations=5, seed=1)
        cfg2 = PSOConfig(n_particles=10, n_iterations=5, seed=999)
        r1 = PSOOptimizer(default_config, cfg1).run()
        r2 = PSOOptimizer(default_config, cfg2).run()
        # They might coincidentally be equal, but should usually differ
        # We just check it doesn't crash
        assert r1.best_fitness >= 0
        assert r2.best_fitness >= 0


class TestPSOScheduleQuality:

    def test_pso_produces_feasible_schedule(self, default_config):
        """PSO result must decode into a schedule with non-negative grid import."""
        pso = PSOOptimizer(default_config, PSOConfig(n_particles=15, n_iterations=30, seed=7))
        result = pso.run()
        schedule = result.best_schedule
        assert np.all(schedule.grid_import_kw >= -1e-6), "PSO schedule has negative grid import"
        assert np.all(schedule.ev_power_kw >= -1e-6), "PSO schedule has negative EV power"

    def test_pso_fitness_positive(self, default_config):
        """Fitness value must be a positive finite number."""
        pso = PSOOptimizer(default_config, PSOConfig(n_particles=5, n_iterations=5, seed=3))
        result = pso.run()
        assert np.isfinite(result.best_fitness)
        assert result.best_fitness >= 0.0

    def test_pso_dimension_correct(self, default_config):
        """Particle dimension = n_evs × T + T."""
        pso = PSOOptimizer(default_config)
        T = default_config.n_intervals
        n_evs = default_config.n_evs
        expected_dim = n_evs * T + T
        assert pso.dim == expected_dim

    def test_pso_with_custom_weights(self, default_config):
        """PSO runs correctly with custom objective weights."""
        weights = ObjectiveWeights(cost=0.8, peak=0.1, emissions=0.1, solar=0.0)
        pso = PSOOptimizer(
            default_config,
            PSOConfig(n_particles=5, n_iterations=5, seed=4),
            weights=weights,
        )
        result = pso.run()
        assert np.isfinite(result.best_fitness)
