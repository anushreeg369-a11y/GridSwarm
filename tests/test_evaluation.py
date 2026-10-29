"""
test_evaluation.py — Tests for metrics, comparison, and CSV export.
"""
import numpy as np
import pytest
import pandas as pd
from src.baseline_scheduler import BaselineScheduler
from src.pso_optimizer import PSOOptimizer, PSOConfig
from src.evaluation import Evaluator, ComparisonResult
from src.objectives import ObjectiveFunction, ObjectiveWeights
from src.export_utils import schedule_to_dataframe, comparison_to_dataframe, config_to_dict


class TestObjectiveFunction:

    def test_objective_value_is_nonnegative(self, default_config):
        """Objective function value must be non-negative."""
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        obj = ObjectiveFunction(default_config)
        fitness = obj.evaluate(schedule)
        assert fitness >= 0.0

    def test_objective_components_dict(self, default_config):
        """component_values() must return all expected keys."""
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        obj = ObjectiveFunction(default_config)
        components = obj.component_values(schedule)
        required_keys = [
            "cost_eur", "peak_kw", "emissions_kg", "solar_utilization",
            "norm_cost", "norm_peak", "norm_emissions",
            "norm_solar_penalty", "composite_fitness",
        ]
        for key in required_keys:
            assert key in components, f"Missing key: {key}"

    def test_zero_weight_removes_objective(self, default_config):
        """Setting a weight to zero should not raise and just de-emphasise that objective."""
        weights = ObjectiveWeights(cost=1.0, peak=0.0, emissions=0.0, solar=0.0)
        obj = ObjectiveFunction(default_config, weights)
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        fitness = obj.evaluate(schedule)
        assert np.isfinite(fitness)

    def test_solar_utilization_in_range(self, default_config):
        """Solar utilization must be in [0, 1]."""
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        util = schedule.solar_utilization()
        assert 0.0 <= util <= 1.0, f"Solar utilization out of range: {util}"


class TestEvaluator:

    def test_evaluator_metrics_finite(self, default_config):
        """All metrics must be finite numbers."""
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        evaluator = Evaluator(default_config)
        metrics = evaluator.evaluate(schedule, label="Baseline")
        assert np.isfinite(metrics.total_cost_eur)
        assert np.isfinite(metrics.peak_demand_kw)
        assert np.isfinite(metrics.total_emissions_kg)
        assert np.isfinite(metrics.solar_utilization)

    def test_ev_satisfaction_in_range(self, default_config):
        """EV satisfaction fraction must be in [0, 1]."""
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        evaluator = Evaluator(default_config)
        metrics = evaluator.evaluate(schedule)
        assert 0.0 <= metrics.ev_target_satisfaction <= 1.0

    def test_comparison_safe_with_zero_baseline(self, simple_config):
        """Comparison must handle zero-valued baseline metrics safely."""
        import numpy as np
        from src.config import SystemConfig, EVConfig, GridConfig
        # Config where cost would be 0 (zero tariff)
        config = SystemConfig(
            time_horizon_h=4.0,
            time_step_h=1.0,
            evs=[EVConfig(0, 0.0, 4.0, 10.0, 0.50, 0.50, 1.0)],
            battery=None,
            grid=GridConfig(max_import_kw=10.0),
            solar_profile_kw=np.array([0.0, 0.0, 0.0, 0.0]),
            tariff_profile=np.array([0.0, 0.0, 0.0, 0.0]),
            emissions_profile=np.array([0.3, 0.3, 0.3, 0.3]),
        )
        baseline = BaselineScheduler(config).schedule()
        pso_sched = BaselineScheduler(config).schedule()  # Use baseline as PSO proxy
        evaluator = Evaluator(config)
        result = evaluator.compare(baseline, pso_sched)
        # Should not raise ZeroDivisionError
        assert isinstance(result.improvements, dict)

    def test_comparison_improvements_computable(self, default_config):
        """Improvement percentages must be computable without errors."""
        baseline_sched = BaselineScheduler(default_config).schedule()
        pso = PSOOptimizer(default_config, PSOConfig(n_particles=10, n_iterations=20, seed=5))
        pso_result = pso.run()
        evaluator = Evaluator(default_config)
        comparison = evaluator.compare(
            baseline_sched, pso_result.best_schedule,
            pso_runtime=pso_result.runtime_seconds,
            pso_fitness=pso_result.best_fitness,
            pso_convergence=pso_result.convergence_history,
        )
        assert all(np.isfinite(v) for v in comparison.improvements.values())


class TestCSVExport:

    def test_schedule_dataframe_columns(self, default_config):
        """Schedule DataFrame must contain required columns."""
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        df = schedule_to_dataframe(schedule)
        required = ["hour", "grid_import_kw", "solar_used_kw", "battery_power_kw", "total_ev_power_kw"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_schedule_dataframe_length(self, default_config):
        """Schedule DataFrame must have exactly n_intervals rows."""
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        df = schedule_to_dataframe(schedule)
        assert len(df) == default_config.n_intervals

    def test_comparison_dataframe(self, default_config):
        """Comparison DataFrame must have required columns and rows."""
        baseline = BaselineScheduler(default_config).schedule()
        pso_sched = BaselineScheduler(default_config).schedule()  # proxy
        evaluator = Evaluator(default_config)
        comparison = evaluator.compare(baseline, pso_sched)
        df = comparison_to_dataframe(comparison)
        assert "Metric" in df.columns
        assert "Baseline" in df.columns
        assert "PSO" in df.columns
        assert len(df) >= 5

    def test_config_to_dict_serializable(self, default_config):
        """config_to_dict must return JSON-serializable output."""
        import json
        d = config_to_dict(default_config)
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 10

    def test_schedule_csv_bytes(self, default_config):
        """CSV bytes output must be non-empty and decodeable."""
        from src.export_utils import df_to_csv_bytes
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        df = schedule_to_dataframe(schedule)
        csv_bytes = df_to_csv_bytes(df)
        assert len(csv_bytes) > 100
        # Should be valid CSV
        df2 = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
        assert len(df2) == default_config.n_intervals
