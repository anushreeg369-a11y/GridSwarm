"""
test_baseline.py — Tests for the rule-based baseline scheduler.
"""
import numpy as np
import pytest
from src.baseline_scheduler import BaselineScheduler
from src.constraints import ConstraintChecker


class TestBaselineScheduler:

    def test_baseline_runs_without_error(self, default_config):
        """Baseline must run to completion without exceptions."""
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        assert schedule is not None

    def test_baseline_grid_import_nonnegative(self, default_config):
        """Baseline schedule must never have negative grid import."""
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        assert np.all(schedule.grid_import_kw >= -1e-6), "Baseline has negative grid import"

    def test_baseline_ev_power_nonnegative(self, default_config):
        """Baseline EV charging powers must be non-negative."""
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        assert np.all(schedule.ev_power_kw >= -1e-6), "Baseline has negative EV power"

    def test_baseline_ev_power_within_limits(self, default_config):
        """Baseline EV charging powers must respect per-EV limits."""
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        for i, ev in enumerate(default_config.evs):
            assert np.all(schedule.ev_power_kw[i] <= ev.max_charge_power_kw + 1e-6), (
                f"Baseline EV {ev.ev_id} exceeded power limit"
            )

    def test_baseline_ev_not_charging_outside_window(self, default_config):
        """Baseline must not charge EVs outside their arrival-departure window."""
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        cfg = default_config
        hours = cfg.time_hours
        for i, ev in enumerate(cfg.evs):
            for t in range(cfg.n_intervals):
                h_start = hours[t]
                h_end = h_start + cfg.time_step_h
                is_avail = (h_end > ev.arrival_hour) and (h_start < ev.departure_hour)
                if not is_avail:
                    assert schedule.ev_power_kw[i, t] < 1e-6, (
                        f"Baseline: EV {ev.ev_id} charging at t={t} outside window"
                    )

    def test_baseline_battery_soc_limits(self, default_config):
        """Baseline battery SoC must stay within configured limits."""
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        bat = default_config.battery
        if bat is not None:
            assert np.all(schedule.battery_soc >= bat.min_soc - 1e-6)
            assert np.all(schedule.battery_soc <= bat.max_soc + 1e-6)

    def test_baseline_simple_config(self, simple_config):
        """Baseline works on a minimal single-EV config."""
        scheduler = BaselineScheduler(simple_config)
        schedule = scheduler.schedule()
        assert np.all(schedule.grid_import_kw >= -1e-6)
        assert np.all(schedule.ev_power_kw >= -1e-6)

    def test_baseline_solar_not_exceeding_available(self, default_config):
        """Baseline solar used must not exceed solar generated."""
        scheduler = BaselineScheduler(default_config)
        schedule = scheduler.schedule()
        solar = default_config.solar_profile_kw
        for t in range(default_config.n_intervals):
            assert schedule.solar_used_kw[t] <= solar[t] + 1e-6, (
                f"Baseline: solar used exceeds available at t={t}"
            )
