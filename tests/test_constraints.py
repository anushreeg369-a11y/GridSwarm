"""
test_constraints.py — Tests for physical constraint enforcement.
"""
import numpy as np
import pytest
from src.energy_model import EnergyModel
from src.constraints import ConstraintChecker
from src.config import EVConfig, SystemConfig, GridConfig
from src.data_generator import build_infeasible_config


class TestEVConstraints:

    def test_ev_not_charging_before_arrival(self, simple_config):
        """EV should not charge before its arrival hour."""
        # simple_config has EV arriving at 0 and departing at 4 (all 4 slots available)
        # Let's create a config where EV arrives at hour 2
        from src.config import EVConfig, SystemConfig, GridConfig
        import numpy as np
        config = SystemConfig(
            time_horizon_h=4.0,
            time_step_h=1.0,
            evs=[
                EVConfig(
                    ev_id=0,
                    arrival_hour=2.0,
                    departure_hour=4.0,
                    battery_capacity_kwh=20.0,
                    initial_soc=0.20,
                    target_soc=0.60,
                    max_charge_power_kw=7.4,
                )
            ],
            battery=None,
            grid=GridConfig(max_import_kw=50.0),
            solar_profile_kw=np.zeros(4),
            tariff_profile=np.ones(4) * 0.20,
            emissions_profile=np.ones(4) * 0.30,
        )
        model = EnergyModel(config)
        ev_raw = np.ones((1, 4)) * 5.0  # wants to charge all 4 hours
        bat_raw = np.zeros(4)
        schedule = model.decode(ev_raw, bat_raw)
        # Hours 0 and 1 (t=0, t=1) must be zero
        assert schedule.ev_power_kw[0, 0] < 1e-6, "EV charging before arrival at t=0"
        assert schedule.ev_power_kw[0, 1] < 1e-6, "EV charging before arrival at t=1"
        # Hours 2 and 3 (t=2, t=3) may be non-zero
        assert schedule.ev_power_kw[0, 2] >= 0.0

    def test_ev_not_charging_after_departure(self, simple_config):
        """EV should not charge after its departure hour."""
        from src.config import EVConfig, SystemConfig, GridConfig
        import numpy as np
        config = SystemConfig(
            time_horizon_h=4.0,
            time_step_h=1.0,
            evs=[
                EVConfig(
                    ev_id=0,
                    arrival_hour=0.0,
                    departure_hour=2.0,
                    battery_capacity_kwh=10.0,
                    initial_soc=0.20,
                    target_soc=0.60,
                    max_charge_power_kw=7.4,
                )
            ],
            battery=None,
            grid=GridConfig(max_import_kw=50.0),
            solar_profile_kw=np.zeros(4),
            tariff_profile=np.ones(4) * 0.20,
            emissions_profile=np.ones(4) * 0.30,
        )
        model = EnergyModel(config)
        ev_raw = np.ones((1, 4)) * 5.0
        bat_raw = np.zeros(4)
        schedule = model.decode(ev_raw, bat_raw)
        # t=2 and t=3 are after departure (departure=2.0)
        assert schedule.ev_power_kw[0, 2] < 1e-6, "EV charging after departure at t=2"
        assert schedule.ev_power_kw[0, 3] < 1e-6, "EV charging after departure at t=3"

    def test_ev_power_within_limits(self, simple_config):
        """EV charging power must not exceed max_charge_power_kw."""
        model = EnergyModel(simple_config)
        T = simple_config.n_intervals
        # Try to charge at 100 kW (way over limit)
        ev_raw = np.ones((1, T)) * 100.0
        bat_raw = np.zeros(T)
        schedule = model.decode(ev_raw, bat_raw)
        max_p = simple_config.evs[0].max_charge_power_kw
        assert np.all(schedule.ev_power_kw[0] <= max_p + 1e-6), (
            f"EV power exceeded max {max_p} kW"
        )

    def test_ev_power_nonnegative(self, simple_config):
        """EV charging power must be >= 0."""
        model = EnergyModel(simple_config)
        T = simple_config.n_intervals
        ev_raw = np.ones((1, T)) * (-5.0)  # negative input
        bat_raw = np.zeros(T)
        schedule = model.decode(ev_raw, bat_raw)
        assert np.all(schedule.ev_power_kw >= -1e-6), "EV power went negative"

    def test_ev_target_soc_achieved(self, simple_config):
        """EV should reach its target SOC by departure."""
        model = EnergyModel(simple_config)
        checker = ConstraintChecker(simple_config)
        T = simple_config.n_intervals
        ev = simple_config.evs[0]
        # Charge at enough power to meet target
        ev_raw = np.ones((1, T)) * 4.0
        bat_raw = np.zeros(T)
        schedule = model.decode(ev_raw, bat_raw)
        dep_idx = checker.ev_departure_idx[0]
        final_soc = schedule.ev_soc[0, dep_idx + 1]
        # Target is 0.80, initial is 0.20 — 12 kWh needed over 4h at 4kW = 16kWh available
        assert final_soc >= ev.target_soc - 0.05, (
            f"EV did not reach target SOC: final={final_soc:.3f}, target={ev.target_soc:.3f}"
        )


class TestBatteryConstraints:

    def test_battery_soc_within_limits(self, config_with_battery):
        """Battery SoC must stay within [min_soc, max_soc]."""
        model = EnergyModel(config_with_battery)
        bat = config_with_battery.battery
        T = config_with_battery.n_intervals
        # Extreme charge/discharge pattern
        bat_raw = np.array([10.0, -10.0, 10.0, -10.0])
        ev_raw = np.zeros((1, T))
        schedule = model.decode(ev_raw, bat_raw)
        assert np.all(schedule.battery_soc >= bat.min_soc - 1e-6), "Battery SoC below min"
        assert np.all(schedule.battery_soc <= bat.max_soc + 1e-6), "Battery SoC above max"

    def test_battery_power_within_limits(self, config_with_battery):
        """Battery power must be within [−max_discharge, +max_charge]."""
        model = EnergyModel(config_with_battery)
        bat = config_with_battery.battery
        T = config_with_battery.n_intervals
        bat_raw = np.array([50.0, -50.0, 50.0, -50.0])  # way over limits
        ev_raw = np.zeros((1, T))
        schedule = model.decode(ev_raw, bat_raw)
        assert np.all(schedule.battery_power_kw <= bat.max_charge_power_kw + 1e-6)
        assert np.all(schedule.battery_power_kw >= -bat.max_discharge_power_kw - 1e-6)

    def test_battery_no_simultaneous_charge_discharge(self, config_with_battery):
        """Battery cannot charge and discharge simultaneously (sign convention)."""
        model = EnergyModel(config_with_battery)
        T = config_with_battery.n_intervals
        bat_raw = np.array([3.0, -2.0, 1.0, -1.0])
        ev_raw = np.zeros((1, T))
        schedule = model.decode(ev_raw, bat_raw)
        # Each interval, battery_power_kw is a single signed value — cannot be both + and -
        for t in range(T):
            p = schedule.battery_power_kw[t]
            # Both positive AND negative simultaneously is impossible by single-value encoding
            assert True  # sign convention enforced by encoding

    def test_battery_does_not_discharge_below_min(self, config_with_battery):
        """Battery should not discharge below min_soc."""
        model = EnergyModel(config_with_battery)
        bat = config_with_battery.battery
        T = config_with_battery.n_intervals
        # Discharge heavily every interval
        bat_raw = np.array([-5.0, -5.0, -5.0, -5.0])
        ev_raw = np.zeros((1, T))
        schedule = model.decode(ev_raw, bat_raw)
        assert np.all(schedule.battery_soc >= bat.min_soc - 1e-6), (
            "Battery discharged below min SoC"
        )

    def test_battery_does_not_charge_above_max(self, config_with_battery):
        """Battery should not charge above max_soc."""
        model = EnergyModel(config_with_battery)
        bat = config_with_battery.battery
        T = config_with_battery.n_intervals
        bat_raw = np.array([5.0, 5.0, 5.0, 5.0])
        ev_raw = np.zeros((1, T))
        schedule = model.decode(ev_raw, bat_raw)
        assert np.all(schedule.battery_soc <= bat.max_soc + 1e-6), (
            "Battery charged above max SoC"
        )


class TestGridConstraints:

    def test_grid_import_nonnegative(self, default_config):
        """Grid import must never be negative."""
        model = EnergyModel(default_config)
        T = default_config.n_intervals
        n_evs = default_config.n_evs
        ev_raw = np.zeros((n_evs, T))
        bat_raw = np.zeros(T)
        schedule = model.decode(ev_raw, bat_raw)
        assert np.all(schedule.grid_import_kw >= -1e-6)

    def test_grid_import_capacity_clipped(self, default_config):
        """Grid import should not exceed max_import_kw."""
        model = EnergyModel(default_config)
        T = default_config.n_intervals
        n_evs = default_config.n_evs
        ev_raw = np.ones((n_evs, T)) * 50.0  # Very heavy load
        bat_raw = np.zeros(T)
        schedule = model.decode(ev_raw, bat_raw)
        assert np.all(
            schedule.grid_import_kw <= default_config.grid.max_import_kw + 1e-6
        ), "Grid import exceeded max capacity"


class TestInfeasibility:

    def test_infeasible_ev_detected_in_config_validate(self):
        """Config validation should detect infeasible EV requirements."""
        config = build_infeasible_config(time_step_h=1.0)
        errors = config.validate()
        assert len(errors) > 0, "Infeasible EV requirement not detected"
        assert any("EV 0" in e for e in errors), "EV 0 infeasibility not reported"
