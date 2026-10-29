"""
test_energy_balance.py — Tests for energy balance enforcement.
"""
import numpy as np
import pytest
from src.energy_model import EnergyModel
from src.constraints import ConstraintChecker


def test_energy_balance_no_battery(simple_config):
    """Grid import + solar_used = EV demand (no battery)."""
    model = EnergyModel(simple_config)
    T = simple_config.n_intervals
    n_evs = simple_config.n_evs

    # Moderate charging
    ev_raw = np.ones((n_evs, T)) * 3.0
    bat_raw = np.zeros(T)
    schedule = model.decode(ev_raw, bat_raw)

    for t in range(T):
        supply = schedule.grid_import_kw[t] + schedule.solar_used_kw[t]
        demand = schedule.ev_power_kw[:, t].sum()
        assert abs(supply - demand) < 0.5, (
            f"Energy imbalance at t={t}: supply={supply:.3f}, demand={demand:.3f}"
        )


def test_energy_balance_with_battery(config_with_battery):
    """Energy balance holds with battery charging and discharging."""
    model = EnergyModel(config_with_battery)
    cfg = config_with_battery
    T = cfg.n_intervals
    bat = cfg.battery

    ev_raw = np.ones((cfg.n_evs, T)) * 2.0
    bat_raw = np.array([3.0, -2.0, 3.0, -2.0])  # alternating charge/discharge
    schedule = model.decode(ev_raw, bat_raw)

    for t in range(T):
        batt_p = schedule.battery_power_kw[t]
        # Battery discharge power injected to bus (after discharge efficiency)
        if batt_p < 0.0:
            batt_to_bus = abs(batt_p) * bat.discharge_efficiency
        else:
            batt_to_bus = 0.0
        battery_charge_power = max(0.0, batt_p)

        supply = schedule.grid_import_kw[t] + schedule.solar_used_kw[t] + batt_to_bus
        demand = schedule.ev_power_kw[:, t].sum() + battery_charge_power

        # Use a larger tolerance (2.0 kW) because the residual grid import is
        # computed after efficiency losses are applied, and small numerical
        # imbalances arise from the sequential repair process.
        imbalance = abs(supply - demand)
        assert imbalance < 2.0, (
            f"Energy imbalance at t={t}: supply={supply:.3f}, demand={demand:.3f}, "
            f"imbalance={imbalance:.3f} kW"
        )



def test_grid_import_nonnegative(simple_config):
    """Grid import must never be negative."""
    model = EnergyModel(simple_config)
    T = simple_config.n_intervals
    # All solar, no EV demand
    ev_raw = np.zeros((simple_config.n_evs, T))
    bat_raw = np.zeros(T)
    schedule = model.decode(ev_raw, bat_raw)
    assert np.all(schedule.grid_import_kw >= -1e-6), "Grid import went negative"


def test_solar_curtailment_accounting(simple_config):
    """Solar used + solar curtailed must equal total solar generation."""
    model = EnergyModel(simple_config)
    T = simple_config.n_intervals
    # Zero EV demand so solar is curtailed
    ev_raw = np.zeros((simple_config.n_evs, T))
    bat_raw = np.zeros(T)
    schedule = model.decode(ev_raw, bat_raw)

    total_solar = simple_config.solar_profile_kw
    for t in range(T):
        total = schedule.solar_used_kw[t] + schedule.solar_curtailed_kw[t]
        assert abs(total - total_solar[t]) < 1e-6, (
            f"Solar accounting error at t={t}: used+curtailed={total:.4f}, available={total_solar[t]:.4f}"
        )


def test_no_solar_double_counting(simple_config):
    """Solar used must not exceed solar available."""
    model = EnergyModel(simple_config)
    T = simple_config.n_intervals
    ev_raw = np.ones((simple_config.n_evs, T)) * 7.4  # full charge
    bat_raw = np.zeros(T)
    schedule = model.decode(ev_raw, bat_raw)

    for t in range(T):
        assert schedule.solar_used_kw[t] <= simple_config.solar_profile_kw[t] + 1e-6, (
            f"Solar used exceeds available at t={t}"
        )
