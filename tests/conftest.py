"""
conftest.py — Shared pytest fixtures for GridSwarm tests.
"""
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from src.config import SystemConfig, EVConfig, BatteryConfig, GridConfig
from src.data_generator import build_default_config, generate_default_battery
from src.energy_model import EnergyModel


@pytest.fixture
def simple_config():
    """Minimal 4-hour config with 1 EV and no battery, for fast unit tests."""
    n_intervals = 4
    config = SystemConfig(
        time_horizon_h=4.0,
        time_step_h=1.0,
        evs=[
            EVConfig(
                ev_id=0,
                arrival_hour=0.0,
                departure_hour=4.0,
                battery_capacity_kwh=20.0,
                initial_soc=0.20,
                target_soc=0.80,
                max_charge_power_kw=7.4,
            )
        ],
        battery=None,
        grid=GridConfig(max_import_kw=50.0),
        solar_profile_kw=np.array([0.0, 2.0, 4.0, 2.0]),
        tariff_profile=np.array([0.10, 0.22, 0.32, 0.14]),
        emissions_profile=np.array([0.35, 0.30, 0.25, 0.30]),
    )
    return config


@pytest.fixture
def default_config():
    """Full 24-hour default config with 5 EVs and battery."""
    return build_default_config(time_step_h=1.0)


@pytest.fixture
def config_with_battery():
    """4-hour config with 1 EV and battery."""
    config = SystemConfig(
        time_horizon_h=4.0,
        time_step_h=1.0,
        evs=[
            EVConfig(
                ev_id=0,
                arrival_hour=0.0,
                departure_hour=4.0,
                battery_capacity_kwh=20.0,
                initial_soc=0.20,
                target_soc=0.80,
                max_charge_power_kw=7.4,
            )
        ],
        battery=BatteryConfig(
            capacity_kwh=10.0,
            initial_soc=0.50,
            min_soc=0.10,
            max_soc=0.90,
            max_charge_power_kw=5.0,
            max_discharge_power_kw=5.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
        ),
        grid=GridConfig(max_import_kw=50.0),
        solar_profile_kw=np.array([0.0, 3.0, 5.0, 1.0]),
        tariff_profile=np.array([0.10, 0.22, 0.32, 0.14]),
        emissions_profile=np.array([0.35, 0.30, 0.25, 0.30]),
    )
    return config
