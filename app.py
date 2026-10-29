"""
app.py — GridSwarm: Smart Energy Management Dashboard

A multi-page Streamlit application for bio-inspired energy schedule optimization.
Uses Particle Swarm Optimization (PSO) to coordinate EV charging, solar PV,
battery storage, and grid electricity across a 24-hour simulation horizon.

All data is SYNTHETIC / SIMULATED for academic demonstration purposes.
"""

import sys
import os

# Ensure src is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import time

from src.config import SystemConfig, EVConfig, BatteryConfig, GridConfig
from src.data_generator import (
    build_default_config,
    generate_solar_profile,
    generate_tariff_profile,
    generate_emissions_profile,
    build_infeasible_config,
)
from src.energy_model import EnergyModel
from src.baseline_scheduler import BaselineScheduler
from src.pso_optimizer import PSOOptimizer, PSOConfig
from src.objectives import ObjectiveWeights
from src.evaluation import Evaluator
from src.visualization import (
    plot_energy_schedule,
    plot_convergence,
    plot_comparison_bar,
    plot_solar_profile,
    plot_ev_soc_trajectories,
    plot_improvement_radar,
)
from src.export_utils import (
    schedule_to_dataframe,
    comparison_to_dataframe,
    config_to_dict,
    df_to_csv_bytes,
    dict_to_json_bytes,
)
from src.constraints import ConstraintChecker

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GridSwarm — Smart Energy Management",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Custom CSS (dark energy-management aesthetic)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b2a3d 100%);
    border-right: 1px solid rgba(99, 179, 237, 0.2);
}
[data-testid="stSidebar"] .stRadio label {
    color: #a0c4d8 !important;
    font-size: 0.9rem;
}

/* Main background */
.main .block-container {
    background: #0e1117;
    padding-top: 1.5rem;
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(23,42,69,0.9), rgba(30,57,90,0.9));
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #63b3ed;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-label {
    font-size: 0.8rem;
    color: #a0aec0;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.kpi-delta-pos { color: #48bb78; font-size: 0.9rem; }
.kpi-delta-neg { color: #fc8181; font-size: 0.9rem; }

/* Section headers */
.section-header {
    color: #63b3ed;
    font-size: 1.4rem;
    font-weight: 600;
    border-bottom: 2px solid rgba(99,179,237,0.3);
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}

/* Badge */
.badge-feasible {
    background: rgba(72,187,120,0.2);
    color: #48bb78;
    padding: 0.2rem 0.7rem;
    border-radius: 99px;
    font-size: 0.8rem;
    border: 1px solid rgba(72,187,120,0.4);
}
.badge-infeasible {
    background: rgba(252,129,129,0.2);
    color: #fc8181;
    padding: 0.2rem 0.7rem;
    border-radius: 99px;
    font-size: 0.8rem;
    border: 1px solid rgba(252,129,129,0.4);
}

/* Algorithm box */
.algo-box {
    background: rgba(23,42,69,0.6);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "system_config" not in st.session_state:
    st.session_state.system_config = build_default_config(time_step_h=1.0)
if "baseline_schedule" not in st.session_state:
    st.session_state.baseline_schedule = None
if "pso_result" not in st.session_state:
    st.session_state.pso_result = None
if "comparison" not in st.session_state:
    st.session_state.comparison = None
if "pso_ran" not in st.session_state:
    st.session_state.pso_ran = False


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0'>
        <div style='font-size:2.5rem'>⚡</div>
        <div style='font-size:1.3rem; font-weight:700; color:#63b3ed'>GridSwarm</div>
        <div style='font-size:0.75rem; color:#718096; margin-top:0.2rem'>Smart Energy Management</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        options=[
            "🏠 Overview",
            "⚙️ System Configuration",
            "🔬 Optimization",
            "📊 Energy Schedule",
            "📈 Results & Comparison",
            "🧠 Algorithm Explanation",
            "💾 Data & Export",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#4a5568; text-align:center'>
        Academic Simulation · Synthetic Data<br/>
        Bio-Inspired Algorithms Course
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: run baseline
# ─────────────────────────────────────────────────────────────────────────────
def ensure_baseline():
    if st.session_state.baseline_schedule is None:
        cfg = st.session_state.system_config
        scheduler = BaselineScheduler(cfg)
        st.session_state.baseline_schedule = scheduler.schedule()


def kpi_card(label: str, value: str, delta: str = "", positive_delta: bool = True):
    delta_class = "kpi-delta-pos" if positive_delta else "kpi-delta-neg"
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<h1 style="color:#63b3ed;">⚡ GridSwarm</h1>', unsafe_allow_html=True)
    st.markdown("""
    **GridSwarm** applies **Particle Swarm Optimization (PSO)** — a bio-inspired algorithm 
    modelling collective animal behaviour — to schedule EV charging, battery storage, and 
    grid electricity across a **24-hour simulation horizon**.
    
    > ⚠️ *All data is synthetic/simulated for academic demonstration. No real vehicles or infrastructure are connected.*
    """)

    ensure_baseline()

    cfg = st.session_state.system_config
    baseline = st.session_state.baseline_schedule

    st.markdown('<div class="section-header">📋 System Summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Electric Vehicles", str(cfg.n_evs), f"Fleet active"), unsafe_allow_html=True)
    with c2:
        bat_cap = f"{cfg.battery.capacity_kwh:.0f} kWh" if cfg.battery else "None"
        st.markdown(kpi_card("Battery Storage", bat_cap), unsafe_allow_html=True)
    with c3:
        solar_peak = f"{cfg.solar_profile_kw.max():.1f} kW" if cfg.solar_profile_kw is not None else "—"
        st.markdown(kpi_card("Solar Peak", solar_peak), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Grid Capacity", f"{cfg.grid.max_import_kw:.0f} kW"), unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Baseline Snapshot</div>', unsafe_allow_html=True)
    if baseline:
        evaluator = Evaluator(cfg)
        bm = evaluator.evaluate(baseline, label="Baseline")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.markdown(kpi_card("Total Cost", f"€{bm.total_cost_eur:.2f}", "Baseline"), unsafe_allow_html=True)
        with b2:
            st.markdown(kpi_card("Peak Demand", f"{bm.peak_demand_kw:.1f} kW", "Baseline"), unsafe_allow_html=True)
        with b3:
            st.markdown(kpi_card("CO₂ Emissions", f"{bm.total_emissions_kg:.1f} kg", "Baseline"), unsafe_allow_html=True)
        with b4:
            st.markdown(kpi_card("Solar Utilized", f"{bm.solar_utilization_pct:.1f}%", "Baseline"), unsafe_allow_html=True)

    if st.session_state.pso_ran and st.session_state.comparison:
        cmp = st.session_state.comparison
        st.markdown('<div class="section-header">🏆 PSO Improvement Over Baseline</div>', unsafe_allow_html=True)

        p1, p2, p3, p4 = st.columns(4)
        imp = cmp.improvements
        with p1:
            v = imp.get("cost_improvement_pct", 0)
            st.markdown(kpi_card("Cost Savings", f"{v:+.1f}%", "vs baseline", v > 0), unsafe_allow_html=True)
        with p2:
            v = imp.get("peak_reduction_pct", 0)
            st.markdown(kpi_card("Peak Reduction", f"{v:+.1f}%", "vs baseline", v > 0), unsafe_allow_html=True)
        with p3:
            v = imp.get("emissions_reduction_pct", 0)
            st.markdown(kpi_card("Emissions Cut", f"{v:+.1f}%", "vs baseline", v > 0), unsafe_allow_html=True)
        with p4:
            v = imp.get("solar_improvement_pct", 0)
            st.markdown(kpi_card("Solar Boost", f"{v:+.1f}%", "vs baseline", v > 0), unsafe_allow_html=True)

        st.plotly_chart(plot_comparison_bar(cmp), use_container_width=True)
    else:
        st.info("👉 Go to **🔬 Optimization** to run PSO and see results here.")

    st.plotly_chart(plot_solar_profile(cfg), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2: System Configuration
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚙️ System Configuration":
    st.markdown('<h2 style="color:#63b3ed;">⚙️ System Configuration</h2>', unsafe_allow_html=True)
    st.info("Configure the simulation parameters. Changes reset any previous optimization results.")

    cfg = st.session_state.system_config

    # Time settings
    st.markdown('<div class="section-header">⏱️ Time Horizon</div>', unsafe_allow_html=True)
    tc1, tc2 = st.columns(2)
    with tc1:
        time_step = st.selectbox("Time Step", [1.0, 0.5, 0.25],
                                 format_func=lambda x: f"{int(x*60)} minutes",
                                 index=0, key="cfg_time_step")
    with tc2:
        horizon = st.number_input("Horizon (hours)", 12, 48, int(cfg.time_horizon_h),
                                  step=1, key="cfg_horizon")

    n_intervals = int(round(horizon / time_step))
    st.caption(f"→ {n_intervals} time intervals")

    # Solar
    st.markdown('<div class="section-header">☀️ Solar PV Profile</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        solar_peak = st.number_input("Peak Power (kW)", 0.0, 100.0, 8.0, 0.5, key="cfg_solar_peak")
    with s2:
        solar_peak_hour = st.number_input("Peak Hour", 8.0, 16.0, 12.5, 0.5, key="cfg_peak_hour")
    with s3:
        solar_sigma = st.number_input("Width σ (hours)", 0.5, 5.0, 2.5, 0.5, key="cfg_solar_sigma")

    solar_profile = generate_solar_profile(n_intervals, time_step, solar_peak, solar_peak_hour, solar_sigma)
    tariff_profile = generate_tariff_profile(n_intervals, time_step)
    emissions_profile = generate_emissions_profile(n_intervals, time_step)

    # Grid
    st.markdown('<div class="section-header">🔌 Grid Connection</div>', unsafe_allow_html=True)
    grid_cap = st.number_input("Max Import Power (kW)", 10.0, 500.0,
                               float(cfg.grid.max_import_kw), 5.0, key="cfg_grid_cap")

    # Battery
    st.markdown('<div class="section-header">🔋 Stationary Battery</div>', unsafe_allow_html=True)
    use_battery = st.checkbox("Enable Battery Storage", value=cfg.battery is not None, key="cfg_use_battery")

    bat_config = None
    if use_battery:
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            bat_cap = st.number_input("Capacity (kWh)", 1.0, 500.0,
                                      float(cfg.battery.capacity_kwh) if cfg.battery else 20.0,
                                      1.0, key="cfg_bat_cap")
            bat_init_soc = st.slider("Initial SoC", 0.0, 1.0,
                                     float(cfg.battery.initial_soc) if cfg.battery else 0.5,
                                     0.05, key="cfg_bat_soc")
        with bc2:
            bat_min_soc = st.slider("Min SoC", 0.0, 0.5,
                                    float(cfg.battery.min_soc) if cfg.battery else 0.1,
                                    0.05, key="cfg_bat_min")
            bat_max_soc = st.slider("Max SoC", 0.5, 1.0,
                                    float(cfg.battery.max_soc) if cfg.battery else 0.9,
                                    0.05, key="cfg_bat_max")
        with bc3:
            bat_max_ch = st.number_input("Max Charge Power (kW)", 0.5, 50.0,
                                         float(cfg.battery.max_charge_power_kw) if cfg.battery else 5.0,
                                         0.5, key="cfg_bat_ch")
            bat_max_dis = st.number_input("Max Discharge Power (kW)", 0.5, 50.0,
                                          float(cfg.battery.max_discharge_power_kw) if cfg.battery else 5.0,
                                          0.5, key="cfg_bat_dis")
            bat_eff = st.slider("Efficiency", 0.7, 1.0,
                                float(cfg.battery.charge_efficiency) if cfg.battery else 0.95,
                                0.01, key="cfg_bat_eff")

        if bat_min_soc < bat_max_soc:
            bat_config = BatteryConfig(
                capacity_kwh=bat_cap,
                initial_soc=bat_init_soc,
                min_soc=bat_min_soc,
                max_soc=bat_max_soc,
                max_charge_power_kw=bat_max_ch,
                max_discharge_power_kw=bat_max_dis,
                charge_efficiency=bat_eff,
                discharge_efficiency=bat_eff,
            )
        else:
            st.error("Min SoC must be less than Max SoC")

    # EVs
    st.markdown('<div class="section-header">🚗 Electric Vehicles</div>', unsafe_allow_html=True)
    n_evs = st.number_input("Number of EVs", 1, 10, cfg.n_evs, 1, key="cfg_n_evs")

    ev_configs = []
    for i in range(int(n_evs)):
        ev_existing = cfg.evs[i] if i < len(cfg.evs) else None
        with st.expander(f"EV {i} Configuration", expanded=(i < 2)):
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                arr = st.number_input(f"Arrival Hour", 0.0, 23.0,
                                      float(ev_existing.arrival_hour) if ev_existing else float(6 + i),
                                      0.5, key=f"ev_{i}_arr")
                dep = st.number_input(f"Departure Hour", 1.0, 24.0,
                                      float(ev_existing.departure_hour) if ev_existing else float(16 + i),
                                      0.5, key=f"ev_{i}_dep")
            with ec2:
                cap = st.number_input(f"Battery Capacity (kWh)", 5.0, 200.0,
                                      float(ev_existing.battery_capacity_kwh) if ev_existing else 40.0,
                                      5.0, key=f"ev_{i}_cap")
                init_soc = st.slider(f"Initial SoC", 0.0, 1.0,
                                     float(ev_existing.initial_soc) if ev_existing else 0.25,
                                     0.05, key=f"ev_{i}_isoc")
            with ec3:
                tgt_soc = st.slider(f"Target SoC at Departure", 0.0, 1.0,
                                    float(ev_existing.target_soc) if ev_existing else 0.80,
                                    0.05, key=f"ev_{i}_tsoc")
                max_p = st.number_input(f"Max Charge Power (kW)", 1.0, 50.0,
                                        float(ev_existing.max_charge_power_kw) if ev_existing else 7.4,
                                        0.1, key=f"ev_{i}_mp")

            if dep > arr:
                ev_needed = max(0, (tgt_soc - init_soc)) * cap
                max_possible = max_p * (dep - arr)
                status = "✅ Feasible" if max_possible >= ev_needed - 0.1 else f"❌ Infeasible (needs {ev_needed:.1f} kWh, max {max_possible:.1f} kWh)"
                st.caption(f"Energy needed: {ev_needed:.1f} kWh | {status}")
                try:
                    ev_configs.append(EVConfig(
                        ev_id=i, arrival_hour=arr, departure_hour=dep,
                        battery_capacity_kwh=cap, initial_soc=init_soc,
                        target_soc=tgt_soc, max_charge_power_kw=max_p,
                    ))
                except AssertionError as e:
                    st.error(f"EV {i}: {e}")
            else:
                st.error(f"EV {i}: Departure must be after arrival.")

    # Apply config button
    st.markdown("---")
    if st.button("✅ Apply Configuration", type="primary", key="apply_cfg"):
        if ev_configs:
            new_config = SystemConfig(
                time_horizon_h=float(horizon),
                time_step_h=time_step,
                evs=ev_configs,
                battery=bat_config,
                grid=GridConfig(max_import_kw=grid_cap),
                solar_profile_kw=solar_profile,
                tariff_profile=tariff_profile,
                emissions_profile=emissions_profile,
            )
            errors = new_config.validate()
            if errors:
                for e in errors:
                    st.error(e)
            st.session_state.system_config = new_config
            st.session_state.baseline_schedule = None
            st.session_state.pso_result = None
            st.session_state.comparison = None
            st.session_state.pso_ran = False
            st.success("✅ Configuration applied! Baseline will be recalculated.")
        else:
            st.error("At least one valid EV configuration is required.")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3: Optimization
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Optimization":
    st.markdown('<h2 style="color:#63b3ed;">🔬 PSO Optimization</h2>', unsafe_allow_html=True)

    cfg = st.session_state.system_config

    # Config validation warnings
    errors = cfg.validate()
    if errors:
        for e in errors:
            st.warning(f"⚠️ {e}")

    st.markdown('<div class="section-header">🔧 PSO Hyperparameters</div>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        n_particles = st.number_input("Population Size", 5, 200, 40, 5, key="pso_n")
        n_iterations = st.number_input("Max Iterations", 10, 1000, 100, 10, key="pso_iter")
    with pc2:
        w_max = st.slider("Inertia w_max", 0.4, 1.0, 0.9, 0.05, key="pso_wmax")
        w_min = st.slider("Inertia w_min", 0.1, 0.8, 0.4, 0.05, key="pso_wmin")
    with pc3:
        c1_val = st.number_input("Cognitive c1", 0.5, 4.0, 2.05, 0.05, key="pso_c1")
        c2_val = st.number_input("Social c2", 0.5, 4.0, 2.05, 0.05, key="pso_c2")

    seed = st.number_input("Random Seed (-1 = random)", -1, 999999, 42, 1, key="pso_seed")
    seed_val = int(seed) if seed >= 0 else None

    st.markdown('<div class="section-header">🎯 Objective Weights</div>', unsafe_allow_html=True)
    st.caption("Adjust the importance of each objective. Higher weight = more emphasis.")
    ow1, ow2, ow3, ow4 = st.columns(4)
    with ow1:
        w_cost = st.slider("💰 Cost", 0.0, 1.0, 0.40, 0.05, key="w_cost")
    with ow2:
        w_peak = st.slider("📈 Peak Demand", 0.0, 1.0, 0.30, 0.05, key="w_peak")
    with ow3:
        w_emit = st.slider("🌱 Emissions", 0.0, 1.0, 0.20, 0.05, key="w_emit")
    with ow4:
        w_solar = st.slider("☀️ Solar Util.", 0.0, 1.0, 0.10, 0.05, key="w_solar")

    total_w = w_cost + w_peak + w_emit + w_solar
    if total_w < 0.01:
        st.error("At least one weight must be positive.")
    else:
        st.caption(f"Total weight: {total_w:.2f} (weights applied as-is, no auto-normalization)")

    st.markdown("---")

    # Run button
    if st.button("🚀 Run PSO Optimization", type="primary", key="run_pso"):
        with st.spinner("Running PSO... please wait"):
            ensure_baseline()

            weights = ObjectiveWeights(
                cost=w_cost, peak=w_peak, emissions=w_emit, solar=w_solar
            )
            pso_cfg = PSOConfig(
                n_particles=int(n_particles),
                n_iterations=int(n_iterations),
                w_max=float(w_max),
                w_min=float(w_min),
                c1=float(c1_val),
                c2=float(c2_val),
                seed=seed_val,
            )

            progress_bar = st.progress(0)
            progress_text = st.empty()
            fitness_display = st.empty()

            def progress_callback(iter_num: int, best_fit: float):
                pct = int(100 * iter_num / n_iterations)
                progress_bar.progress(pct)
                progress_text.text(f"Iteration {iter_num}/{int(n_iterations)}")
                fitness_display.metric("Current Best Fitness", f"{best_fit:.6f}")

            pso = PSOOptimizer(cfg, pso_cfg, weights, progress_callback)
            result = pso.run()

            progress_bar.progress(100)
            progress_text.text("✅ Optimization complete!")

            st.session_state.pso_result = result
            st.session_state.pso_ran = True

            # Build comparison
            evaluator = Evaluator(cfg)
            comparison = evaluator.compare(
                st.session_state.baseline_schedule,
                result.best_schedule,
                pso_runtime=result.runtime_seconds,
                pso_fitness=result.best_fitness,
                pso_convergence=result.convergence_history,
            )
            st.session_state.comparison = comparison

    # Show results if available
    if st.session_state.pso_ran and st.session_state.pso_result:
        result = st.session_state.pso_result
        st.markdown('<div class="section-header">📊 Optimization Results</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Best Fitness", f"{result.best_fitness:.6f}")
        with r2:
            st.metric("Runtime", f"{result.runtime_seconds:.2f} s")
        with r3:
            st.metric("Iterations Run", str(result.n_iterations_run))

        st.plotly_chart(plot_convergence(result.convergence_history), use_container_width=True)

        # Feasibility check
        checker = ConstraintChecker(cfg)
        report = checker.check(result.best_schedule)
        st.markdown('<div class="section-header">✅ Constraint Validation</div>', unsafe_allow_html=True)
        if report.is_feasible:
            st.markdown('<span class="badge-feasible">✅ Schedule is FEASIBLE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-infeasible">❌ Constraint Violations Detected</span>', unsafe_allow_html=True)

        if report.violations:
            with st.expander("View Violations"):
                for v in report.violations:
                    icon = "❌" if v.severity == "error" else "⚠️"
                    st.markdown(f"{icon} **[{v.constraint}]** {v.message}")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 4: Energy Schedule
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Energy Schedule":
    st.markdown('<h2 style="color:#63b3ed;">📊 Energy Schedule Visualization</h2>', unsafe_allow_html=True)

    ensure_baseline()
    cfg = st.session_state.system_config

    tab1, tab2 = st.tabs(["🔵 Baseline Schedule", "🟠 PSO Optimized Schedule"])

    with tab1:
        if st.session_state.baseline_schedule:
            sched = st.session_state.baseline_schedule
            st.plotly_chart(plot_energy_schedule(sched, "Baseline Schedule — Rule-Based EDF"),
                            use_container_width=True)
            st.plotly_chart(plot_ev_soc_trajectories(sched), use_container_width=True)
        else:
            st.info("Baseline schedule not yet computed.")

    with tab2:
        if st.session_state.pso_ran and st.session_state.pso_result:
            sched = st.session_state.pso_result.best_schedule
            st.plotly_chart(plot_energy_schedule(sched, "PSO Optimized Schedule"),
                            use_container_width=True)
            st.plotly_chart(plot_ev_soc_trajectories(sched), use_container_width=True)
        else:
            st.info("👉 Run PSO optimization first (🔬 Optimization page).")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 5: Results & Comparison
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Results & Comparison":
    st.markdown('<h2 style="color:#63b3ed;">📈 Results & Comparison</h2>', unsafe_allow_html=True)

    ensure_baseline()
    cfg = st.session_state.system_config

    if not st.session_state.pso_ran or st.session_state.comparison is None:
        st.info("👉 Run PSO optimization first (🔬 Optimization page).")
    else:
        cmp = st.session_state.comparison
        b = cmp.baseline
        p = cmp.pso
        imp = cmp.improvements

        st.markdown('<div class="section-header">🏅 Side-by-Side Comparison</div>', unsafe_allow_html=True)

        comparison_data = {
            "Metric": [
                "💰 Total Cost (€)",
                "📈 Peak Demand (kW)",
                "🌱 CO₂ Emissions (kg)",
                "☀️ Solar Utilization (%)",
                "🚗 EV Satisfaction (%)",
                "⚡ Grid Energy (kWh)",
            ],
            "Baseline": [
                f"{b.total_cost_eur:.2f}",
                f"{b.peak_demand_kw:.2f}",
                f"{b.total_emissions_kg:.2f}",
                f"{b.solar_utilization_pct:.1f}",
                f"{b.ev_satisfaction_pct:.1f}",
                f"{b.total_grid_energy_kwh:.2f}",
            ],
            "PSO": [
                f"{p.total_cost_eur:.2f}",
                f"{p.peak_demand_kw:.2f}",
                f"{p.total_emissions_kg:.2f}",
                f"{p.solar_utilization_pct:.1f}",
                f"{p.ev_satisfaction_pct:.1f}",
                f"{p.total_grid_energy_kwh:.2f}",
            ],
            "Improvement": [
                f"{imp.get('cost_improvement_pct', 0):+.1f}%",
                f"{imp.get('peak_reduction_pct', 0):+.1f}%",
                f"{imp.get('emissions_reduction_pct', 0):+.1f}%",
                f"{imp.get('solar_improvement_pct', 0):+.1f}%",
                "—",
                "—",
            ],
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_comparison_bar(cmp), use_container_width=True)
        with col2:
            st.plotly_chart(plot_improvement_radar(cmp), use_container_width=True)

        if st.session_state.pso_result:
            st.plotly_chart(plot_convergence(st.session_state.pso_result.convergence_history,
                                              "PSO Convergence"), use_container_width=True)

        # Feasibility badges
        st.markdown('<div class="section-header">✅ Feasibility Status</div>', unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)
        with fc1:
            baseline_badge = ('<span class="badge-feasible">✅ Feasible</span>' if b.is_feasible else '<span class="badge-infeasible">❌ Infeasible</span>')
            st.markdown(f"**Baseline:** {baseline_badge}", unsafe_allow_html=True)
            if b.constraint_report.violations:
                with st.expander("Baseline Violations"):
                    for v in b.constraint_report.violations:
                        st.markdown(f"- {v.message}")
        with fc2:
            pso_badge = ('<span class="badge-feasible">✅ Feasible</span>' if p.is_feasible else '<span class="badge-infeasible">❌ Infeasible</span>')
            st.markdown(f"**PSO:** {pso_badge}", unsafe_allow_html=True)
            if p.constraint_report.violations:
                with st.expander("PSO Violations"):
                    for v in p.constraint_report.violations:
                        st.markdown(f"- {v.message}")

        # Auto-generated narrative
        st.markdown('<div class="section-header">📝 Auto-Generated Summary</div>', unsafe_allow_html=True)
        cost_dir = "reduced" if imp.get("cost_improvement_pct", 0) > 0 else "increased"
        peak_dir = "reduced" if imp.get("peak_reduction_pct", 0) > 0 else "increased"
        emit_dir = "reduced" if imp.get("emissions_reduction_pct", 0) > 0 else "increased"
        solar_dir = "improved" if imp.get("solar_improvement_pct", 0) > 0 else "decreased"
        narrative = (
            f"PSO optimization **{cost_dir} total electricity cost** by "
            f"**{abs(imp.get('cost_improvement_pct', 0)):.1f}%** (€{b.total_cost_eur:.2f} → €{p.total_cost_eur:.2f}). "
            f"Peak grid demand was **{peak_dir}** by **{abs(imp.get('peak_reduction_pct', 0)):.1f}%** "
            f"({b.peak_demand_kw:.1f} → {p.peak_demand_kw:.1f} kW). "
            f"Carbon emissions were **{emit_dir}** by **{abs(imp.get('emissions_reduction_pct', 0)):.1f}%** "
            f"({b.total_emissions_kg:.1f} → {p.total_emissions_kg:.1f} kg CO₂). "
            f"Solar self-consumption **{solar_dir}** by **{abs(imp.get('solar_improvement_pct', 0)):.1f}%** "
            f"({b.solar_utilization_pct:.1f}% → {p.solar_utilization_pct:.1f}%). "
        )
        if p.runtime_seconds:
            narrative += f"PSO ran {p.runtime_seconds:.1f} seconds with {st.session_state.pso_result.n_iterations_run} iterations."
        st.info(narrative)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 6: Algorithm Explanation
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧠 Algorithm Explanation":
    st.markdown('<h2 style="color:#63b3ed;">🧠 PSO Algorithm Explanation</h2>', unsafe_allow_html=True)

    st.markdown("""
    ## Why is PSO Bio-Inspired?

    **Particle Swarm Optimization (PSO)** was proposed by Kennedy & Eberhart (1995).
    It models the collective movement of *birds flocking* and *fish schooling*.
    Animals in a swarm share information about food sources — the flock adapts its movement
    based on what any individual has discovered, accelerating the search.

    In PSO, each **particle** is a candidate energy schedule. The swarm collectively
    searches for the schedule that minimizes electricity cost, peak demand, and emissions
    while maximizing solar use.

    ---
    ## Particle Encoding

    Each particle is a 1-D vector of length **(n_EVs × T) + T**:

    ```
    [ EV_0_t0, EV_0_t1, ..., EV_0_T-1,
      EV_1_t0, ..., EV_n-1_T-1,
      Battery_t0, Battery_t1, ..., Battery_T-1 ]
    ```

    - EV elements ∈ [0, max_charge_power_kW] (charging rate per interval)
    - Battery elements ∈ [−max_discharge, +max_charge] (signed; + = charge)

    ---
    ## Velocity and Position Updates

    ```
    v(i,t+1) = w(t) · v(i,t)
             + c1 · r1 · (pbest(i) − x(i,t))    ← cognitive (personal best)
             + c2 · r2 · (gbest    − x(i,t))    ← social (swarm best)

    x(i,t+1) = x(i,t) + v(i,t+1)
    ```

    - **w(t)**: inertia weight (linear decay from w_max → w_min)
    - **c1**: cognitive coefficient — attraction to personal best
    - **c2**: social coefficient — attraction to global best
    - **r1, r2**: random values in [0,1] (adds stochasticity)

    ---
    ## Fitness Function (Weighted Sum)

    ```
    J = w_cost     × norm_cost
      + w_peak     × norm_peak_demand
      + w_emissions × norm_emissions
      + w_solar    × (1 − solar_utilization)
    ```

    Each objective is independently normalized to [0,1] using theoretical upper bounds.
    **Lower J is better.** This is a *weighted-sum* formulation, NOT a Pareto-front optimizer.

    ---
    ## Constraint Handling (Repair Strategy)

    After each velocity/position update, particles are **decoded and repaired**:

    1. **EV availability**: Zero charging power outside [arrival, departure]
    2. **EV power limits**: Clip to [0, max_charge_power_kW]
    3. **EV energy target**: Scale or boost charging to meet target SOC
    4. **Battery SoC limits**: Reduce power when SoC would violate min/max
    5. **Battery power limits**: Clip to rated power
    6. **Grid import**: Computed residually; negative values clipped to 0

    This **repair approach** ensures every evaluated schedule is physically meaningful,
    avoiding pure penalty methods that often evaluate infeasible schedules.

    ---
    ## Stopping Criteria

    - Maximum iterations reached (configurable, default 100)
    - No early stopping — the full budget is used for reliable convergence

    ---
    ## Limitations & Assumptions

    - **Weighted-sum multi-objective** — not Pareto-optimal
    - **Deterministic solar** — no forecast uncertainty
    - **No V2G** (vehicle-to-grid EV discharge)
    - **Single building scope** — no distribution grid modelling
    - **Repair creates bias** toward feasibility boundary — solver may not explore infeasible regions
    - Runtime scales with population size × iterations × n_EVs × T
    """)

    st.markdown('<div class="section-header">📐 PSO Pseudocode</div>', unsafe_allow_html=True)
    st.code("""
INITIALIZE swarm (n_particles positions and velocities within bounds)
EVALUATE fitness for each particle
SET pbest[i] = positions[i]  for each particle i
SET gbest = particle with minimum fitness

FOR iteration = 1 to max_iterations:
    w = w_max - (w_max - w_min) × iteration / max_iterations   ← linear decay

    FOR each particle i:
        r1, r2 ← uniform random in [0, 1]
        v[i] = w × v[i]
             + c1 × r1 × (pbest[i] - x[i])
             + c2 × r2 × (gbest    - x[i])
        v[i] = CLAMP(v[i], -v_max, +v_max)
        x[i] = CLAMP(x[i] + v[i], lb, ub)

        schedule = DECODE_AND_REPAIR(x[i])  ← apply physical constraints
        fitness  = EVALUATE_OBJECTIVES(schedule)

        IF fitness < pbest_fitness[i]:
            pbest[i] = x[i]
            IF fitness < gbest_fitness:
                gbest = x[i]

RETURN gbest → decode → optimal energy schedule
""", language="text")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 7: Data & Export
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💾 Data & Export":
    st.markdown('<h2 style="color:#63b3ed;">💾 Data & Export</h2>', unsafe_allow_html=True)

    ensure_baseline()
    cfg = st.session_state.system_config

    # Input data overview
    st.markdown('<div class="section-header">📋 Input Data Preview</div>', unsafe_allow_html=True)
    hours = cfg.time_hours
    input_df = pd.DataFrame({
        "Hour": hours,
        "Solar (kW)": cfg.solar_profile_kw if cfg.solar_profile_kw is not None else [0] * cfg.n_intervals,
        "Tariff (€/kWh)": cfg.tariff_profile if cfg.tariff_profile is not None else [0] * cfg.n_intervals,
        "Emissions (kg CO₂/kWh)": cfg.emissions_profile if cfg.emissions_profile is not None else [0] * cfg.n_intervals,
    })
    st.dataframe(input_df.round(4), use_container_width=True, hide_index=True)

    # EV config table
    st.markdown("**EV Fleet Configuration:**")
    ev_df = pd.DataFrame([{
        "EV ID": ev.ev_id,
        "Arrival (h)": ev.arrival_hour,
        "Departure (h)": ev.departure_hour,
        "Capacity (kWh)": ev.battery_capacity_kwh,
        "Initial SoC": ev.initial_soc,
        "Target SoC": ev.target_soc,
        "Max Power (kW)": ev.max_charge_power_kw,
        "Energy Needed (kWh)": round(ev.energy_needed_kwh, 2),
    } for ev in cfg.evs])
    st.dataframe(ev_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">📥 Downloads</div>', unsafe_allow_html=True)

    dcol1, dcol2 = st.columns(2)

    with dcol1:
        # Download input data
        csv_bytes = df_to_csv_bytes(input_df)
        st.download_button(
            label="📥 Download Input Profiles (CSV)",
            data=csv_bytes,
            file_name="gridswarm_inputs.csv",
            mime="text/csv",
            key="dl_inputs",
        )

        # Download config JSON
        config_dict = config_to_dict(cfg)
        json_bytes = dict_to_json_bytes(config_dict)
        st.download_button(
            label="📥 Download Configuration (JSON)",
            data=json_bytes,
            file_name="gridswarm_config.json",
            mime="application/json",
            key="dl_config",
        )

    with dcol2:
        # Download baseline schedule
        if st.session_state.baseline_schedule:
            baseline_df = schedule_to_dataframe(st.session_state.baseline_schedule)
            st.download_button(
                label="📥 Download Baseline Schedule (CSV)",
                data=df_to_csv_bytes(baseline_df),
                file_name="gridswarm_baseline_schedule.csv",
                mime="text/csv",
                key="dl_baseline",
            )

        # Download PSO schedule
        if st.session_state.pso_ran and st.session_state.pso_result:
            pso_df = schedule_to_dataframe(st.session_state.pso_result.best_schedule)
            st.download_button(
                label="📥 Download PSO Schedule (CSV)",
                data=df_to_csv_bytes(pso_df),
                file_name="gridswarm_pso_schedule.csv",
                mime="text/csv",
                key="dl_pso",
            )

        # Download comparison
        if st.session_state.comparison:
            cmp_df = comparison_to_dataframe(st.session_state.comparison)
            st.download_button(
                label="📥 Download Results Summary (CSV)",
                data=df_to_csv_bytes(cmp_df),
                file_name="gridswarm_results_summary.csv",
                mime="text/csv",
                key="dl_summary",
            )

    if st.session_state.pso_ran and st.session_state.pso_result:
        st.markdown('<div class="section-header">📋 PSO Schedule Preview</div>', unsafe_allow_html=True)
        pso_df = schedule_to_dataframe(st.session_state.pso_result.best_schedule)
        st.dataframe(pso_df.round(4), use_container_width=True, hide_index=True)
