"""
visualization.py — Plotly chart builders for GridSwarm dashboard.

All functions return plotly.graph_objects.Figure objects.
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .energy_model import EnergySchedule
from .config import SystemConfig
from .evaluation import ScheduleMetrics, ComparisonResult

# Consistent colour palette
COLORS = {
    "solar": "#FFA500",
    "grid": "#4169E1",
    "battery_charge": "#32CD32",
    "battery_discharge": "#FF6347",
    "battery_soc": "#9370DB",
    "ev_total": "#20B2AA",
    "tariff": "#DC143C",
    "baseline": "#6495ED",
    "pso": "#FF8C00",
    "emissions": "#708090",
    "curtailed": "#DDA0DD",
}


def _time_labels(hours: np.ndarray) -> List[str]:
    """Convert float hours to 'HH:MM' strings."""
    labels = []
    for h in hours:
        hh = int(h)
        mm = int((h - hh) * 60)
        labels.append(f"{hh:02d}:{mm:02d}")
    return labels


def plot_energy_schedule(schedule: EnergySchedule, title: str = "Energy Schedule") -> go.Figure:
    """
    Interactive stacked chart showing all power flows over the simulation horizon.
    """
    hours = schedule.time_hours
    xlabels = _time_labels(hours)
    n_evs = schedule.config.n_evs

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Power Flows (kW)",
            "Battery State of Charge",
            "EV Charging Power (kW)",
            "Electricity Tariff (€/kWh)",
        ),
        vertical_spacing=0.07,
        row_heights=[0.40, 0.20, 0.25, 0.15],
    )

    # Row 1: Power flows
    fig.add_trace(go.Bar(
        x=xlabels, y=schedule.grid_import_kw,
        name="Grid Import", marker_color=COLORS["grid"],
        opacity=0.85,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=xlabels, y=schedule.solar_used_kw,
        name="Solar Used", marker_color=COLORS["solar"],
        opacity=0.85,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=xlabels, y=schedule.solar_curtailed_kw,
        name="Solar Curtailed", marker_color=COLORS["curtailed"],
        opacity=0.6,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=xlabels, y=schedule.battery_charge_kw,
        name="Battery Charge", marker_color=COLORS["battery_charge"],
        opacity=0.85,
    ), row=1, col=1)

    discharge_kw = schedule.battery_discharge_kw
    fig.add_trace(go.Bar(
        x=xlabels, y=-discharge_kw,
        name="Battery Discharge", marker_color=COLORS["battery_discharge"],
        opacity=0.85,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=xlabels, y=schedule.total_ev_power_kw,
        name="Total EV Demand", line=dict(color=COLORS["ev_total"], width=2.5),
        mode="lines",
    ), row=1, col=1)

    # Row 2: Battery SoC
    if schedule.config.battery is not None:
        bat = schedule.config.battery
        soc_pct = schedule.battery_soc[:-1] * 100  # exclude final extra point for display
        soc_labels_ext = _time_labels(np.append(hours, hours[-1] + schedule.config.time_step_h))
        fig.add_trace(go.Scatter(
            x=soc_labels_ext[:-1], y=soc_pct,
            name="Battery SoC (%)", line=dict(color=COLORS["battery_soc"], width=2),
            fill="tozeroy", fillcolor="rgba(147,112,219,0.15)",
        ), row=2, col=1)
        fig.add_hline(
            y=bat.min_soc * 100, line_dash="dot", line_color="red",
            annotation_text="Min SoC", row=2, col=1,
        )
        fig.add_hline(
            y=bat.max_soc * 100, line_dash="dot", line_color="green",
            annotation_text="Max SoC", row=2, col=1,
        )

    # Row 3: Per-EV charging
    ev_colors = px.colors.qualitative.Set2
    for i, ev in enumerate(schedule.config.evs):
        c = ev_colors[i % len(ev_colors)]
        fig.add_trace(go.Bar(
            x=xlabels, y=schedule.ev_power_kw[i],
            name=f"EV {ev.ev_id}", marker_color=c, opacity=0.8,
        ), row=3, col=1)

    # Row 4: Tariff
    if schedule.config.tariff_profile is not None:
        fig.add_trace(go.Scatter(
            x=xlabels, y=schedule.config.tariff_profile,
            name="Tariff (€/kWh)", line=dict(color=COLORS["tariff"], width=2),
            fill="tozeroy", fillcolor="rgba(220,20,60,0.10)",
        ), row=4, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        barmode="stack",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        margin=dict(l=60, r=20, t=80, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    return fig


def plot_convergence(convergence_history: List[float], title: str = "PSO Convergence") -> go.Figure:
    """Plot PSO fitness convergence over iterations."""
    iterations = list(range(len(convergence_history)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iterations,
        y=convergence_history,
        mode="lines+markers",
        line=dict(color=COLORS["pso"], width=2.5),
        marker=dict(size=4),
        name="Best Fitness",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Composite Fitness (lower = better)",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=350,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    return fig


def plot_comparison_bar(comparison: ComparisonResult) -> go.Figure:
    """Side-by-side bar chart of key metrics."""
    metrics_labels = ["Cost (€)", "Peak Demand (kW)", "CO₂ Emissions (kg)", "Solar Util. (%)"]
    baseline_vals = [
        comparison.baseline.total_cost_eur,
        comparison.baseline.peak_demand_kw,
        comparison.baseline.total_emissions_kg,
        comparison.baseline.solar_utilization_pct,
    ]
    pso_vals = [
        comparison.pso.total_cost_eur,
        comparison.pso.peak_demand_kw,
        comparison.pso.total_emissions_kg,
        comparison.pso.solar_utilization_pct,
    ]

    fig = go.Figure(data=[
        go.Bar(name="Baseline", x=metrics_labels, y=baseline_vals,
               marker_color=COLORS["baseline"], opacity=0.85),
        go.Bar(name="PSO Optimized", x=metrics_labels, y=pso_vals,
               marker_color=COLORS["pso"], opacity=0.85),
    ])
    fig.update_layout(
        barmode="group",
        title="Baseline vs. PSO: Key Performance Metrics",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=400,
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    return fig


def plot_solar_profile(config: SystemConfig) -> go.Figure:
    """Plot solar generation profile alongside electricity tariff."""
    hours = config.time_hours
    xlabels = _time_labels(hours)
    solar = config.solar_profile_kw if config.solar_profile_kw is not None else np.zeros(len(hours))
    tariff = config.tariff_profile if config.tariff_profile is not None else np.zeros(len(hours))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=xlabels, y=solar, name="Solar PV (kW)",
               marker_color=COLORS["solar"], opacity=0.85),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=xlabels, y=tariff, name="Tariff (€/kWh)",
                   line=dict(color=COLORS["tariff"], width=2.5)),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Solar Power (kW)", secondary_y=False)
    fig.update_yaxes(title_text="Tariff (€/kWh)", secondary_y=True)
    fig.update_layout(
        title="Solar Generation Profile & Electricity Tariff",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=350,
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_ev_soc_trajectories(schedule: EnergySchedule) -> go.Figure:
    """Plot EV State-of-Charge trajectories over time."""
    cfg = schedule.config
    hours_ext = np.append(cfg.time_hours, cfg.time_hours[-1] + cfg.time_step_h)
    xlabels = _time_labels(hours_ext)
    ev_colors = px.colors.qualitative.Set2

    fig = go.Figure()
    for i, ev in enumerate(cfg.evs):
        c = ev_colors[i % len(ev_colors)]
        soc_pct = schedule.ev_soc[i] * 100
        fig.add_trace(go.Scatter(
            x=xlabels, y=soc_pct,
            name=f"EV {ev.ev_id}",
            line=dict(color=c, width=2),
            mode="lines",
        ))
        # Target SOC marker at departure
        fig.add_hline(
            y=ev.target_soc * 100,
            line_dash="dot", line_color=c,
            opacity=0.4,
        )

    fig.update_layout(
        title="EV State-of-Charge Trajectories",
        xaxis_title="Time of Day",
        yaxis_title="State of Charge (%)",
        yaxis=dict(range=[0, 105]),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=400,
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    return fig


def plot_improvement_radar(comparison: ComparisonResult) -> go.Figure:
    """Radar chart showing PSO improvement over baseline."""
    imp = comparison.improvements
    categories = ["Cost Savings", "Peak Reduction", "Emissions Reduction", "Solar Increase"]
    values = [
        max(0, imp.get("cost_improvement_pct", 0)),
        max(0, imp.get("peak_reduction_pct", 0)),
        max(0, imp.get("emissions_reduction_pct", 0)),
        max(0, imp.get("solar_improvement_pct", 0)),
    ]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        name="PSO Improvement (%)",
        line_color=COLORS["pso"],
        fillcolor="rgba(255,140,0,0.25)",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(max(values) + 5, 10)]),
            bgcolor="#0e1117",
        ),
        title="PSO Improvement Over Baseline (%)",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=380,
        margin=dict(l=60, r=60, t=60, b=40),
    )
    return fig
