# GridSwarm — Smart Energy Management

**GridSwarm** is an academic simulation project demonstrating bio-inspired optimization (Particle Swarm Optimization) applied to smart energy management for Electric Vehicles, Solar PV, Battery Storage, and Grid coordination.

> **Note:** This is a simulation using synthetic data. It does not control real vehicles, batteries, or electrical infrastructure.

---

## Abstract

GridSwarm applies Particle Swarm Optimization (PSO) — a bio-inspired algorithm modelling collective animal behaviour — to schedule EV charging, battery storage, and grid imports across a 24-hour horizon. The optimizer simultaneously minimizes electricity cost, peak demand, carbon emissions, and maximizes solar self-consumption under realistic physical constraints.

---

## Problem Statement

Modern energy systems face complex multi-objective optimization: EVs must charge before departure, solar generation is intermittent, battery storage has physical limits, and electricity prices vary hourly. A naive greedy strategy wastes renewable energy and incurs unnecessary costs. PSO provides an efficient population-based search over the high-dimensional feasible schedule space.

---

## Objectives

1. Minimize total electricity cost (€/day)  
2. Minimize peak grid demand (kW)  
3. Minimize grid-related carbon emissions (kg CO₂)  
4. Maximize solar self-consumption (%)

---

## Software Requirements

- Python 3.9+
- NumPy, Pandas, Plotly, Streamlit, SciPy (optional), Pytest

## Installation

```bash
cd GridSwarm
pip install -r requirements.txt
```

## Running the Dashboard

```bash
streamlit run app.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
GridSwarm/
├── app.py                    # Streamlit dashboard entry point
├── requirements.txt
├── README.md
├── src/
│   ├── config.py             # System configuration dataclasses
│   ├── data_generator.py     # Synthetic profiles (solar, tariff, emissions)
│   ├── energy_model.py       # Energy balance and schedule decoding
│   ├── constraints.py        # Physical constraint checking
│   ├── objectives.py         # Multi-objective weighted-sum function
│   ├── pso_optimizer.py      # PSO algorithm implementation
│   ├── baseline_scheduler.py # Rule-based baseline scheduler
│   ├── evaluation.py         # Metrics and comparison
│   ├── visualization.py      # Plotly chart builders
│   └── export_utils.py       # CSV/JSON export utilities
├── tests/                    # Automated test suite
├── docs/
│   ├── algorithm.md
│   ├── system_architecture.md
│   └── testing_report.md
├── data/
└── outputs/
```

## Bio-Inspired Algorithm: PSO

Particle Swarm Optimization is inspired by flocking birds and schooling fish. Each "particle" represents a candidate energy schedule. Particles move through the solution space guided by their own best-found position (cognitive) and the swarm's global best (social).

**Update equations:**
```
v(t+1) = w·v(t) + c1·r1·(pbest - x(t)) + c2·r2·(gbest - x(t))
x(t+1) = x(t) + v(t+1)
```

See `docs/algorithm.md` for full mathematical formulation.

## Limitations

- Synthetic data only; no real-world API integration
- Weighted-sum multi-objective (not Pareto-front)
- No vehicle-to-grid (V2G) discharge
- Deterministic solar profile (no forecast uncertainty in base version)
- Single building / microgrid scope

## References

1. Kennedy, J. & Eberhart, R. (1995). Particle swarm optimization. *ICNN*.
2. Clerc, M. & Kennedy, J. (2002). The particle swarm—explosion, stability, and convergence. *IEEE TEC*.
3. Shi, Y. & Eberhart, R. (1998). A modified particle swarm optimizer. *CEC*.
4. Ipsakis, D. et al. (2009). Power management strategies for autonomous power systems. *Energy*.
