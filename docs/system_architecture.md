# System Architecture — GridSwarm

## High-Level Data Flow

```
User Input (Streamlit)
        │
        ▼
  SystemConfig  ─────────────────────────────────────────────────────┐
  (config.py)                                                         │
        │                                                             │
        ├──────────────────────────────┐                              │
        ▼                              ▼                              │
 DataGenerator              BaselineScheduler                        │
 (data_generator.py)        (baseline_scheduler.py)                  │
  · Solar profile                │                                    │
  · Tariff profile               │                                    │
  · Emissions profile            ▼                                    │
                           EnergyModel.decode()                       │
                           (energy_model.py)                          │
                                 │                                    │
                                 ▼                                    │
                          EnergySchedule ─────────────────────────── │
                                 │                                    │
                                 ▼                                    │
                          ConstraintChecker                          │
                          (constraints.py)                            │
                                 │                                    │
                                 ▼                                    │
                          ScheduleMetrics                             │
                          (evaluation.py)                             │
                                                                      │
                                                   PSOOptimizer ◄────┘
                                                   (pso_optimizer.py)
                                                         │
                                                         │ (particles → decode → evaluate)
                                                         ▼
                                                   ObjectiveFunction
                                                   (objectives.py)
                                                         │
                                                         ▼
                                                   PSOResult
                                                   → best_schedule
                                                         │
                                                         ▼
                                                   ComparisonResult
                                                   (evaluation.py)
                                                         │
                                             ┌───────────┘
                                             ▼
                              Visualization (visualization.py)
                              Plotly figures for all pages
                                             │
                                             ▼
                              Dashboard (app.py) → Streamlit UI
                                             │
                                             ▼
                              Export (export_utils.py)
                              CSV / JSON download
```

## Module Descriptions

| Module | Responsibility |
|---|---|
| `config.py` | EVConfig, BatteryConfig, GridConfig, SystemConfig dataclasses with validation |
| `data_generator.py` | Synthetic solar, tariff, emissions profiles; default/infeasible scenario builders |
| `energy_model.py` | EnergyModel: decode particles → EnergySchedule, apply repair, compute balance |
| `constraints.py` | ConstraintChecker: validate decoded schedules against all physical constraints |
| `objectives.py` | ObjectiveFunction: weighted-sum multi-objective fitness; normalization |
| `pso_optimizer.py` | PSOOptimizer: full PSO implementation with inertia decay, velocity clamping |
| `baseline_scheduler.py` | BaselineScheduler: rule-based EDF + solar priority schedule |
| `evaluation.py` | Evaluator: compute ScheduleMetrics, compare baseline vs PSO |
| `visualization.py` | Plotly chart builders for all dashboard pages |
| `export_utils.py` | schedule_to_dataframe, comparison_to_dataframe, config_to_dict, CSV/JSON export |
| `app.py` | 7-page Streamlit dashboard integrating all modules |

## Key Design Decisions

1. **Repair over penalty**: The EnergyModel repairs particles after each PSO step, ensuring all evaluated schedules are physically meaningful. This avoids fitness landscapes corrupted by large penalty terms.

2. **Residual grid import**: Grid import is never an optimization variable — it is computed as the residual after EVs, battery, and solar are allocated. This automatically enforces non-negativity (import ≥ 0).

3. **Modular and testable**: PSO has no Streamlit dependency. All modules can be imported and tested independently (see `tests/`).

4. **Signed battery convention**: A single signed value per interval for battery power prevents simultaneous charge/discharge by construction.

5. **Configurable objective weights**: Users can shift optimization focus between cost, peak, emissions, and solar without changing code.
