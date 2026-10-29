# GridSwarm Testing Report

> This document will be updated with actual test results after running `pytest tests/ -v`.

## Test Environment

- OS: macOS (Apple Silicon / Intel)
- Python: 3.11+
- Dependencies: see requirements.txt

## Test Command

```bash
cd GridSwarm
pip install -r requirements.txt
pytest tests/ -v --tb=short 2>&1 | tee docs/test_output.txt
```

## Test Modules

| Module | Tests | Description |
|---|---|---|
| test_energy_balance.py | 4 | Grid import, solar accounting, balance with battery |
| test_constraints.py | 12 | EV availability, power limits, battery limits, infeasibility detection |
| test_pso.py | 7 | Particle bounds, convergence monotonicity, reproducibility, fitness |
| test_baseline.py | 8 | Baseline validity, EV windows, solar limits |
| test_evaluation.py | 9 | Metrics, safe comparison, CSV export structure |

## Known Limitations

- Dashboard startup tests require Streamlit runtime (skipped in headless CI)
- Runtime tests may vary by hardware (PSO timing)

## Actual Results

> Run `pytest tests/ -v` to generate actual results. Results will appear here.
