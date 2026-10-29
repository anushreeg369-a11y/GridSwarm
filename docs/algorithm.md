# Algorithm Documentation — GridSwarm PSO

## Overview

GridSwarm uses **Particle Swarm Optimization (PSO)** as its primary bio-inspired algorithm.

---

## Biological Inspiration

PSO was introduced by Kennedy & Eberhart (1995), inspired by:
- **Bird flocking**: individuals adjust direction based on neighbours' positions
- **Fish schooling**: collective memory of the best food location found by any member

Each "particle" is an energy schedule. The swarm collectively discovers better schedules.

---

## Mathematical Formulation

### Velocity Update
```
v(i,t+1) = w(t) · v(i,t)
           + c1 · r1(t) · [pbest(i) − x(i,t)]    (cognitive)
           + c2 · r2(t) · [gbest    − x(i,t)]    (social)
```

### Position Update
```
x(i,t+1) = x(i,t) + v(i,t+1)
```

### Inertia Weight (Linear Decay)
```
w(t) = w_max − (w_max − w_min) × t / T_max
```
Default: w_max = 0.9, w_min = 0.4

### Symbols
| Symbol | Meaning |
|---|---|
| x(i,t) | Position of particle i at iteration t (= candidate schedule) |
| v(i,t) | Velocity of particle i at iteration t |
| pbest(i) | Best position found by particle i |
| gbest | Best position found by the entire swarm |
| w | Inertia weight |
| c1, c2 | Cognitive and social coefficients (default 2.05 each) |
| r1, r2 | Uniform random ∈ [0,1], independent each iteration |

---

## Particle Representation

Particle vector length = n_EVs × T + T:

```
x = [p_EV0_t0, p_EV0_t1, ..., p_EV0_T-1,
     p_EV1_t0, ..., p_EVn_T-1,
     p_bat_t0, p_bat_t1, ..., p_bat_T-1]
```

- EV elements: charging power in kW, bounded ∈ [0, max_charge_kW_per_EV]
- Battery elements: signed power in kW, bounded ∈ [-max_discharge, +max_charge]

---

## Fitness Function

### Multi-Objective Weighted Sum

```
J = w_cost     × (Total_Cost  / Max_Cost)
  + w_peak     × (Peak_Demand / Max_Peak)
  + w_emissions × (Emissions  / Max_Emissions)
  + w_solar    × (1 − Solar_Utilization)
```

Lower J → better schedule.

### Normalization Bounds
- Max_Cost = max_tariff × grid_capacity × T × Δt
- Max_Peak = grid_capacity
- Max_Emissions = max_emissions_intensity × grid_capacity × T × Δt

---

## Constraint Handling (Feasibility Repair)

After each position update, the particle is **decoded and repaired**:

1. Zero EV charging power outside availability window [arrival, departure]
2. Clip EV power to [0, max_charge_power]
3. Scale or boost EV energy delivery to match target SOC
4. Clip battery power to rated limits
5. Simulate battery SoC trajectory; reduce charge/discharge if limits violated
6. Compute grid import residually (= demand − solar − battery_discharge)
7. Clip grid import to ≥ 0 (no export modelled)

---

## Convergence Pseudocode

```
Initialize n_particles positions in [lb, ub]
Initialize velocities in [-span/2, span/2]
Evaluate all positions, set pbest and gbest

For iter = 1..max_iterations:
    w = w_max - (w_max - w_min) * iter / max_iterations
    For each particle i:
        r1, r2 ~ U(0,1)
        v_new = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        v_new = clip(v_new, -v_max, v_max)  # velocity clamping
        x_new = clip(x + v_new, lb, ub)    # position bounding
        schedule = decode_and_repair(x_new)
        fitness = evaluate_objectives(schedule)
        if fitness < pbest_fitness[i]:
            pbest[i] = x_new, pbest_fitness[i] = fitness
        if fitness < gbest_fitness:
            gbest = x_new, gbest_fitness = fitness
    Record gbest_fitness in convergence_history

Return decode(gbest) as optimal schedule
```

---

## Limitations

- Weighted-sum approach: different weight choices yield different solutions; no Pareto front
- Repair strategy may bias solutions toward feasibility boundaries
- Convergence not guaranteed to global optimum (heuristic)
- Fixed 24-hour horizon; no multi-day rolling optimization
