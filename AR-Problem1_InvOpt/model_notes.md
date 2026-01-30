## Inverse Optimization Model (Fan Vote Shares)

**Objective.** Infer weekly fan vote shares `s_{i,w}` that satisfy elimination rules. The rules alone do not identify `s`, so we add only priors that are defensible for *reconstruction* and state their bias explicitly.

### Data & preprocessing
- Source: `Data/2026_MCM_Problem_C_Data.csv`.
- Weekly judge total `J_{i,w}` = sum of `weekX_judgeY_score` (ignoring `N/A`).
- Active set `A_w`: contestants with `J_{i,w} > 0` in week `w`.
- Elimination week from `results` (“Withdrew” uses last active week).

### Decision variables
For each week `w` and active contestant `i`:  
`s_{i,w} ≥ 0`, `∑_{i∈A_w} s_{i,w} = 1`.

### Elimination rules (hard constraints)
Let `j_pct_{i,w} = J_{i,w} / ∑_{k∈A_w} J_{k,w}`.

1) **Percent regime (S3–S27)**  
`C_{i,w} = j_pct_{i,w} + s_{i,w}` with eliminated contestants at the minimum.

2) **Rank regime (S1–S2)**  
Combined rank `R_{i,w} = rJ_{i,w} + rF_{i,w}` with eliminated contestants worst.

3) **Bottom‑two regime (S28+)**  
Eliminated contestants must be members of the bottom‑two in `R_{i,w}`.

### Priors (used and justified)
- **Popularity prior (all regimes).**  
Fans vote on popularity more than technical skill. We estimate popularity as **survival beyond judge‑only expectation**.  
“Judge‑only expectation” = a per‑season linear model of elimination timing using **only** judge information:  
`expected_elim_week = a + b · avg_judge_score`, fitted across all contestants in that season.  
Then `pop_score = actual_elim_week − expected_elim_week` and softmax to `pop_w`.  
This uses future outcomes by design and is **not** predictive.

- **Temporal continuity (all regimes).**  
Popularity should not jump arbitrarily week‑to‑week, so we prefer solutions close to the previous week’s shares.

Implementation:
- **Percent seasons (QP):** use **popularity only** as the target, then add explicit smoothness:  
  minimize `‖s_w − pop_w‖² + ‖s_w − s_{w−1}‖²` subject to elimination constraints.  
  This avoids double‑counting smoothness.
- **Rank/Bottom seasons (MILP):** use a blended target because there is **no explicit smoothness term** in the MILP:  
  `target = α·previous_shares + (1−α)·popularity`, and minimize L1 distance to `target`.

### Optimization (why QP vs MILP)
- **Percent:** the objective and constraints are convex and continuous, so a QP gives a global optimum efficiently (cvxpy/OSQP).
- **Rank/Bottom:** rank ordering and bottom‑two membership are discrete; this is naturally modeled with binary variables, so we use a MILP (pulp/CBC). A fast greedy heuristic is tried first for bottom‑two weeks.

### Outputs
- `AR-Problem1/inferred_shares.csv`: `season, week, celebrity_name, s_map`
- `AR-Problem1/elimination_match.csv`: elimination consistency + rank diagnostics
- `AR-Problem1/inferred_shares_ensemble.csv`: per‑week mean/uncertainty (`s_mean`, `s_std`, `s_p10`, `s_p90`)
- `AR-Problem1/elimination_match_ensemble.csv`: mean/std match rates across ensemble runs

### Assumptions & critical notes
- **Fan‑vote ties are possible but rare.** Shares are continuous; ties can occur but are not enforced or prevented.
- **No‑elimination weeks** provide no elimination information; shares follow the priors.
- **Bottom‑two seasons** omit judge‑save details (data unavailable); identifiability is weak.
- **Popularity prior uses future outcomes.** Acceptable for reconstruction only; biases shares toward final placement.
- **Rank/bottom seasons are under‑identified.** Priors dominate when constraints are weak.
- **Solver fallback.** If a weekly MILP is infeasible, we fall back to the blended target (never uniform).

### Uncertainty (ensemble)
We estimate uncertainty by repeatedly solving the model with small random perturbations to the popularity prior, then reporting variability across solutions. This yields contestant/week‑level uncertainty summaries (std and quantiles) and a distribution of elimination match rates.

### Hyperparameters (default)
`λ_smooth = 1`, `λ_pop = 1`, `ε_share = 1e-6`,  
`α = 0.85`, `γ_pop = 0.7` (see `InvOpt.py`).
