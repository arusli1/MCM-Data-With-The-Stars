Model summary (AR-Problem1-Base)
================================

Purpose and model name
----------------------
We use a Latent Popularity State-Space Model (LPSSM): a deterministic
state-space model with multiplicative (replicator-style) updates and
rule-based eliminations. The model searches for an initial fan distribution
(s0) and a judge influence parameter (alpha) that best reproduce observed
placements and weekly eliminations under season-specific voting regimes.

Data and preprocessing
----------------------
- Input: Data/2026_MCM_Problem_C_Data.csv
- Weekly judge totals: sum of four judge scores per week.
- True elimination schedule: inferred from judge-score availability
  (how many eliminations occur each week).
- True elimination week per contestant: derived from placement order and the
  weekly elimination schedule; falls back to judge-based elim week if
  placement is missing (not present in this dataset).
- Ties in placement are treated as interchangeable groups.

State and dynamics
------------------
Let s be the vector of fan-share weights for active contestants (a simplex-
constrained latent state). Weekly updates form a multiplicative state transition
driven by standardized judge performance (a log-linear influence model).
- Initialization: s = s0 (Dirichlet draw, optimized).
-- Weekly update:
  1) Compute judge z-scores among active contestants.
  2) Update fan shares:
     s <- s * exp(alpha * Jz * JUDGE_SCALE)
  3) Renormalize and clip: s_i in [MIN_SHARE, MAX_SHARE] then renormalize.
  4) Eliminate k contestants based on the season regime:
     - percent (seasons 3-27): lowest (judge_pct + s)
     - rank (seasons 1-2): highest combined rank from judges and s
     - rank_bottom2 (season 28+): bottom-two with judge tie-break
  5) Redistribute eliminated fan share evenly to remaining contestants.
- Final placements are derived from elimination week; finalists are ordered
  by final-week combined score.

Constraint-based uncertainty (math-only)
----------------------------------------
We separately analyze uncertainty using regime-specific feasibility constraints
without solving the inverse optimization model. For each week, we sample fan
shares on the simplex and retain those that satisfy the elimination constraints
within a small slack, producing uncertainty bands.

Regime constraints:
- Percent (Seasons 3-27): C_{i,w} = j_pct_{i,w} + s_{i,w} and
  C_{e,w} <= C_{j,w} - epsilon for eliminated e vs survivor j.
- Rank (Seasons 1-2): R_{i,w} = rJ_{i,w} + rF_{i,w}; eliminated are largest R.
- Bottom-two (Seasons 28+): eliminated must be in bottom-2 of R_{i,w} with lower
  judge score (or bottom-k for multi-elim weeks).

Determinism
-----------
Given s0 and alpha, the simulation is deterministic. Randomness enters only
through the search over candidate s0 and alpha (bootstrap seeds).

Objective (loss)
----------------
Minimize:
  RANK_MSE_WEIGHT * mean_sq_rank_diff
+ WEEKLY_ELIM_WEIGHT * (1 - weekly_elim_match_rate)
- ENTROPY_WEIGHT * s0_entropy
  + ALPHA_PENALTY * alpha^2

Where:
- mean_sq_rank_diff: mean squared error between predicted and true placement.
  When weekly elim match is perfect, placement is pinned by elimination order
  except for finalists (and same-week ties); so any rank MSE is from finalist
  ranking (and within-week ordering) only.
- weekly_elim_match_rate: fraction of weeks with correct elimination sets
  (tie-aware).
- s0_entropy: Shannon entropy of s0.
- alpha^2 penalty discourages overly strong judge influence.

Current weights / hyperparameters
---------------------------------
- RANK_MSE_WEIGHT = 1.0
- WEEKLY_ELIM_WEIGHT = 3.0
- ENTROPY_WEIGHT = 0.02
- ALPHA_PENALTY = 0.0
- JUDGE_SCALE = 0.5
- MIN_SHARE = 0.01
- MAX_SHARE = 0.8
- S0_PRIOR_SCALE = 6.0
- S0_PRIOR_CONC = 5.0
- alpha_grid = linspace(-0.2, 1.5, 25) (fast mode: -0.2..1.5, 13)
- N_S0_SAMPLES = 100 (fast mode: 60)
- REFINE_STEPS = 20 (fast mode: 15)
- BOOTSTRAP_RUNS = 6 (fast mode: 4)

Optimization procedure
----------------------
For each season:
1) Grid search over alpha.
2) For each alpha, sample multiple s0 via Dirichlet.
3) Keep best candidate by objective.
4) Local refinement of s0 around the best candidate.
5) Bootstrap runs over random seeds; select the best run by objective.

Uncertainty quantification
--------------------------
- Bootstrap variability across runs.
- Outputs per-season means, stds, and percentile bands in:
  AR-Problem1-Base/base_metrics.csv
- Full bootstrap runs in:
  AR-Problem1-Base/base_bootstrap_results.csv
- Constraint feasibility sampling outputs:
  AR-Problem1-Constraints/outputs/constraints_shares_uncertainty.csv

Sensitivity analysis
--------------------
Optional grid over:
- JUDGE_SCALE in {0.3, 0.5, 0.7}
- ALPHA_PENALTY in {0.0, 0.2, 0.5}
- One-at-a-time sweep: judge_scale, alpha_penalty, entropy_weight,
  s0_prior_scale, s0_prior_conc

Run with:
  SENSITIVITY_MODE=1 python3 -u AR-Problem1-Base/base.py
  OAT_SENSITIVITY_MODE=1 python3 -u AR-Problem1-Base/base.py
Outputs:
  AR-Problem1-Base/base_sensitivity.csv

Outputs
-------
- base_results/base_metrics.csv (season-level metrics + uncertainty bands)
- base_results/base_bootstrap_results.csv (per-bootstrap results)
- base_results/base_placement_orders.csv (true vs predicted placements)
- base_results/base_inferred_shares.csv (weekly s trajectories)
- base_results/base_s0.csv (optimized s0)
- base_results/base_truth_elim_weeks.csv (true elim weeks derived from placement)
- base_results/base_overall_metrics.csv (overall model score)
- base_results/base_inferred_shares_summary.csv (share distribution summary)

Assumptions
-----------
- Judges' scores are an exogenous performance signal.
- Fan-share is a latent proportion that evolves multiplicatively with
  performance and is redistributed upon elimination.
- Initial popularity prior is weakly informed by final placements (inverse-rank weighting),
  used as a proxy for baseline popularity when external popularity data are unavailable.
- Weekly elimination counts are correct and follow the data schedule.
- Tied placements are interchangeable for weekly matching.
- The voting regime is season-dependent and piecewise constant.

Limitations
-----------
- Fan-share dynamics are stylized (no explicit demographics or external data).
- Alpha is global per season (no weekly drift).
- Judge influence is modeled through z-scores only.
- Elimination schedule depends on judge-score availability, not broadcast rules.
- Max-share cap and entropy term are heuristic regularizers.

Justification of components (brief)
-----------------------------------
- Rank MSE: penalizes large placement errors more than small swaps.
- Weekly elimination match: enforces correct weekly structure from the data.
- Entropy term: prevents degenerate s0 (single-candidate dominance).
- MAX_SHARE: reflects realistic upper bounds on vote share.
- S0 prior (final placements): regularizes initial popularity toward the final ordering
  as a weak proxy for baseline fan support; the simulation can still override it.
