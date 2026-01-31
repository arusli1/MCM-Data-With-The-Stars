Base model outputs for Problem 1
================================

These files directly answer the fan-vote estimation, consistency, and certainty questions.

Estimated fan votes (weekly)
----------------------------
- base_inferred_shares.csv
  - Columns: season, week, celebrity_name, s_share (estimated fan vote share)

Consistency with eliminations
-----------------------------
- base_placement_orders.csv
  - True vs predicted elimination weeks and placements for each contestant.
- base_metrics.csv
  - Per-season consistency metrics (weekly_elim_match_rate, placement_match_rate,
    elim_week_mae, rank errors) plus bootstrap bands.
- base_overall_metrics.csv
  - Overall (all seasons) consistency metrics.

Uncertainty of fan-vote estimates
---------------------------------
- base_inferred_shares_uncertainty.csv
  - Per season/week/contestant uncertainty from bootstrap runs:
    s_share_mean, s_share_std, s_share_p10, s_share_p50, s_share_p90.
- base_bootstrap_results.csv
  - Bootstrap distribution of season-level metrics and alpha.

Constraint-based uncertainty (math-only)
----------------------------------------
- AR-Problem1-Constraints/outputs/constraint_equations.md
  - Regime-specific feasibility constraints (percent, rank, bottom-two).
- AR-Problem1-Constraints/outputs/constraints_shares_uncertainty.csv
  - Feasible-set sampling bands for each contestant/week.
- AR-Problem1-Constraints/outputs/constraints_week_summary.csv
  - Acceptance rates and margin summaries by week (identifiability proxy).
- AR-Problem1-Constraints/outputs/constraints_regime_summary.csv
  - Regime-level acceptance and margin summaries.

Model bounds and settings
-------------------------
- base_hyperparams.json
  - Includes min_share=0.01 and max_share=0.8 bounds for realistic vote shares.
