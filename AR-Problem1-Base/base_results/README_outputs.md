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

Model bounds and settings
-------------------------
- base_hyperparams.json
  - Includes min_share=0.01 and max_share=0.8 bounds for realistic vote shares.
