Paper draft (MCM-style)
=======================

Summary Sheet
-------------
We propose a Latent Popularity State-Space Model (LPSSM): a deterministic
state-space model with multiplicative (replicator-style) updates and
rule-based eliminations. The model simulates weekly eliminations under
season-specific voting regimes and infers an initial fan distribution (s0)
plus a judge influence parameter (alpha) that best reproduce observed
placements. We calibrate the model by minimizing a placement-focused loss
with a weekly-elimination consistency term and a mild entropy regularizer on
s0. Uncertainty is quantified via bootstrap runs; sensitivity is evaluated
over judge-influence scale. The model produces season-level metrics, full
placement predictions, and inferred weekly fan-share trajectories.

1. Problem Restatement
----------------------
Given historical weekly judge scores and final placements, infer a dynamic
model of voting that explains elimination order and final ranking. The model
must produce realistic fan-share dynamics, capture season-specific voting
rules, and quantify uncertainty of the inferred parameters.

2. Data and Preprocessing
-------------------------
- Data source: Data/2026_MCM_Problem_C_Data.csv.
- Weekly judge totals: sum of four judge scores per week.
- Weekly elimination schedule: inferred from judge-score availability.
- True elimination week per contestant: derived from placement order aligned
  to the weekly schedule (tie-aware).
- Missing placements: none in this dataset.

3. Notation
-----------
- N: number of contestants in a season.
- W: number of weeks with scores.
- J_{w,i}: total judges' score for contestant i in week w.
- s_i: fan-share for contestant i.
- s0: initial fan-share vector.
- alpha: judge influence parameter.
- JUDGE_SCALE: scaling factor for judge influence.
- MAX_SHARE: cap on individual fan-share within a week.

4. Model Assumptions
--------------------
1) Judges' scores are exogenous and reflect performance quality.
2) Fan-share evolves multiplicatively with relative performance.
3) Eliminations follow observed weekly counts from the data schedule.
4) Tied placements are interchangeable for weekly matching.
5) A single alpha applies across the season.

5. Model Formulation
--------------------
Initialization:
  s = s0, s0 ~ Dirichlet((1 + S0_PRIOR_SCALE * normalized(1 / placement)) * S0_PRIOR_CONC)
  (simplex prior weakly anchored to final placements as a proxy for baseline
  popularity when external data are unavailable; S0_PRIOR_CONC controls strength)

Weekly update for active contestants:
  Jz = zscore(J_week among active)
  s <- s * exp(alpha * Jz * JUDGE_SCALE)  (log-linear state transition)
  s <- s / sum(s); clip s_i to [MIN_SHARE, MAX_SHARE] then renormalize

Elimination regimes:
- Seasons 1-2: rank-based combination of judge and fan ranks.
- Seasons 3-27: percent-based combination (judge_pct + s).
- Seasons 28+: bottom-two with judge tie-break (rank_bottom2).

Final placements:
  By elimination week; finalists ordered by final-week combined score.

6. Objective and Calibration
----------------------------
Per season objective:
  RANK_MSE_WEIGHT * mean_sq_rank_diff
  + WEEKLY_ELIM_WEIGHT * (1 - weekly_elim_match_rate)
  - ENTROPY_WEIGHT * s0_entropy
  + ALPHA_PENALTY * alpha^2

Optimization:
- Grid search over alpha (linspace(-0.2, 1.5, 25)).
- Sample multiple s0 vectors per alpha.
- Local refinement around the best s0.
- Bootstrap across random seeds; select best run.

Current hyperparameters:
  RANK_MSE_WEIGHT = 1.0
  WEEKLY_ELIM_WEIGHT = 3.0
  ENTROPY_WEIGHT = 0.02
  ALPHA_PENALTY = 0.0
  S0_PRIOR_SCALE = 6.0
  S0_PRIOR_CONC = 5.0
  JUDGE_SCALE = 0.5
  MIN_SHARE = 0.01
  MAX_SHARE = 0.8

7. Results
----------
Outputs include:
- base_results/base_metrics.csv (season metrics + uncertainty bands)
- base_results/base_placement_orders.csv (true vs predicted placements)
- base_results/base_inferred_shares.csv (weekly fan-share trajectories)
- base_results/base_s0.csv (optimized s0)
- base_results/base_truth_elim_weeks.csv (placement-derived truth)
- base_results/base_bootstrap_results.csv (bootstrap runs)
- base_results/base_overall_metrics.csv (overall model score)
- base_results/base_inferred_shares_summary.csv (share distribution summary)

Key performance indicators:
- mean_sq_rank_diff
- weekly_elim_match_rate
- elim_week_mae
- alpha distribution across bootstrap runs

8. Uncertainty Quantification
-----------------------------
Bootstrap runs provide:
- Mean and standard deviation of each metric per season.
- Percentile bands (p10/p50/p90) for alpha, rank error, and elimination match.

9. Sensitivity Analysis
-----------------------
Optional sensitivity grid:
- JUDGE_SCALE in {0.3, 0.5, 0.7}
- ALPHA_PENALTY in {0.0, 0.2, 0.5}
- One-at-a-time sweep: judge_scale, alpha_penalty, entropy_weight,
  s0_prior_scale, s0_prior_conc

Run with:
  SENSITIVITY_MODE=1 python3 -u AR-Problem1-Base/base.py
  OAT_SENSITIVITY_MODE=1 python3 -u AR-Problem1-Base/base.py

10. Limitations
---------------
- Fan-share dynamics are stylized; no explicit external popularity signals.
- Alpha is fixed per season; no temporal drift.
- Judge influence modeled via z-scores only.
- Elimination schedule inferred from score availability.
- Share cap and entropy regularization are heuristic.

11. Justification of Model Choices (brief)
------------------------------------------
- Rank MSE emphasizes large placement errors more than small swaps.
- Weekly elimination match ensures the weekly structure is respected.
- Entropy on s0 avoids degenerate, single-contestant dominance.
- MAX_SHARE captures realistic caps on vote share.
- Placement-based prior anchors s0 to baseline popularity using final rank
  as a weak proxy when external popularity data are unavailable.

12. Conclusions
---------------
The model integrates performance and popularity into a single dynamical
framework that explains elimination order and final placements. With a small
set of interpretable parameters, it yields realistic fan-share trajectories,
supports uncertainty quantification, and exposes sensitivity to judge
influence. The framework can be extended with external popularity data or
time-varying alpha to improve realism.

Appendix: Reproducibility
-------------------------
Run full model:
  python3 -u AR-Problem1-Base/base.py

Fast mode:
  FAST_MODE=1 python3 -u AR-Problem1-Base/base.py

Sensitivity mode:
  SENSITIVITY_MODE=1 python3 -u AR-Problem1-Base/base.py
