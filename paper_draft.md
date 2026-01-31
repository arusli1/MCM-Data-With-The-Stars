3 Problem 1: Estimate Fan Vote Shares
=====================================

3.1 Problem framing and voting constraints
------------------------------------------
Assumptions:
1) Judge scores are exogenous performance signals.
2) Fan-share evolves multiplicatively with relative performance.
3) Eliminations follow the weekly count inferred from score availability.
4) Tied placements are interchangeable for weekly matching.
5) One judge influence parameter applies per season.

Goal and challenge:
We infer weekly fan vote shares s_{i,w} that explain eliminations and
placements. The key difficulty is identifiability: elimination rules do not
uniquely determine s_{i,w}. Many vote-share vectors satisfy the same
elimination outcomes, especially in rank-based regimes where only ordering
matters. Thus elimination match can be 100% while vote estimates remain
uncertain.

Constraints by voting regime:
All regimes: s_{i,w} >= 0 and sum_{i in A_w} s_{i,w} = 1.

Percent regime (Seasons 3–27):
  C_{i,w} = j_pct_{i,w} + s_{i,w}
  For each eliminated e and survivor j:
  C_{e,w} <= C_{j,w} - epsilon

Rank regime (Seasons 1–2):
  rJ_{i,w}: descending rank of J_{i,w}
  rF_{i,w}: descending rank of s_{i,w}
  R_{i,w} = rJ_{i,w} + rF_{i,w}
  Eliminated set has the largest R_{i,w}

Bottom-two regime (Seasons 28+):
  Same R_{i,w}
  If k=1: eliminated is in bottom two of R_{i,w} with lower J_{i,w};
  if k>1: eliminated are the bottom-k by R_{i,w}

Constraint equations and regime summaries:
- AR-Problem1-Base/final_results/constraint_equations.md
- AR-Problem1-Base/final_results/constraints_regime_summary.csv

Uncertainty analysis (constraint + bootstrap):
We sample s on the simplex and retain samples satisfying the constraints (with
small slack). This yields feasible-set uncertainty bands without solving the
inverse optimization model. Example figure:
- AR-Problem1-Base/final_results/constraints_uncertainty_p90_season_27.png

Outputs:
- constraints_shares_uncertainty.csv (p10/p50/p90 by week)
- constraints_regime_summary.csv (regime-level accept rates and margins)

Bootstrap uncertainty (LPSSM trajectories):
- base_inferred_shares_uncertainty.csv provides p10/p50/p90 bands.

3.2 LPSSM design and calibration rationale
------------------------------------------
We model fan-share as a latent simplex-valued state that evolves with weekly
performance. A log-linear update preserves positivity and captures relative
performance effects without requiring explicit vote totals:
  Jz = zscore(J_week among active)
  s <- s * exp(alpha * Jz * JUDGE_SCALE)
  s <- s / sum(s); clip s_i to [MIN_SHARE, MAX_SHARE] then renormalize

Motivations:
- Multiplicative updates align with proportional vote dynamics and preserve
  the simplex constraint.
- JUDGE_SCALE and alpha isolate judge influence from fan momentum.
- Share bounds prevent unrealistic dominance and stabilize optimization.
- The Dirichlet s0 prior anchors baseline popularity when external signals
  are unavailable.
- Regime-specific elimination rules reflect historical voting formats.

Initialization:
  s0 ~ Dirichlet((1 + S0_PRIOR_SCALE * normalized(1 / placement)) * S0_PRIOR_CONC)

Objective (per season):
  RANK_MSE_WEIGHT * mean_sq_rank_diff
  + WEEKLY_ELIM_WEIGHT * (1 - weekly_elim_match_rate)
  - ENTROPY_WEIGHT * s0_entropy
  + ALPHA_PENALTY * alpha^2

3.3 Results: consistency and uncertainty
----------------------------------------
Consistency measures:
- weekly_elim_match_rate (fraction of weeks with correct elimination sets)
- mean_sq_rank_diff (placement rank MSE)
- elim_week_mae (mean absolute elimination-week error)

Files and figure:
- base_metrics.csv, base_overall_metrics.csv, base_placement_orders.csv
- AR-Problem1-Base/final_results/consistency_by_season.png

Uncertainty measures:
- Bootstrap LPSSM bands: base_inferred_shares_uncertainty.csv (p10/p50/p90)
- Constraint feasibility bands: constraints_shares_uncertainty.csv (p10/p50/p90)

Estimated fan vote outputs:
- AR-Problem1-Base/final_results/base_inferred_shares.csv
- Data/estimate_votes.csv (copy)

3.4 Interpretation and limitations
----------------------------------
The LPSSM yields consistent eliminations while producing reasonable share
trajectories. Constraint-based feasibility shows that, especially in rank
regimes, many s vectors satisfy the same eliminations; hence, uncertainty
remains high even when eliminations are perfectly matched.
