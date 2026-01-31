# Inverse Optimization (InvOpt) outcomes: consistency & uncertainty report

## Abstract
We evaluate the inverse-optimization reconstruction outputs in `AR-Problem1_InvOpt/`, quantifying (i) **consistency** with the stated elimination rules and (ii) **uncertainty** in reported evaluation metrics via bootstrap confidence intervals. Because the InvOpt model’s popularity prior uses future outcomes by design (see `model_notes.md`), these results should be interpreted as **reconstruction consistency**, not predictive accuracy.

## Data & outputs evaluated
- **DWTS source**: `Data/2026_MCM_Problem_C_Data.csv`
- **Inferred fan shares (MAP)**: `AR-Problem1_InvOpt/inferred_shares.csv` (`s_map`)
- **Season-level reported diagnostics**: `AR-Problem1_InvOpt/elimination_match.csv`

## Methods
### Active set and eliminations
For each season/week, the active set is contestants with positive weekly judge totals (J>0). True elimination week is parsed from the `results` field; withdrawals are treated as elimination at the last active week.

### Regime-specific risk ordering
- **Percent (S3–S27)**: C = j_pct + s. Lower C implies higher elimination risk.
- **Rank (S1–S2)**: combined rank R = r_J + r_F. Higher R implies higher risk.
- **Bottom-two (S28+)**: the eliminated must lie within the bottom-two by R (for single eliminations).

### Consistency metrics
- **Set match**: whether the true eliminated set is a subset of the model’s predicted worst set (mirrors the code’s logic).
- **Top-1 accuracy** (single-elimination weeks): whether the model’s single highest-risk contestant matches the eliminated.
- **Eliminated risk rank**: rank position of the true eliminated under the model’s risk ordering (1 = most at risk).

### Uncertainty metrics
The folder does not currently include the ensemble uncertainty CSVs referenced in `model_notes.md` (they would require `cvxpy` and repeated randomized solves). We therefore report uncertainty as **bootstrap 95% confidence intervals** over weeks for aggregate evaluation metrics, plus **per-week separation margins** (how close the decision boundary is) as a proxy for identifiability.

## Results
- Overall set-match rate (week-weighted): **0.952**
- Overall top-1 accuracy (single-elim weeks): **0.830**

### Regime-level summary (recomputed)
| regime | weeks | match_rate | top1_acc | mean_elim_rank | median_margin | mean_entropy_norm | mean_hhi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bottom | 73 | 0.8356 | 0.75 | 1.3304 | 3.0 | 0.9362 | 0.1491 |
| percent | 248 | 0.996 | 0.8593 | 1.0879 | 0.0072 | 0.916 | 0.1971 |
| rank | 14 | 0.7857 | 0.7 | 1.4 | 3.0 | 0.9299 | 0.2604 |

### Season-level summary (recomputed + reported diagnostics)
| season | regime | weeks | matched | match_rate | top1_acc | mean_elim_rank | median_margin | match_rate_reported | kendall_tau | mean_abs_rank_diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | rank | 6 | 4 | 0.6667 | 0.3333 | 1.6667 | 1.0 | 0.6667 | 0.8667 | 0.3333 |
| 2 | rank | 8 | 7 | 0.875 | 0.8571 | 1.2857 | 3.0 | 0.875 | 0.9556 | 0.2 |
| 3 | percent | 10 | 10 | 1.0 | 0.8571 | 1.0714 | 0.0 | 1.0 | 0.9273 | 0.3636 |
| 4 | percent | 10 | 10 | 1.0 | 1.0 | 1.0 | 0.0361 | 1.0 | 0.7455 | 1.0909 |
| 5 | percent | 10 | 10 | 1.0 | 1.0 | 1.0 | 0.0066 | 1.0 | 0.8485 | 0.8333 |
| 6 | percent | 10 | 10 | 1.0 | 0.875 | 1.0625 | 0.0109 | 1.0 | 0.8182 | 0.8333 |
| 7 | percent | 10 | 10 | 1.0 | 0.75 | 1.125 | 0.0104 | 1.0 | 0.8205 | 1.0769 |
| 8 | percent | 11 | 11 | 1.0 | 0.8889 | 1.0556 | 0.0107 | 1.0 | 0.7692 | 1.0769 |
| 9 | percent | 10 | 10 | 1.0 | 0.5556 | 1.2222 | 0.0113 | 1.0 | 0.8333 | 1.25 |
| 10 | percent | 10 | 10 | 1.0 | 1.0 | 1.0 | 0.0278 | 1.0 | 0.7455 | 1.0909 |
| 11 | percent | 10 | 10 | 1.0 | 1.0 | 1.0 | 0.0107 | 1.0 | 0.7879 | 0.8333 |
| 12 | percent | 10 | 10 | 1.0 | 1.0 | 1.0 | 0.0094 | 1.0 | 0.7091 | 1.2727 |
| 13 | percent | 10 | 10 | 1.0 | 1.0 | 1.0 | 0.0103 | 1.0 | 0.8182 | 1.0 |
| 14 | percent | 10 | 10 | 1.0 | 0.875 | 1.0625 | 0.0012 | 1.0 | 0.7273 | 1.3333 |
| 15 | percent | 10 | 10 | 1.0 | 0.5714 | 1.2143 | 0.0 | 1.0 | 0.8462 | 0.9231 |
| 16 | percent | 10 | 10 | 1.0 | 1.0 | 1.0 | 0.0213 | 1.0 | 0.6667 | 1.6667 |
| 17 | percent | 11 | 11 | 1.0 | 1.0 | 1.0 | 0.0092 | 1.0 | 0.7273 | 1.0 |
| 18 | percent | 10 | 9 | 0.9 | 0.8571 | 1.4286 | 0.0001 | 1.0 | 0.697 | 1.3333 |
| 19 | percent | 11 | 11 | 1.0 | 1.0 | 1.0 | 0.0019 | 1.0 | 0.7436 | 1.2308 |
| 20 | percent | 10 | 10 | 1.0 | 0.875 | 1.0625 | 0.0096 | 1.0 | 0.8182 | 1.0 |
| 21 | percent | 11 | 11 | 1.0 | 0.7143 | 1.2143 | 0.0009 | 1.0 | 0.6923 | 1.5385 |
| 22 | percent | 10 | 10 | 1.0 | 0.7143 | 1.1429 | 0.0 | 1.0 | 0.5758 | 1.8333 |
| 23 | percent | 11 | 11 | 1.0 | 0.8889 | 1.0556 | 0.023 | 1.0 | 0.6154 | 1.8462 |
| 24 | percent | 10 | 10 | 1.0 | 0.875 | 1.0625 | 0.0006 | 1.0 | 0.8788 | 0.6667 |
| 25 | percent | 10 | 10 | 1.0 | 0.75 | 1.125 | 0.0 | 1.0 | 0.6667 | 1.5385 |
| 26 | percent | 4 | 4 | 1.0 | 0.0 | 1.6667 | 0.0 | 1.0 | 0.6444 | 1.2 |
| 27 | percent | 9 | 9 | 1.0 | 0.7143 | 1.1429 | 0.0126 | 1.0 | 0.8974 | 0.6154 |
| 28 | bottom | 11 | 10 | 0.9091 | 0.7143 | 1.3571 | 4.0 | 0.9091 | 0.7273 | 1.1667 |
| 29 | bottom | 11 | 9 | 0.8182 | 0.7778 | 1.3333 | 3.0 | 0.8182 | 0.8857 | 0.8 |
| 30 | bottom | 10 | 7 | 0.7 | 0.625 | 1.5 | 3.0 | 0.7 | 0.8286 | 1.0667 |
| 31 | bottom | 10 | 7 | 0.7 | 0.6667 | 1.3889 | 3.0 | 0.7 | 0.8667 | 0.875 |
| 32 | bottom | 11 | 11 | 1.0 | 1.0 | 1.0 | 4.0 | 1.0 | 0.8681 | 0.8571 |
| 33 | bottom | 9 | 7 | 0.7778 | 0.5 | 1.6667 | 3.0 | 0.7778 | 0.641 | 1.6923 |
| 34 | bottom | 11 | 10 | 0.9091 | 0.875 | 1.1875 | 4.5 | 0.9091 | 0.8462 | 0.9286 |

### Bootstrap uncertainty (95% CI)
We bootstrap weeks within each slice (overall / per-regime) and report the sampling distribution of aggregate metrics.

| slice | metric | estimate | ci95_low | ci95_high | n |
| --- | --- | --- | --- | --- | --- |
| overall | match_rate | 0.95224 | 0.92836 | 0.97313 | 335 |
| overall | top1_acc | 0.83019 | 0.78491 | 0.87547 | 265 |
| overall | mean_elim_rank | 1.15094 | 1.10377 | 1.20189 | 265 |
| overall | median_margin | 0.97428 | 0.74833 | 1.20806 | 265 |
| regime:bottom | match_rate | 0.83562 | 0.75342 | 0.91781 | 73 |
| regime:bottom | top1_acc | 0.75 | 0.64286 | 0.85714 | 56 |
| regime:bottom | mean_elim_rank | 1.33036 | 1.18728 | 1.49107 | 56 |
| regime:bottom | median_margin | 4.10714 | 3.53571 | 4.69643 | 56 |
| regime:percent | match_rate | 0.99597 | 0.9879 | 1.0 | 248 |
| regime:percent | top1_acc | 0.8593 | 0.80905 | 0.90452 | 199 |
| regime:percent | mean_elim_rank | 1.08794 | 1.05528 | 1.13065 | 199 |
| regime:percent | median_margin | 0.02605 | 0.02047 | 0.03212 | 199 |
| regime:rank | match_rate | 0.78571 | 0.57143 | 1.0 | 14 |
| regime:rank | top1_acc | 0.7 | 0.4 | 1.0 | 10 |
| regime:rank | mean_elim_rank | 1.4 | 1.0 | 1.8 | 10 |
| regime:rank | median_margin | 2.3 | 1.4 | 3.1 | 10 |

### Plots
- `/Users/archittamhane/Documents/Projects/MCM/2026 Competition/Repository/MCM-Data-With-The-Stars/AR-Problem1_InvOpt/outputs/plots/elim_rank_by_regime.png`
- `/Users/archittamhane/Documents/Projects/MCM/2026 Competition/Repository/MCM-Data-With-The-Stars/AR-Problem1_InvOpt/outputs/plots/margin_by_regime_single_elim.png`

## Discussion
1) **Very high match rates in percent-era seasons are expected** because the optimization enforces the elimination rules as hard constraints, and the popularity prior is constructed using realized elimination timing.

2) **Bottom-two seasons show weaker guarantees** because the constraints are less informative (membership in a bottom-two set instead of a unique minimum) and discrete ranking introduces many equivalent solutions.

3) **Uncertainty is dominated by identifiability rather than sampling noise**: when margins are small, many nearby share vectors can satisfy the constraints. Ensemble-based uncertainty (randomized priors) would quantify this more directly.

## Reproducibility notes
- This analysis recomputes weekly metrics from `inferred_shares.csv` and the original DWTS data.
- If you want true solution uncertainty (std/quantiles of shares), run the ensemble pipeline (see `infer_votes_ensemble.py`) after installing `cvxpy` and `pulp`.
