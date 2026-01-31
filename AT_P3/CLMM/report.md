# DWTS Cumulative Link Mixed Model (CLMM) Report

## Data cleaning & construction

- Source rows (raw): **421**
- Contestant-season rows (df_cs): **421**
- Seasons present: **34**
- Unique contestants: **408**
- Missing `placement_final` after parsing: **0**

Placement source breakdown (`placement_audit.csv`):

- placement_col: 421

## Model specification

**Primary model (no season):**

- Outcome: `placement_final` as ordered factor (1 is best).
- Link: logit.
- Fixed effects: standardized age + industry factor.
- Random intercepts: partner, homestate, homecountry/region.

**Sensitivity model:** same, plus `(1 | season)` random intercept.

## Cross-validation (5-fold)

We report two prediction modes:
- **Marginal (population-level)**: random effects set to 0 (robust for unseen group levels).
- **Conditional (BLUP-assisted)**: adds estimated random intercepts for partner/state/country when that level was seen in the training fold; unseen levels fall back to 0.

Summary (mean ± sd across folds):

Marginal:
- Exact accuracy: **0.083 ± 0.018**
- Within-1 accuracy: **0.314 ± 0.061**
- MAE (expected placement): **2.845 ± 0.079**
- Spearman(expected, true): **0.418 ± 0.091**

Conditional (BLUP-assisted):
- Exact accuracy: **0.085 ± 0.030**
- Within-1 accuracy: **0.295 ± 0.046**
- MAE (expected placement): **2.794 ± 0.087**
- Spearman(expected, true): **0.443 ± 0.103**

## Fixed effects (primary model)

See `fixed_effects.csv` for full table (log-odds scale, OR with 95% CI).

- Age (z-scored): estimate=0.922, OR=2.513, p=2.83e-18

### Coefficient direction sanity check (computed)

For `clmm` (logit link), the model uses \(\text{logit}(P(Y \le k)) = \theta_k - \eta\).
With `placement_ord` ordered as 1 < 2 < 3 < ... (1 is best), a **positive** coefficient increases \(\eta\) and therefore **decreases** \(P(Y \le k)\), i.e., shifts mass toward worse placements.

- Holding industry/partner/state/country at baseline and using fixed-effects-only probabilities:
  - P(finish top-3) at age_z = -1: **0.488**
  - P(finish top-3) at age_z = +1: **0.131**

## Sensitivity: adding `(1 | season)`

See `fixed_effects_sensitivity.csv` for the full sensitivity fixed-effects table.

- Age (z-scored) sensitivity: estimate=0.922, OR=2.513, p=2.83e-18

Interpretation: if the age/industry effects keep the **same sign and similar magnitude** under the sensitivity model,
the primary conclusions are not driven purely by between-season differences.

## Random effects

See `random_effects_summary.csv` for variance/SD by grouping factor.


### Ballroom partner BLUPs (primary model; interpret cautiously)

**Top 10 (most negative random intercept; tends to better placements):**

- Derek Hough: -0.7920
- Cheryl Burke: -0.3169
- Lindsay Arnold: -0.1973
- Daniella Karagach: -0.1955
- Corky Ballas: -0.1860
- Kym Johnson: -0.1809
- Valentin Chmerkovskiy: -0.1474
- Julianne Hough: -0.1301
- Witney Carson: -0.1205
- Charlotte Jorgensen: -0.1157

**Bottom 10 (most positive random intercept; tends to worse placements):**

- Keo Motsepe: 0.3631
- Gleb Savchenko: 0.3526
- Pasha Pashkov: 0.3182
- Brandon Armstrong: 0.2854
- Britt Stewart: 0.2715
- Chelsie Hightower: 0.2163
- Koko Iwasaki: 0.1294
- Ezra Sosa: 0.1276
- Peta Murgatroyd: 0.1271
- Anna Trebunskaya: 0.1251

## Limitations

- Ordinal proportional-odds assumption may be violated (effect sizes assumed constant across thresholds).
- Cross-validation uses fixed-effects-only probabilities for robustness to unseen random-effect levels in test folds.
- Some random-effect levels are sparse; variance estimates and BLUP rankings are shrinkage-regularized and should be interpreted cautiously.
- Placement reconstruction from `results` is implemented for completeness, but this dataset largely provides a numeric `placement` column.

