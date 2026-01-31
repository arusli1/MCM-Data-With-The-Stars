# Wikipedia fame (preseason edits) vs outcomes (placement)
## Dataset join quality
- Total contestant-season rows (DWTS outcomes): **421**
- Rows with wiki preseason `status=ok` + numeric edits: **392**
- Rows missing wiki edits (dropped from fame analyses): **29**

## Correlations (lower placement is better)
| Metric | r | p | n |
|---|---:|---:|---:|
| Pearson(log_edits, judge_avg_week_total) | -0.083 | 0.1 | 392 |
| Pearson(placement_pct, judge_avg_week_total) | -0.702 | 2e-59 | 392 |
| Pearson(placement_pct, log_edits) | 0.123 | 0.015 | 392 |
| Spearman(log_edits, judge_avg_week_total) | -0.080 | 0.11 | 392 |
| Spearman(placement, judge_avg_week_total) | -0.658 | 4.7e-50 | 392 |
| Spearman(placement, log_edits) | 0.194 | 0.00011 | 392 |
| Spearman(placement_pct, judge_avg_week_total) | -0.709 | 4.7e-61 | 392 |
| Spearman(placement_pct, log_edits) | 0.148 | 0.0034 | 392 |

## Partial correlation: fame vs placement controlling for judge score
- Using residualization with control `judge_avg_week_total` (HC0 OLS residuals).
- Pearson r (residuals): **0.091** (p=0.072, n=392)
- Spearman r (residuals): **0.105** (p=0.037, n=392)

## Regression (OLS): placement ~ judge + fame
| term | coef | se | t | p | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| Intercept | 15.897412 | 0.8458 | 18.795707 | 0.0 | 14.2345 | 17.560324 |
| judge_avg_week_total | -0.440445 | 0.026123 | -16.860171 | 0.0 | -0.491805 | -0.389084 |
| log_edits | 0.241147 | 0.079043 | 3.050823 | 0.002439 | 0.085741 | 0.396552 |
- R²: **0.439** (n=392)

## Regression (OLS): placement_pct ~ judge + fame
| term | coef | se | t | p | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| Intercept | 1.389022 | 0.066948 | 20.747709 | 0.0 | 1.257396 | 1.520647 |
| judge_avg_week_total | -0.039899 | 0.002068 | -19.295765 | 0.0 | -0.043964 | -0.035834 |
| log_edits | 0.01127 | 0.006257 | 1.801339 | 0.072424 | -0.001031 | 0.023571 |
- R²: **0.497** (n=392)

## Plots
- `/Users/archittamhane/Documents/Projects/MCM/2026 Competition/Repository/MCM-Data-With-The-Stars/Archit_P1/outputs/plots/binned_mean_placementpct_by_fame_quintile.png`
- `/Users/archittamhane/Documents/Projects/MCM/2026 Competition/Repository/MCM-Data-With-The-Stars/Archit_P1/outputs/plots/binned_mean_placementpct_by_judge_quintile.png`
- `/Users/archittamhane/Documents/Projects/MCM/2026 Competition/Repository/MCM-Data-With-The-Stars/Archit_P1/outputs/plots/scatter_placementpct_vs_judge_hue_logedits.png`
- `/Users/archittamhane/Documents/Projects/MCM/2026 Competition/Repository/MCM-Data-With-The-Stars/Archit_P1/outputs/plots/scatter_placementpct_vs_logedits_hue_judge.png`

## Notes / caveats
- This is **observational** and not causal: judges’ scores and fan responses co-evolve through the season.
- `judge_avg_week_total` uses only weeks with positive totals (to avoid the post-elimination 0 padding).
- Some wiki rows are missing due to disambiguation / missing titles; those contestants are excluded from fame analyses.
