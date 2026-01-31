# Google Trends (pytrends) fame vs outcomes (placement)

## Data
- Total contestant-season rows (DWTS outcomes): **421**
- Rows with pytrends fame (status=ok): **414**
- Fame window: **30** days before season premiere (per season)
- Anchor term: **Barack Obama** (used to stitch batches)

## Correlations (lower placement_pct is better)
| Metric | r | p | n |
|---|---:|---:|---:|
| Pearson(placement_pct, log_fame_ratio) | -0.070 | 0.16 | 414 |
| Spearman(log_fame_ratio, judge_avg_week_total) | 0.072 | 0.15 | 414 |
| Spearman(placement_pct, judge_avg_week_total) | -0.722 | 9.1e-68 | 414 |
| Spearman(placement_pct, log_fame_ratio) | -0.073 | 0.14 | 414 |

## Partial correlation controlling for judge score
- Spearman residual r: **-0.033** (p=0.5, n=414)
- Pearson residual r: **-0.095** (p=0.053, n=414)

## Regression (OLS): placement_pct ~ judge + fame
| term | coef | se | t | p | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| Intercept | 1.503726 | 0.049481 | 30.389801 | 0.0 | 1.406458 | 1.600994 |
| judge_avg_week_total | -0.040928 | 0.001978 | -20.687059 | 0.0 | -0.044818 | -0.037039 |
| log_fame_ratio | -0.06464 | 0.033418 | -1.934272 | 0.053766 | -0.130331 | 0.001052 |

## Plots
- `/Users/archittamhane/Documents/Projects/MCM/2026 Competition/Repository/MCM-Data-With-The-Stars/Archit_P1/pytrends_fame/outputs/plots/binned_mean_placementpct_by_pytrends_fame_quintile.png`
- `/Users/archittamhane/Documents/Projects/MCM/2026 Competition/Repository/MCM-Data-With-The-Stars/Archit_P1/pytrends_fame/outputs/plots/scatter_placementpct_vs_judge_hue_pytrends_fame.png`
- `/Users/archittamhane/Documents/Projects/MCM/2026 Competition/Repository/MCM-Data-With-The-Stars/Archit_P1/pytrends_fame/outputs/plots/scatter_placementpct_vs_pytrends_fame_hue_judge.png`

## Notes / caveats
- Google Trends values are *relative within each request*. The anchor-ratio helps comparability, but it is still noisy.
- Names that are ambiguous (e.g., common names) may yield polluted Trends signals. A future improvement is to query with disambiguating context.
- This is observational and not causal; judge scores and public attention co-evolve.
