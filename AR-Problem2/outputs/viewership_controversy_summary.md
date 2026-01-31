## Viewership vs. Controversy Analysis

**Data:** Wikipedia viewership, Problem 2b controversy. **Wikipedia-verified seasons only.**

**Controversy metric:** Mean |judge_percentile − placement_percentile| per contestant per season (continuous).

**Seasons:** 30
**Pearson r (mean controversy vs viewership):** 0.333 (p = 0.0721)
**Spearman ρ:** 0.327 (p = 0.0781)
**Partial r (controlling season):** -0.542 (p = 0.0024)
**Residual correlation:** -0.469 (p = 0.0090)
**Max controversy vs viewership:** r = 0.115 (p = 0.5440)

**Interpretation:** Raw correlation is positive (r ≈ 0.33) but confounded by time: early seasons had both higher viewership and different controversy levels. **Partial correlation (controlling season) is strongly negative (r ≈ -0.54, p ≈ 0.002)**: within-era, seasons with higher mean controversy tended to have *lower* viewership than expected.

**Limitations:** Viewership declines over time; small N.

**Figures:** viewership_controversy_scatter.pdf, viewership_controversy_by_season.pdf, viewership_controversy_residual.pdf, viewership_controversy_max.pdf