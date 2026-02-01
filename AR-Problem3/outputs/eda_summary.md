# Problem 3 EDA: Impact of Pro Dancers and Celebrity Characteristics

## Data
- Contestants: 421
- Seasons: 34
- With judge scores (W1-3): 421
- With fan vote estimates (W1-3): 421

## Outcomes
- **mean_judge_w1_3**: Mean judge total score across weeks 1–3
- **mean_fan_w1_3**: Mean estimated fan vote share across weeks 1–3
- **success_score**: 1 = winner, 0 = last place (within season)
- **placement**: Final placement (1 = winner)

## Age Effects
- Age vs judge score: r = -0.354, p < 0.0001
- Age vs fan share: r = -0.263, p < 0.0001
- Age vs success: r = -0.430, p < 0.0001

**Interpretation:** Younger celebrities tend to do better (higher judge scores, fan support, success). Age has a stronger effect on judge scores than fan support (|r| larger for judges).

## Industry Effects
Kruskal-Wallis: industry significantly affects judge scores (H≈24.2, p<0.01), fan share (H≈10.7), and success (H≈16.4).

## Pro Partner Effects
Kruskal-Wallis: pro partner significantly affects all outcomes (H≈24–45, p<0.01). Top partners (Derek Hough, Cheryl Burke, etc.) have consistently higher contestant success.

## Judge vs Fan: Same or Different? (See `judge_fan_impact.py` for full analysis)
- Correlation(age, judge) = -0.354; Correlation(age, fan) = -0.263
Age has stronger effect on judge scores than fan support. Industry and partner affect both; pro partner effect is larger for judge scores (Kruskal H≈45 vs H≈24 for fans).

**Bootstrap test (age effect judge vs fan):** |r| difference = 0.091, 95% CI [-0.014, 0.180], p = 0.0960. 
Difference not significant.

**Pro partner residualized boost:** Top: Derek Hough, Cheryl Burke, Kym Johnson. Celebs with these partners outperform age+industry expectation.

**Judge–fan agreement by subgroup:** Highest in industry_Comedian (r=0.54), industry_Athlete (r=0.51).

**Improvement (W1→W3):** r(age, judge_slope) = -0.079 (p = 0.115); r(age, fan_slope) = -0.201 (p < 0.0001).
Younger celebs show more fan improvement; age does not significantly predict judge improvement.

## Regression (OLS: outcome ~ age + industry + partner)

- mean_judge_w1_3: R² = 0.236, R²_adj = 0.195
- mean_fan_w1_3: R² = 0.151, R²_adj = 0.106
- success_score: R² = 0.280, R²_adj = 0.243

## Statistical Tests (Summary)
See `outputs/statistical_tests.csv` for full results.

## Figures
- `age_effects.pdf`: Age vs judge, fan, success
- `industry_effects.pdf`: Outcomes by industry
- `partner_effects.pdf`: Outcomes by pro partner
- `judge_vs_fan_comparison.pdf`: Judge vs fan by factor
- `pro_partner_ranking.pdf`: Pro partner mean success
- `industry_partner_heatmap.pdf`: Industry × partner interaction
- `pro_partner_residualized_boost.pdf`: Pro boost controlling for age+industry
- `judge_fan_agreement_by_subgroup.pdf`: Judge–fan correlation by industry/age
- `slope_by_age.pdf`: Does age predict improvement (W1→W3)?
- **Judge vs Fan impact (run `python3 judge_fan_impact.py`):**
  - `judge_fan_effect_comparison.pdf`: Divergence plot + bar chart
  - `judge_fan_impact_numbers.pdf`: Compact numbers panel
  - `age_judge_vs_fan_overlay.pdf`: Age effect overlay
  - `judge_fan_effect_ratio.pdf`: Judge/Fan effect ratio
