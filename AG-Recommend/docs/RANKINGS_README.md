# Factor Influence Rankings - Dancing with the Stars Analysis

## Overview

This document provides comprehensive rankings of factors influencing performance in Dancing with the Stars, analyzing both **judges' scores** and **fan votes** separately using a two-channel hierarchical Bayesian model.

---

## Key Questions Answered

1. **What factors most influence judges' scores vs. fan votes?**
2. **Do celebrity characteristics (age, gender, industry) affect judges and fans the same way?**
3. **How much do professional dancers matter?**
4. **Which variable types (demographics, performance, temporal) are most influential?**

---

## Ranking Types Generated

### **Ranking 1: Overall Influence on Judges' Scores**
- Lists all fixed effects ranked by absolute effect size
- Shows which factors judges care about most
- Includes 95% credible intervals for each effect

### **Ranking 2: Overall Influence on Fan Votes**
- Lists all fixed effects ranked by absolute effect size
- Shows which factors fans care about most
- Includes 95% credible intervals for each effect

### **Ranking 3: Comparative Analysis (Judges vs. Fans)**
- Compares how the same factors affect judges vs. fans
- Identifies factors where judges and fans agree/disagree
- Highlights rank differences to show divergent preferences

### **Ranking 4: By Variable Type**
- Groups factors into categories:
  - **Temporal**: Week effects, progression over time
  - **Performance**: Improvement, momentum effects
  - **Demographics**: Age, gender
  - **Background**: Industry/profession
- Shows average influence by category for judges and fans separately

### **Ranking 5: Industry-Specific Effects**
- Detailed breakdown of how different celebrity industries perform
- Separate rankings for judges and fans
- Identifies which industries have advantages/disadvantages

---

## How to Interpret Rankings

### Effect Size Interpretation
- **Positive effect**: Factor increases scores/votes
- **Negative effect**: Factor decreases scores/votes
- **Larger absolute value**: Stronger influence
- **95% CI excludes zero**: Statistically credible effect

### Example Interpretation
```
1. improve_judge_z
   Effect: 0.45 [95% CI: 0.38, 0.52]
   Absolute Effect: 0.45
```
This means: A 1 SD increase in judge score improvement from the previous week is associated with a 0.45 SD increase in current judge scores (strong positive effect).

---

## Random Effects Variance

The rankings also show variance components for:
- **Pro dancers** (pro_id): How much variation is due to professional dancer skill
- **Celebrities** (celebrity_id): How much variation is due to individual celebrity talent
- **Seasons** (season): How much variation is due to season-specific factors

**Higher SD = More important source of variation**

---

## Key Insights to Look For

1. **Performance vs. Demographics**: Do performance factors (improvement, week) matter more than demographics (age, gender)?

2. **Judges vs. Fans Divergence**: Where do judges and fans disagree most?
   - High rank difference = different priorities
   - Same sign but different magnitude = agree on direction but not strength
   - Opposite signs = fundamental disagreement

3. **Pro Dancer Correlation**: The model estimates correlation between pro effects on judges vs. fans
   - Positive correlation: Pros who help with judges also help with fans
   - Near zero: Pro effects are independent across channels
   - Negative correlation: Pros who help with judges hurt with fans (unlikely)

4. **Industry Effects**: Which celebrity backgrounds have systematic advantages?

---

## Files Generated

1. **judge_fixed_effects_ranking.csv**: Complete ranking for judges
2. **fan_fixed_effects_ranking.csv**: Complete ranking for fans
3. **judges_vs_fans_comparison.csv**: Side-by-side comparison
4. **factor_influence_comparison.png**: Coefficient plots for both channels
5. **judges_vs_fans_comparison.png**: Direct comparison visualization
6. **top_factors_ranking.png**: Bar chart of top 5 factors per channel

---

## Usage Instructions

### Step 1: Fit the Model
```r
source("brms_two_channel_model.R")
```

### Step 2: Generate Rankings
```r
source("brms_factor_rankings.R")
```

### Step 3: Create Visualizations
```r
source("brms_visualizations.R")
```

### Step 4: Review Results
- Check console output for detailed rankings
- Review CSV files for complete data
- Examine PNG files for visual summaries

---

## Statistical Notes

- All predictors are standardized (mean=0, SD=1) for comparability
- Effects are on standardized outcome scales
- 95% credible intervals from Bayesian posterior distributions
- Rankings use absolute effect sizes to capture magnitude regardless of direction
- Random effects show standard deviations, not individual effects

---

## Next Steps for Analysis

1. **Examine specific industries**: Which celebrity backgrounds perform best?
2. **Check pro dancer correlation**: Do pros affect judges and fans similarly?
3. **Temporal patterns**: Does week number matter? Do effects change over season?
4. **Interaction effects**: Consider adding interactions (e.g., age × gender)
5. **Model diagnostics**: Check MCMC convergence, posterior predictive checks
