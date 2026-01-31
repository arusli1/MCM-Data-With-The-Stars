# DWTS Factor Analysis: Final Rigorous Results

This analysis implements the **rigorous methodology** including target leakage prevention, robust SHAP handling, and geographic bucketing.

## Model Performance (Leave-One-Season-Out CV)

| Target                   | Pearson r | Spearman ρ | R²    |
|:-------------------------|:----------|:-----------|:------|
| Success (Placement)      | 0.458     | 0.456      | 0.208 |
| Judge Appeal (Avg Score) | 0.544     | 0.544      | 0.294 |
| Fan Appeal (Avg Share)   | 0.353     | 0.390      | 0.110 |

---

## NEW: Fans vs. Judges Comparison
We compared the relative impact of each variable on **Judges** (Scores) vs. **Fans** (Vote Share).

### Key Insight: The Age Paradox
- **Judges**: Heavily penalize older contestants (negative SHAP).
- **Fans**: Show a slight **positive** preference for older/more established celebrities.

![Fan vs Judge Discrepancy](/Users/athenagao/Downloads/MCM-Data-With-The-Stars/analysis/results/fan_vs_judge_discrepancy.png)

### Other Key Discrepancies:
- **TV Personalities**: Heavily favored by **Judges** more than Fans.
- **States like Louisiana**: Show significantly higher **Fan Appeal** relative to Judge Appeal.

**Detailed Comparison Data**: [fan_vs_judge_comparison.csv](file:///Users/athenagao/Downloads/MCM-Data-With-The-Stars/analysis/results/fan_vs_judge_comparison.csv)

---

## Key Findings

### 1. The "Age" Factor
Across all three models, **Age** is the single most dominant static trait. It consistently has a negative impact on success and judge scores, showing that older contestants face a measurable disadvantage in placement, even if they have fan support.

### 2. Standardized Visualizations
All results are presented as **Diverging Bar Charts**:
- **Green (Right)**: Positive impact (helps the contestant)
- **Red (Left)**: Negative impact (hurts the contestant)

````carousel
![Top Drivers of Success](/Users/athenagao/.gemini/antigravity/brain/82bf4c24-d144-4ea2-b5fd-9f96edcc7866/shap_Success_Placement.png)
<!-- slide -->
![Top Drivers of Judge Appeal](/Users/athenagao/.gemini/antigravity/brain/82bf4c24-d144-4ea2-b5fd-9f96edcc7866/shap_Judge_Appeal_Avg_Score.png)
<!-- slide -->
![Top Drivers of Fan Appeal](/Users/athenagao/.gemini/antigravity/brain/82bf4c24-d144-4ea2-b5fd-9f96edcc7866/shap_Fan_Appeal_Avg_Share.png)
````

### 3. Partner Impact
These charts isolate the influence of professional partners:

````carousel
![Partner Impact - Success](/Users/athenagao/.gemini/antigravity/brain/82bf4c24-d144-4ea2-b5fd-9f96edcc7866/shap_Success_Placement_partners.png)
<!-- slide -->
![Partner Impact - Judge Appeal](/Users/athenagao/.gemini/antigravity/brain/82bf4c24-d144-4ea2-b5fd-9f96edcc7866/shap_Judge_Appeal_Avg_Score_partners.png)
<!-- slide -->
![Partner Impact - Fan Appeal](/Users/athenagao/.gemini/antigravity/brain/82bf4c24-d144-4ea2-b5fd-9f96edcc7866/shap_Fan_Appeal_Avg_Share_partners.png)
````

## Data & Methodology
- **Signed SHAP Analysis**: We use the mean signed SHAP value to determine the direction and magnitude of feature effects.
- **No Target Leakage**: The Success model uses only static traits, ensuring the results explain "what helps a contestant win from the start."
- **Full Results**: [analysis_summary.md](file:///Users/athenagao/Downloads/MCM-Data-With-The-Stars/analysis/results/analysis_summary.md)
