# How to Quantify Accuracy in DWTS Predictions

When predicting a competition like *Dancing with the Stars*, "Accuracy" is not a single number. We use four complementary metrics to measure how well the model is performing.

## 1. Directional Accuracy (Pearson & Spearman Correlation)
This is the most important metric for ranking problems.
*   **What it measures**: If the model says Contestant A will beat Contestant B, how often is that correct?
*   **Scale**: -1.0 to +1.0 (Higher is better).
*   **Interpretation**: 
    *   **> 0.7**: Strong - The model has a very good grasp of the relative order.
    *   **0.4 - 0.7**: Moderate - (Our current Success Model is ~0.46) The model predicts the general "tier" of the celebrity (Top 3 vs Middle vs Bottom) but misses the exact order.
    *   **< 0.4**: Weak - The model is mostly guessing.

## 2. Average Error in "Places" (Mean Absolute Error - MAE)
*   **What it measures**: On average, how many ranks off is the prediction?
*   **Interpretation**: 
    *   Our current model has an MAE of **~2.9 places**.
    *   This means if we predict someone to finish in **5th place**, they usually finish between **2nd and 8th**.

## 3. Explanatory Power (R-Squared)
*   **What it measures**: What % of the outcome is explained by "Static Traits" (Age, Industry, Partner) vs. "Randomness/Talent"?
*   **Interpretation**: 
    *   Our current Success Model has an R² of **~0.21 (21%)**.
    *   This means 21% of your placement is "decided" before you even dance (based on who you are), while the remaining 79% is decided by performance, personality, and luck.

## 4. Winner Success Rate
*   **What it measures**: Did the model correctly identify the Season Winner?
*   **Interpretation**:
    *   Our model currently correctly identifies the winner in **~24% of seasons**. Since there are 10-15 contestants per season, a random guess would be ~7-10%, so the model is significantly better than chance.

---

### Comparison of Predictions
| Metric | Static Traits Only (Pre-Season) | With Early Scores (Week 3) |
| :--- | :--- | :--- |
| **Correlation** | ~0.46 | **~0.74** |
| **Rank Error (MAE)** | ~2.9 places | **~1.8 places** |
| **Variance Explained** | ~21% | **~55%** |

**Summary**: To "quantify accuracy," you should report a balance of these numbers. For a competition, **Spearman Correlation** is generally the gold standard because it handles rankings specifically.
