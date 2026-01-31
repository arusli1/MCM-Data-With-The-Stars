# How to Quantify Accuracy in DWTS Predictions

## Technical Optimization (The Loss Function)
Since we are using **Random Forest Regressors**, the model does not "optimize" using a single global loss function like a Neural Network (Gradient Descent). Instead, it uses **Greedy Variance Reduction**.

*   **Criterion (Loss Function)**: The model uses **Mean Squared Error (MSE)**. 
*   **The Goal**: At every branch in a decision tree, the model searches for the exact "split" (e.g., "Age < 35") that minimizes the MSE of the predictions in the resulting groups.
*   **Averaging**: The final prediction is the average of 400 separate trees, which further reduces variance and prevents overfitting.

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

## 4. Winner Accuracy (Hit Rate)
This measures how often the model's highest-ranked predictions actually win the show.

### Pre-Season Model (Static Traits)
*   **Top-1 Accuracy**: **23.5%** (95% CI: [9.2%, 37.9%])
*   **Top-3 Accuracy**: **50.0%** (95% CI: [33.2%, 66.8%])

### High-Accuracy Model (Includes Week 1-2 Scores)
*   **Top-1 Accuracy**: **38.2%** (95% CI: [21.8%, 54.6%])
*   **Top-3 Accuracy**: **79.4%** (95% CI: [65.8%, 93.0%])

**Why this matters**: Adding just two weeks of performance data allows the model to capture "actual talent." Picking the winner in the Top 3 **~80% of the time** makes this a highly reliable forecasting tool once the season begins.

---

### Comparison of Predictions
| Metric | Static Traits Only (Pre-Season) | With Early Scores (Week 3) |
| :--- | :--- | :--- |
| **Correlation** | ~0.46 | **~0.74** |
| **Rank Error (MAE)** | ~2.9 places | **~1.8 places** |
| **Variance Explained** | ~21% | **~55%** |

**Summary**: To "quantify accuracy," you should report a balance of these numbers. For a competition, **Spearman Correlation** is generally the gold standard because it handles rankings specifically.
