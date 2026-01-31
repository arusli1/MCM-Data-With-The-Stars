# DWTS Analysis: Methodology & Code Explanation

This document explains the analysis code, the statistical methods used, and how to interpret the generated results.

## 1. Overview of the Approach

We built three machine learning models to understand what drives success in *Dancing with the Stars*:
1.  **Success Model**: Predicts final placement (normalized 0-1 score).
2.  **Judge Model**: Predicts the average judge scores across all available weeks in the season.
3.  **Fan Model**: Predicts the average estimated fan vote share across all available weeks.

### Key Techniques
*   **Top-K Bucketing**: High-cardinality categories (like Home State or Partner) are grouped into the Top K most frequent, with the rest labeled as "Other".
*   **Random Forest Regressor**: A robust, non-linear model was used to capture complex interactions between features.
*   **Leave-One-Season-Out Cross-Validation (LOGO-CV)**: To ensure the model isn't "cheating" by memorizing seasons, we train on all seasons except one, predict the missing season, and repeat.

## 2. Interpreting the Charts

### Diverging Bar Charts (Green vs. Red)
We use a standardized visualization to show "Feature Importance":
*   **Right Side (Green Bars)**: These features **HELP** the contestant. (Positive Impact)
*   **Left Side (Red Bars)**: These features **HURT** the contestant. (Negative Impact)
*   **X-Axis (SHAP Value)**: The length of the bar represents the **magnitude** of the impact. Longer bars = deeper effect.

### Pro Boost Residuals
Using raw placement to judge partners is unfair because some partners get "easier" celebrities (younger, better dancers).
*   **Method**: We trained a model to predict success based *only* on Celebrity Traits + Performance.
*   **Residual**: We compared the *actual* success to this *predicted* success.
*   **Result**: If a partner's celebrities consistently do *better* than the model predicts, that partner adds "Active Value" (Positive Boost).

## 3. Code Structure (`run_analysis.py`)

*   **`load_and_preprocess()`**: Merges the main data with the fan vote estimates. Calculates averages across all available weeks (Judges and Fans) to use as descriptors.
*   **`feature_engineering()`**: Cleans up labels (e.g., categorizing "Actor", "Reality Star"). Normalizes placement to a 0-1 scale.
*   **`train_and_explain()`**:
    *   Trains the Random Forest models.
    *   Uses **SHAP (SHapley Additive exPlanations)** to calculate the contribution of each feature.
    *   Generates the **Green/Red Diverging Charts**.
    *   Exports `shap_comparison_signed.csv` with the raw numbers.
*   **`pro_boost_residualization()`**: Performs the special partner analysis described above.

## 5. Quantifying Accuracy

Accuracy in a ranking competition is measured using multiple metrics:
*   **Pearson & Spearman Correlation**: Measures the correctness of the *order*. Our success model is currently **~0.46** (Moderate).
*   **Mean Absolute Error (MAE)**: Measures how many places off we are. Currently **~2.9 places**.
*   **Winner Success Rate**: Correctly predicts the winner in **~24%** of seasons.
*   **R² (Variance Explained)**: Currently **~21%** based on static traits alone.

For a full breakdown of these metrics and how they compare to performance-based models, see [ACCURACY_METRICS.md](file:///Users/athenagao/Downloads/MCM-Data-With-The-Stars/analysis/ACCURACY_METRICS.md).
