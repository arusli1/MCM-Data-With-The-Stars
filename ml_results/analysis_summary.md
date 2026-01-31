# DWTS Machine Learning Analysis Summary

## Methods
We trained tuned XGBoost and Random Forest regression models to predict competition success, early judge scores, and early fan support using Leave-One-Season-Out (LOGO) cross-validation (N=421 contestants). Feature engineering focused on celebrity characteristics and early-window (Weeks 1-3) performance metrics. Model explainability was quantified using SHAP values.

## Results
*   **Success Score**: Achieving a Spearman correlation of 0.736 (RF) and 0.674 (XGB), models identified **Early Judge Mean**, **Early Fan Share**, and **Week 1 Fan Score** as the top predictors. Advanced Celebrity Age remains the primary negative driver.
*   **Early Judges (Spearman 0.445)**: Judges favor **Television/Acting backgrounds** and specific pro-partners like **Derek Hough**.
*   **Early Fans (Spearman 0.201)**: Fans are significantly driven by **Celebrity Age** (favoring younger) and high-profile pro-partner pairings.

## Robustness & Sensitivity
- **Stability**: Feature importance for success remains consistent when comparing a Week 1 only model vs. the Weeks 1-3 window, confirming that initial impressions are highly persistent throughout the competition.
- **Fan Noise Sensitivity**: Fan drivers (Age, Partner) remained the top predictors even when adding Gaussian noise ($\pm 10\%$) to inferred vote shares, indicating that the identified drivers are not artifacts of vote estimation noise.
- **Pro Boost**: Residualized analysis confirms **Derek Hough**, **Cheryl Burke**, and **Kym Johnson** provides the most consistent "boost" to their celebrities, outperforming baseline expectations.

## Comparison: Judges vs. Fans
Judges are more responsive to professional entertainment backgrounds, whereas fans show higher sensitivity to the celebrity's age and the specific professional dancer's reputation. Both groups are influenced by the "pro-partner effect," but fans weigh the social prestige of certain dancers more heavily in early voting rounds.
